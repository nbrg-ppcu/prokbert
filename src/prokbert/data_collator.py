from typing import Any, Optional, Tuple

import logging
from dataclasses import dataclass
import torch
from transformers import PreTrainedTokenizerBase


logger = logging.getLogger(__name__)


@dataclass
class DataCollatorForGenomeNetwork:
    """
    Collates a batch of genome examples where each example contains a 2D matrix per field:
      - input_ids:       [batch size / genoms, genes, sequences]
      - attention_mask:  [batch size / genoms, genes, sequences]
      - token_type_ids:  [batch size / genoms, genes, sequences] (optional)

    It pads all examples to `max_gene_len`, `max_seq_len` along genes and tokens, respectively, and stacks
    the batch into tensors of shape:
      - input_ids:       [batch size, max_gene_len, max_seq_len]
      - attention_mask:  [batch size, max_gene_len, max_seq_len]
      - token_type_ids:  [batch size, max_gene_len, max_seq_len] (if present)

    Furthermore, it can prepare masked language modeling (MLM) targets. This collator
    defaults to MLM (masked language modeling) but can be configured to perform
    causal language modeling (CLM) by setting `mlm` to `False`. The resulting tensors are of shape:
        - input_ids:       [batch size, max_gene_len, max_seq_len]
        - attention_mask:  [batch size, max_gene_len, max_seq_len]
        - token_type_ids:  [batch size, max_gene_len, max_seq_len] (if present)
        - labels:          [batch size, max_gene_len, max_seq_len] (if `mlm` is True)
        - labels_mask:     [batch size, max_gene_len] (if `mlm` is True) - indicates which genes are masked

    Batches are filled / padded with “virtual genes” with if needed:
      - input_ids:       [CLS, SEP, 0, ..., 0]
      - attention_mask:  1 for the first two positions, 0 elsewhere
      - token_type_ids:  0 (or a configured value)

    Notes
    -----
    * This collator expects that tokenization is already done at the sequence level (per gene).
    * `return_tensors="pt"` only.
    """

    tokenizer: PreTrainedTokenizerBase
    mlm: bool = False
    mlm_probability: float = 0.15
    mask_replace_prob: float = 0.8
    random_replace_prob: float = 0.1
    return_tensors: str = "pt"
    torch_token_dtype: torch.dtype = torch.int64
    attention_mask: int = 0
    token_type_id_mask: int = 0
    generator: Optional[Any] = None
    attention_mask_genome: bool = True # create attention mask for genome (2D)

    def __post_init__(self):
        self.pad_token_id: int = self.tokenizer.pad_token_id # type: ignore[arg-type]
        self.cls_token_id: int = self.tokenizer.cls_token_id # type: ignore[arg-type]
        self.sep_token_id: int = self.tokenizer.sep_token_id # type: ignore[arg-type]

        if self.mlm:
            if self.tokenizer.mask_token is None:
                raise ValueError(
                    "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                    "You should pass `mlm=False` to train on causal language modeling instead."
                )
            self.mask_token_id = self.tokenizer.mask_token_id
            if self.mlm_probability is None or self.mlm_probability < 0 or self.mlm_probability > 1:
                raise ValueError("mlm_probability should be between 0 and 1.")
            self.mlm_probability = float(self.mlm_probability)
        if self.mask_replace_prob + self.random_replace_prob > 1:
            raise ValueError("The sum of mask_replace_prob and random_replace_prob should not exceed 1")
        if self.mask_replace_prob < 0 or self.mask_replace_prob > 1:
            raise ValueError("mask_replace_prob should be between 0 and 1.")
        if self.random_replace_prob < 0 or self.random_replace_prob > 1:
            raise ValueError("random_replace_prob should be between 0 and 1.")

    # tokenisation should be already done at sequence level
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:

        max_gene_len = max(m["input_ids"].shape[0] for m in features)
        max_seq_len = max(m["input_ids"].shape[1] for m in features)
        pad_rows = [max_gene_len - m["input_ids"].size(0) for m in features]

        input_ids = [self.pad_sequence(m["input_ids"], max_gene_len, max_seq_len, self.pad_token_id) for m in features]
        attention_mask = [self.pad_sequence(m["attention_mask"], max_gene_len, max_seq_len, self.attention_mask) for m in features]
        token_type_ids = [self.pad_sequence(m["token_type_ids"], max_gene_len, max_seq_len, self.token_type_id_mask) for m in features]

        input_ids = [self.create_virtual_gene(input_id, pad_row) for (input_id, pad_row) in zip(input_ids, pad_rows)]
        attention_mask = [self.create_virtual_gene(mask, pad_row, is_attn=True) for (mask, pad_row) in zip(attention_mask, pad_rows)]
        batch = {
            "input_ids": torch.stack([input_id for input_id in input_ids], dim=0) ,
            "attention_mask": torch.stack([mask for mask in attention_mask], dim=0) ,
            "token_type_ids": torch.stack([token_type_id for token_type_id in token_type_ids], dim=0)
        }
        if self.attention_mask_genome:
            batch["attention_mask_genome"] = self.create_attention_mask_genome(batch["attention_mask"])
        if self.mlm:
            batch["input_ids"], batch["labels"], batch["labels_mask"] = self.mask_genes(batch["input_ids"])
        return batch

    def pad_sequence(self, input, max_gene_len: int, max_seq_len: int, value: int):
        return torch.nn.functional.pad(input, (0, max_seq_len - input.size(1), 0, max_gene_len - input.size(0)), value=value)

    def create_virtual_gene(self, input: torch.Tensor, pad_rows: int, is_attn: bool = False):
        if not pad_rows:
            return input
        if not is_attn:
            input[-pad_rows:, 0] = self.cls_token_id
            input[-pad_rows:, 1] = self.sep_token_id
        else:
            input[-pad_rows:, 0] = 1
            input[-pad_rows:, 1] = 1
        return input

    def create_attention_mask_genome(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Create attention mask at genome level (2D) from attention mask for genes (3D).

        Virtual genes have attention mask of all 0s, except for CLS and SEP tokens (both has 1 attention mask value),
        so we create a genome-level attention mask where virtual genes are masked out.
        1 indicates the gene is NOT a virtual gene, 0 indicates it is a virtual gene.

        The resulting tensor is of shape:
            - attention_mask_genome: [batch size, max_gene_len]
        """
        return (~(attention_mask.sum(dim=-1) == 2)).int()

    def mask_genes(
            self,
            inputs: torch.Tensor,
            special_tokens_mask: Optional[Any] = None
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare masked genes inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        labels = inputs.clone()
        batch_size, genes, seq_len = labels.shape

        # sample a few genes in each genome for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape[:-1], self.mlm_probability)

        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
                for gene in labels.tolist() for val in gene
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
            special_genes_mask = special_tokens_mask.all(dim=1).view(batch_size, -1)
        else:
            special_tokens_mask = special_tokens_mask.bool()
            special_genes_mask = special_tokens_mask.all(dim=1).view(batch_size, -1)

        # never mask virtual genes
        probability_matrix.masked_fill_(special_genes_mask, value=0.0)
        # sample genes to mask
        masked_genes_indices = torch.bernoulli(probability_matrix).bool()

        labels[~masked_genes_indices] = -100 # only compute loss on masked tokens

        # mask_replace_prob% of the time, we replace genes tokens with tokenizer.mask_token ([MASK])
        gene_indices_replaced = (
            torch.bernoulli(torch.full(labels.shape[:-1], 0.8)).bool()
            & masked_genes_indices
        )
        extended_gene_indices_replaced = gene_indices_replaced.unsqueeze(-1).expand(batch_size, genes, seq_len)
        indices_replaced = extended_gene_indices_replaced & ~special_tokens_mask.view(batch_size, genes, seq_len)

        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        if self.mask_replace_prob == 1 or self.random_replace_prob == 0:
            return inputs, labels

        remaining_prob = 1 - self.mask_replace_prob
        # scaling the random_replace_prob to the remaining probability for example if
        # mask_replace_prob = 0.8 and random_replace_prob = 0.1,
        # then random_replace_prob_scaled = 0.1 / 0.2 = 0.5
        random_replace_prob_scaled = self.random_replace_prob / remaining_prob

        # random_replace_prob% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape[:-1], random_replace_prob_scaled)).bool()
            & masked_genes_indices
            & ~gene_indices_replaced
        )
        extended_indices_random = indices_random.unsqueeze(-1).expand(batch_size, genes, seq_len)
        indices_random_replaced = extended_indices_random & ~special_tokens_mask.view(batch_size, genes, seq_len)

        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random_replaced] = random_words[indices_random_replaced]

        # labels mask (masked_genes_indices) signals which genes are masked
        return inputs, labels, masked_genes_indices


if __name__ == "__main__":
    import random
    from datasets import Dataset
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    model_name = 'neuralbioinfo/prokbert-mini'
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print("PAD token:", tokenizer.pad_token, "->", tokenizer.pad_token_id)
    print("CLS token:", tokenizer.cls_token, "->", tokenizer.cls_token_id)
    print("SEP token:", tokenizer.sep_token, "->", tokenizer.sep_token_id)
    print("MASK token:", tokenizer.mask_token, "->", tokenizer.mask_token_id)
    print("UNK token:", tokenizer.unk_token, "->", tokenizer.unk_token_id)
    print("VOCAB SIZE:", tokenizer.vocab_size, len(tokenizer))
    def random_gene_sequence(low = 10, high = 20):
        n = random.randint(low, high)
        return "".join(random.choice("ACGT") for _ in range(n))

    def create_random_genome_dataset(
            dataset_num=100,
            gene_per_genom_low=2,
            gene_per_genom_high=5,
            gene_seq_low=10,
            gene_seq_high=20
    ):
        genoms = {"genom": [], "gene_nums": [], "sequences": []}
        for i in range(1, dataset_num + 1):

            gene_nums = []
            gene_sequences = []

            n = random.randint(gene_per_genom_low, gene_per_genom_high)
            for j in range(n):

                gene_sequence = random_gene_sequence(gene_seq_low, gene_seq_high)
                gene_nums.append(j)
                gene_sequences.append(gene_sequence)

            genoms["genom"].append(i)
            genoms["gene_nums"].append(gene_nums)
            genoms["sequences"].append(gene_sequences)

        return genoms


    genoms = create_random_genome_dataset(
        dataset_num=100,
        gene_per_genom_low=4,
        gene_per_genom_high=7,
        gene_seq_low=10,
        gene_seq_high=15
    )

    dataset = Dataset.from_dict(genoms)
    tokenized_dataset = [tokenizer(genom["sequences"], padding=True, return_tensors="pt") for genom in dataset.to_list()]
    dataset.remove_columns(["sequences", "gene_nums", "genom"])

    data_collator = DataCollatorForGenomeNetwork(
        tokenizer,
        mlm=True,
        mlm_probability=0.7,
        mask_replace_prob=0.6,
        random_replace_prob=0.4
    )
    loader = DataLoader(tokenized_dataset, batch_size=2, collate_fn=data_collator)
    for batch in loader:
        print({k: v.shape for k, v in batch.items()})
        break