import logging
from collections.abc import Mapping
from dataclasses import dataclass
from random import randint
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import torch

import numpy as np
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)




@dataclass
class ExpProkBERTDataCollator(DataCollatorForLanguageModeling):
    """
    Data collator for overlapping k-mers.

    Args:
        mask_to_left (int): Number of tokens masked to the left of a given token. Default is 0.
        mask_to_right (int): Number of tokens masked to the right of a given token. Default is 0.
        replace_prob (float): Probability of replacing a token. Default is 0.8.
        random_prob (float): Probability of changing a token to a random token. Default is 0.1.
    """
    mask_to_left: int = 0
    mask_to_right: int = 0
    replace_prob: float = 0.8
    random_prob: float = 0.1
    torch_token_dtype = torch.int64

    def __str__(self) -> str:
        """
        Returns:
            str: A formatted string representation of the collator parameters.
        """
        collator_params = '''\
Collator Parameters:
  Number of tokens masked to left:  {0}
  Number of tokens masked to right: {1}
  Probability of restoring a masked token: {2}
  Probability of changing to a random token: {3}
  MLM Probability: {4}
  Default token type: {5}
'''.format(self.mask_to_left, self.mask_to_right, self.replace_prob, self.random_prob, self.mlm_probability, self.torch_token_dtype)
        return collator_params

    def set_mask_neighborhood_params(self, mask_to_left: int = 0, mask_to_right: int = 0):
        """
        Set the number of tokens that should be masked to the left and right of a given masked token.

        Args:
            mask_to_left (int): Number of tokens to be masked to the left. Default is 0.
            mask_to_right (int): Number of tokens to be masked to the right. Default is 0.

        Raises:
            AssertionError: If mask_to_left or mask_to_right are not non-negative integers.
        """
        assert isinstance(mask_to_left, int) and mask_to_left >= 0, "mask_to_left should be a non-negative integer."
        assert isinstance(mask_to_right, int) and mask_to_right >= 0, "mask_to_right should be a non-negative integer."

        self.mask_to_left = mask_to_left
        self.mask_to_right = mask_to_right
        logger.info(f"Mask neighborhood parameters set to: mask_to_left={mask_to_left}, mask_to_right={mask_to_right}")

    def set_torch_token_dtype(self, torch_token_dtype=torch.int64):
        self.torch_token_dtype = torch_token_dtype

    def torch_mask_tokens(
        self,
        inputs: torch.Tensor,
        special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling:
        80% MASK, random_prob random, else original.
        """
        device = inputs.device
        # 1. Clone inputs to labels
        labels = inputs.clone().to(device)

        # 2. Create probability matrix
        probability_matrix = torch.full(labels.shape, self.mlm_probability, device=device)

        # 3. Mask out special tokens
        if special_tokens_mask is None:
            # build mask from tokenizer
            mask_list = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(mask_list, dtype=torch.bool, device=device)
        else:
            special_tokens_mask = special_tokens_mask.bool().to(device)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        # 4. Sample which tokens to mask
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # 5. Expand mask neighborhood
        bsz, seq_len = masked_indices.shape
        indices = torch.arange(seq_len, device=device)
        for i in range(bsz):
            active = indices[masked_indices[i]].tolist()
            neigh = []
            for idx in active:
                start = max(1, idx - self.mask_to_left)
                end = min(idx + self.mask_to_right + 1, seq_len - 1)
                neigh += list(range(start, end))
            masked_indices[i, list(set(neigh))] = True

        # 6. Set labels for non-masked tokens to -100
        labels[~masked_indices] = -100

        # 7. Replace 80% masked tokens with [MASK]
        replace_matrix = torch.full(labels.shape, self.replace_prob, device=device)
        indices_replaced = torch.bernoulli(replace_matrix).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 8. Replace random_prob masked tokens with random tokens
        if self.random_prob > 0:
            random_matrix = torch.full(labels.shape, self.random_prob, device=device)
            indices_random = torch.bernoulli(random_matrix).bool() & masked_indices & ~indices_replaced
            random_words = torch.randint(
                low=0,
                high=len(self.tokenizer),
                size=labels.shape,
                dtype=self.torch_token_dtype,
                device=device,
            )
            inputs[indices_random] = random_words[indices_random]

        # 9. Return inputs, labels
        return inputs, labels.to(dtype=torch.int64)


@dataclass
class ProkBERTDataCollator(DataCollatorForLanguageModeling):
    """
    Data collator for overlapping k-mers.

    Args:
        mask_to_left (int): Number of tokens masked to the left of a given token. Default is 0.
        mask_to_right (int): Number of tokens masked to the right of a given token. Default is 0.
        replace_prob (float): Probability of replacing a token. Default is 0.8.
        random_prob (float): Probability of changing a token to a random token. Default is 0.1.
    """
    mask_to_left: int = 0
    mask_to_right: int = 0
    replace_prob: float = 0.8
    random_prob: float = 0.1
    torch_token_dtype = torch.int64

    def __str__(self) -> str:
        """
        Returns:
            str: A formatted string representation of the collator parameters.
        """
        collator_params = '''\
Collator Parameters:
  Number of tokens masked to left:  {0}
  Number of tokens masked to right: {1}
  Probability of restoring a masked token: {2}
  Probability of changing to a random token: {3}
  MLM Probability: {4}
  Default token type: {5}
'''.format(self.mask_to_left, self.mask_to_right, self.replace_prob, self.random_prob, self.mlm_probability, self.torch_token_dtype)
        return collator_params

    def set_mask_neighborhood_params(self, mask_to_left: int = 0, mask_to_right: int = 0):
        """
        Set the number of tokens that should be masked to the left and right of a given masked token.

        Args:
            mask_to_left (int): Number of tokens to be masked to the left. Default is 0.
            mask_to_right (int): Number of tokens to be masked to the right. Default is 0.

        Raises:
            AssertionError: If mask_to_left or mask_to_right are not non-negative integers.
        """
        assert isinstance(mask_to_left, int) and mask_to_left >= 0, "mask_to_left should be a non-negative integer."
        assert isinstance(mask_to_right, int) and mask_to_right >= 0, "mask_to_right should be a non-negative integer."

        self.mask_to_left = mask_to_left
        self.mask_to_right = mask_to_right
        logger.info(f"Mask neighborhood parameters set to: mask_to_left={mask_to_left}, mask_to_right={mask_to_right}")

    def set_torch_token_dtype(self, torch_token_dtype=torch.int16):
        self.torch_token_dtype = torch_token_dtype


    def torch_mask_tokens(self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.

        Args:
            inputs (torch.Tensor): Input tensor to be masked.
            special_tokens_mask (Optional[torch.Tensor]): Tensor representing special tokens (e.g., [CLS], [SEP]). Default is None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of input tensor with masked values and labels tensor.
        """
        import os
        #subinputs = inputs[0:10,0:10]
        #myrank = int(os.environ["RANK"])
        #print(f'RANK: {myrank}; dataset_sample: {subinputs}', flush=True)


        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        masked_indices_shape = masked_indices.shape
        act_indeces = np.arange(masked_indices_shape[1])
        for i in range(masked_indices_shape[0]):
            mask_coords = act_indeces[masked_indices[i]]
            new_mask_coords = []
            for act_coord in mask_coords:
                index_start = max(1, act_coord - self.mask_to_left)
                index_end = min (act_coord + self.mask_to_right + 1, masked_indices_shape[1]-1)
                new_range = list(range(index_start, index_end))
                new_mask_coords.extend(new_range)
            new_mask_coords = list(set(new_mask_coords))

            masked_indices[i][new_mask_coords]=True
        labels[~masked_indices] = -100

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, self.replace_prob)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # X% of the time, we replace masked input tokens with random word
        if self.random_prob > 0:
            indices_random = torch.bernoulli(torch.full(labels.shape, self.random_prob)).bool() & masked_indices & ~indices_replaced
            random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=self.torch_token_dtype)
            inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        labels = labels.to(dtype=torch.int64)

        return inputs, labels


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

    Batches are filled / padded with “virtual genes” with:
      - input_ids:       [CLS, SEP, 0, ..., 0]
      - attention_mask:  1 for the first two positions, 0 elsewhere
      - token_type_ids:  0 (or a configured value)

    Notes
    -----
    * This collator expects that tokenization is already done at the sequence level (per gene).
    * It does not create MLM labels; pair with a separate MLM collator if needed.
    * `return_tensors="pt"` only.
    """

    tokenizer: PreTrainedTokenizerBase
    mlm: bool = False
    mlm_probability: float = 0.15
    mask_replace_prob: float = 0.8
    random_replace_prob: float = 0.1
    return_tensors: str = "pt"
    torch_token_dtype = torch.int64
    attention_mask: int = 0
    token_type_id_mask: int = 0
    generator: Optional[Any] = None

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

    # tokenisation should already done at sequence level
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
