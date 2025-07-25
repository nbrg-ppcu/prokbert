from transformers import PreTrainedTokenizerFast
from transformers.tokenization_utils_base import BatchEncoding
from transformers.tokenization_utils import PaddingStrategy

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from tokenizers.processors import TemplateProcessing


import torch
from typing import List, Dict, Any, Optional, Union


class DNATokenizerFast(PreTrainedTokenizerFast):
    def __init__(self, tokenizer_object: PreTrainedTokenizerFast = None,
                 kmer_size: int = 6, kmer_shift: int = 6,
                 model_max_length: int =512,
                 **kwargs):
        super().__init__(tokenizer_object=tokenizer_object, **kwargs)
        self.kmer_size = kmer_size
        self.kmer_shift = kmer_shift
        self.model_max_length = model_max_length


        # Ensure these are tracked so `save_pretrained` saves them
        self.init_kwargs["kmer_size"] = kmer_size
        self.init_kwargs["kmer_shift"] = kmer_shift

        self.cls_token_id = self.vocab['[CLS]']


    def pad(
            self,
            encoded_inputs: Union[Dict[str, Any], List[Dict[str, Any]]],
            padding: Union[bool, str, PaddingStrategy] = True,
            max_length: Optional[int] = None,
            pad_to_multiple_of: Optional[int] = None,
            return_attention_mask: Optional[bool] = True,
            return_tensors: Optional[str] = "pt",
            **kwargs
    ) -> BatchEncoding:
        pad_id = self.pad_token_id
        k = self.kmer_size
        s = self.kmer_shift

        # Normalize input structure
        is_dict_input = isinstance(encoded_inputs, dict)
        if is_dict_input and isinstance(encoded_inputs["input_ids"][0], int):
            encoded_inputs = [encoded_inputs]

        # Extract input_ids
        input_ids_batch = [ex["input_ids"] for ex in encoded_inputs]

        # Compute padded lengths
        padded_lens = [
            ((max(len(seq) - k, 0) + s - 1) // s) * s + k
            for seq in input_ids_batch
        ]

        if padding == "longest" or (padding is True and max_length is None):
            target_len = max(padded_lens)
        elif isinstance(padding, int):
            target_len = padding
        elif max_length is not None:
            target_len = max_length
        else:
            target_len = None

        batch_input_ids = []
        batch_attention_mask = []

        for input_ids in input_ids_batch:
            padded_len = target_len or len(input_ids)
            pad_len = max(0, padded_len - len(input_ids))
            padded = input_ids + [pad_id] * pad_len

            attention_mask = [
                0 if all(tok == pad_id for tok in padded[i:i + k]) else 1
                for i in range(0, len(padded) - k + 1, s)
            ]

            batch_input_ids.append(padded)
            batch_attention_mask.append(attention_mask)

        max_input_len = max(len(seq) for seq in batch_input_ids)
        max_mask_len = max(len(mask) for mask in batch_attention_mask)

        for i in range(len(batch_input_ids)):
            batch_input_ids[i] += [pad_id] * (max_input_len - len(batch_input_ids[i]))
            batch_attention_mask[i] += [0] * (max_mask_len - len(batch_attention_mask[i]))

        # Prepare output by preserving existing fields
        if is_dict_input:
            output = encoded_inputs[0].copy()  # single example
        else:
            keys = encoded_inputs[0].keys()
            batch_dict = {key: [ex[key] for ex in encoded_inputs] for key in keys}
            output = BatchEncoding(batch_dict)

        output["input_ids"] = batch_input_ids
        if return_attention_mask:
            output["attention_mask"] = batch_attention_mask

        if return_tensors == "pt":
            output = {k: torch.tensor(v, dtype=torch.long) for k, v in output.items()}
            return BatchEncoding(output)
        else:
            return BatchEncoding(output)


def create_prokbert_tokenizer(vocab: Optional[Dict[str, int]],
                              kmer_size: Optional[int] = 6,
                              kmer_shift: Optional[int] = 6,
                              max_seq_len: Optional[int] = 512):

    vocab = vocab if vocab is not None else {
        "[PAD]": 0,
        "[CLS]": 1,
        "[SEP]": 2,
        "[MASK]": 3,
        "[BOS]": 4,
        "[EOS]": 5,
        "[UNK]": 6,
        "A": 7,
        "C": 8,
        "T": 9,
        "G": 10
    }

    base_tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    base_tokenizer.pre_tokenizer = Split(pattern="", behavior="isolated")

    base_tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A",
        special_tokens=[
            ("[CLS]", vocab["[CLS]"]),
        ],
    )

    # Wrap it to e a PreTrainedTokenizerFast
    base_tokenizer = PreTrainedTokenizerFast(tokenizer_object=base_tokenizer)

    tokenizer = DNATokenizerFast(tokenizer_object=base_tokenizer, kmer_size=kmer_size,
                                 kmer_shift=kmer_shift,
                                 max_seq_len=max_seq_len)

    tokenizer.pad_token = "[PAD]"
    tokenizer.cls_token = "[CLS]"
    tokenizer.sep_token = "[SEP]"
    tokenizer.mask_token = "[MASK]"
    tokenizer.unk_token = "[UNK]"

    return tokenizer