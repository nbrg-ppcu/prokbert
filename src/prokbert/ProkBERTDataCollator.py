from typing import Any, AnyStr, Dict, List, Optional, Tuple, Union, Set

import logging
from dataclasses import dataclass

import torch
import numpy as np
from transformers import DataCollatorForLanguageModeling, BatchEncoding, PreTrainedTokenizer, PreTrainedTokenizerFast


logger = logging.getLogger(__name__)


class VarLenDataCollatorWithPadding:
    def __init__(
        self,
        tokenizer: Union[PreTrainedTokenizerFast, PreTrainedTokenizer],
        max_output_seq_len: int, # TODO: could be inferred from the tokenizer
        min_length: int = 0,
        distribution: str = "uniform",
        seed: int = 42,
        distribution_kwargs: Optional[Dict] = None,
        special_token_ids_to_mask: Optional[Set[int]] = None,
        truncation_probability: float = 0.8,
    ):
        self.tokenizer = tokenizer

        assert min_length <= max_output_seq_len, "min_length should be smaller than max_length"
        assert 0 < max_output_seq_len, "max_length should be larger than 0"

        self.min_length = max(0,min_length)
        self.max_output_seq_len = max_output_seq_len

        # Internal max length that takes the added special tokens into account
        self._max_seq_len = max_output_seq_len - tokenizer.num_special_tokens_to_add(pair=False)

        # Make sure all ID-s to mask are integers
        if special_token_ids_to_mask is not None:
            assert all(type(x) is int for x in special_token_ids_to_mask), "special_token_ids_to_mask must contain only integer IDs."
        self.special_token_ids_to_mask = special_token_ids_to_mask

        self. truncation_probability = max(0.0, truncation_probability)

        self.rng = np.random.default_rng(seed=seed)
        kw = distribution_kwargs if distribution_kwargs is not None else {}

        # Create internal generator with the given distribution
        if distribution == "uniform":
            self._generator = lambda n: self.rng.uniform(low=min_length, high=self._max_seq_len, size=n)
        elif distribution == "normal":
            center = (min_length + self._max_seq_len) / 2
            sigma = (self._max_seq_len - min_length) / 4
            self._generator = lambda n: self.rng.normal(loc=center, scale=sigma, size=n)
        elif distribution == "exponential":
            scale = kw.get("scale", 100 )
            self._generator = lambda n: self.rng.exponential(scale=scale, size=n)
        else:
            raise ValueError(f"Unknown distribution: {distribution}. Possible distributions: 'uniform', 'normal', 'exponential'")


    def _sample_seq_start(self, features: List[Dict[AnyStr,Any]], truncate_samples: List[bool] ) -> List[int]:
        input_id_lens = []
        for f in features:
            current_len = len(f['input_ids'])
            assert self.min_length <= current_len, (f"All input_ids must have at least min_length {self.min_length} tokens to choose from,"
                                                    f"got sequence with length: {current_len}.")
            input_id_lens.append(current_len)

        random_lengths =  [self.rng.integers(low=0, high=max(0, input_id_len - self.max_output_seq_len), endpoint=True).astype(int)
                for input_id_len in input_id_lens]

        return [random_length if trunc else 0 for trunc, random_length in zip(truncate_samples, random_lengths) ]

    # Helper function to clip the values to the right len and set their type
    def _sample_seq_end(self, features: List[Dict[AnyStr,Any]],  truncate_samples: List[bool] ) -> List[int]:
        # Create a random number for each input_id, make sure they are integers
        sample_lens = self._generator(len(features)).astype(int)

        # Make sure the sample end is smaller than the max_len and the sequences length
        random_lengths = [min(max(self.min_length, min(sample_len, self._max_seq_len)), len(f['input_ids']))
                         for sample_len, f in zip(sample_lens, features)]

        return [random_length if trunc else min(len(f['input_ids']), self._max_seq_len)
                for trunc, random_length, f in zip(truncate_samples, random_lengths, features) ]


    def __call__(self, features: List[Dict[AnyStr,Any]], return_tensors: Optional[str] = "pt") -> Optional[BatchEncoding] :
        if len(features) == 0:
            return None
        if 0.0 < self.truncation_probability:
            truncate_samples = [ self.truncation_probability >= prob for prob in self.rng.uniform(low=0.0, high=1.0, size=len(features))]
        else:
            truncate_samples = [ False for _ in range(len(features))]

        seq_starts = self._sample_seq_start(features=features, truncate_samples=truncate_samples)

        # Generate random seq_ends between the boundaries
        seq_ends = self._sample_seq_end(features=features, truncate_samples=truncate_samples)


        # Extract labels if available the 'labels' column takes precedence but try with 'y' too
        labels = None
        if "labels" in features[0]:
            labels = [f["labels"] for f in features]
        elif "label" in features[0]:
            labels = [f["label"] for f in features]
        elif "y" in features[0]:
            labels = [f["y"] for f in features]

        # Truncate the input_ids based on the seq_ends
        input_ids = [{'input_ids': self.tokenizer.build_inputs_with_special_tokens(f["input_ids"][start:end])}
                     for f, start, end in zip(features, seq_starts, seq_ends)]

        # Pad input_ids
        batch = self.tokenizer.pad(input_ids,
                                   padding='longest',
                                   max_length=self.max_output_seq_len,
                                   return_tensors=return_tensors)

        if self.special_token_ids_to_mask is not None:
            mask = batch['input_ids']

            if isinstance(mask, torch.Tensor):
                # Create a tensor holding the set's values
                special_ids = torch.tensor(list(self.special_token_ids_to_mask), device=mask.device, dtype=mask.dtype)
                batch['attention_mask'] = torch.isin(elements=mask, test_elements=special_ids, invert=True).long()
            else: # List of lists version
                batch['attention_mask'] = [
                    [0 if tok in self.special_token_ids_to_mask else 1 for tok in seq]
                    for seq in mask
                ]

        # Add labels if present in the dataset
        if labels is not None:
            if return_tensors == "pt":
                batch["labels"] = torch.tensor(labels)
            else:
                batch["labels"] = labels

        return batch



class VarLenDataCollatorForMaskedLanguageModeling(VarLenDataCollatorWithPadding):
    def __call__(self, features: List[Any], return_tensors: Optional[str] = "pt") -> BatchEncoding:
        seq_starts = self._sample_seq_start(features=features)

        # Generate random seq_ends between the boundaries
        seq_ends = self._sample_seq_end(features=features)

        # Sample the input_ids
        input_ids = [{'input_ids': f["input_ids"][start:end]} for f, start, end in zip(features, seq_starts, seq_ends)]
        # Pad input_ids
        batch = self.tokenizer.pad(input_ids, padding='longest', max_length=self.max_output_seq_len, return_tensors=return_tensors)

        # To handle the weighted loss if the label distribution is in the ds
        if isinstance(features[0], dict) and "labels" in features[0]:
            labels = [f["labels"][start: end] for f, start, end in zip(features, seq_starts, seq_ends)]
            # what should be the distribution here? [0.0, 0.0, 0.0, 0.0] I guess?
            zero_dist = [0.0, 0.0, 0.0,0.0]
            # Pad labels to the same length as the corresponding padded input_id is
            labels[:] = [lbl + [zero_dist] * (len(f) - len(lbl)) for lbl, f in zip(labels, batch['input_ids'])]

            if 'pt' == return_tensors:
                batch["labels"] = torch.tensor(labels)
            else:
                batch["labels"] = labels
        else:         # Create target labels if they were missing from the dataset
            batch['labels'] = batch['input_ids'] # Since we copy this will be tensor or list depending on return_tensors

        return batch


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



