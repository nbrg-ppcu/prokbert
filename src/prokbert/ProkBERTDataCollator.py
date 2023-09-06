import logging
from collections.abc import Mapping
from dataclasses import dataclass
from random import randint
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import torch

import numpy as np
from transformers import DataCollatorForLanguageModeling

logger = logging.getLogger(__name__)

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
    torch_token_dtype = torch.int16

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



