import random
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from random import randint
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import torch

import numpy as np
from transformers import DataCollatorForLanguageModeling


@dataclass
class ProkBERTDataCollator(DataCollatorForLanguageModeling):
    """
    Data collator for overlapping k-mers
    """

    #mask_to_left = 0, mask_to_right = 0
    #def __init__(self, mask_to_left = 0, mask_to_right = 0):
    #    self.mask_to_left = mask_to_left
    #    self.mask_to_right = mask_to_right
    #    super().__init__()
    mask_to_left: int = 0
    mask_to_right: int = 0
    replace_prob: float = 0.8
    random_prob: float = 0.1

    def print_params(self):
        
        print('Collator params: ')
        collator_params = '''\
Number of token mask to left to a mask token:   {0}
Number of token mask to right to a mask token:  {1}
1-Probability of restore a mask token:          {2}
Probablity of chaning to a random token:        {3}
mlm_probability:                                {4}
'''.format(self.mask_to_left, self.mask_to_right, self.replace_prob, self.random_prob, self.mlm_probability)

        print(collator_params)

    def set_mask_neightbourhood_params(self, mask_to_left = 0, mask_to_right = 0):
        '''Set how much tokens should be 
        '''
        self.mask_to_left = mask_to_left
        self.mask_to_right = mask_to_right   
        



    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        #print('Prepare overlapping mask tokens!')
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
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
        
        #print(masked_indices)
        #print(masked_indices.shape)
        masked_indices_shape = masked_indices.shape
        act_indeces = np.arange(masked_indices_shape[1])
        #print('Extending coordinates')
        for i in range(masked_indices_shape[0]):
            mask_coords = act_indeces[masked_indices[i]]
            new_mask_coords = []
            #print(mask_coords)
            for act_coord in mask_coords:
                index_start = max(1, act_coord - self.mask_to_left) #végeket nem maszkoljuk ki
                index_end = min (act_coord + self.mask_to_right + 1, masked_indices_shape[1]-1) #végeket nem maszkoljuk ki
                #print(index_start, index_end)
                new_range = list(range(index_start, index_end))
                new_mask_coords.extend(new_range)
            new_mask_coords = list(set(new_mask_coords))
            #print(mask_coords)
            #print(new_mask_coords)
            masked_indices[i][new_mask_coords]=True
            #print('__________________')
        #print(masked_indices)
        labels[~masked_indices] = -100
        #print('New labels')
        #print(labels)


        #masked_indeces_coord = torch.where(masked_indices)
        #print(masked_indeces_coord)
        #print(inputs[masked_indeces_coord])

        labels[~masked_indices] = -100  # We only compute loss on masked tokens

 
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, self.replace_prob)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        
        # 10% of the time, we replace masked input tokens with random word
        if self.random_prob > 0:
            indices_random = torch.bernoulli(torch.full(labels.shape, self.random_prob)).bool() & masked_indices & ~indices_replaced
            random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
            inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels



