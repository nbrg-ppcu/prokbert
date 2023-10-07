import logging
from collections.abc import Mapping
from dataclasses import dataclass
from random import randint, uniform
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import torch
import random
import numpy as np
from transformers import DataCollatorForLanguageModeling

logger = logging.getLogger(__name__)

seed_value = 860500 
#seed_value = randint(0,10000)
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)


def get_nr_tokens_to_mask(kmer, shift, Nchar_to_mask):
    ''' Calculating how many tokens should be mask to left and to right if position is masked'''

    Nr_token_to_mask = int(np.ceil( (Nchar_to_mask+kmer)/shift)-1)
    #print(Nr_token_to_mask)

    if Nr_token_to_mask % 2==0:
        Nr_token_to_mask_left = Nr_token_to_mask //2 
        Nr_token_to_mask_right = Nr_token_to_mask //2 -1
    else:
        Nr_token_to_mask_left = (Nr_token_to_mask-1) //2
        Nr_token_to_mask_right = (Nr_token_to_mask-1) //2
    return Nr_token_to_mask_left, Nr_token_to_mask_right, Nr_token_to_mask


def sample_contiguous(matrix: torch.Tensor, N: int) -> torch.Tensor:
    """
    Samples a contiguous part of each row of the input matrix of length N-1 and adds a first column of 2s.
    
    Args:
    - matrix (torch.Tensor): The input matrix.
    - N (int): The length of the contiguous part to sample from each row plus the first column.

    Returns:
    - torch.Tensor: A matrix with the same number of rows as the input matrix and N columns.
    """
    
    # Ensure N is not greater than the number of columns in the matrix
    if N > matrix.shape[1] + 1:
        raise ValueError(f"N ({N}) should not be greater than the number of columns in the matrix ({matrix.shape[1]}) plus 1")
    start_indices = torch.randint(0, matrix.shape[1] - N + 2, (matrix.shape[0],))    
    sampled_matrix = torch.stack([row[start: start + N - 1] for row, start in zip(matrix, start_indices)])
        # Add a first column of 2s
    sampled_matrix_with_first_col = torch.cat([torch.full((sampled_matrix.shape[0], 1), 2), sampled_matrix], dim=1)
    
    return sampled_matrix_with_first_col


@dataclass
class ProkBERTDataCollatorRand(DataCollatorForLanguageModeling):
    mask_to_left: int = 0
    mask_to_right: int = 0
    replace_prob: float = 0.8
    random_prob: float = 0.1
    torch_token_dtype = torch.long

    minL: int = 60 # Minimum length
    replace_to_padding_token: float = 0.005
    replace_prob_min: float = 0.8
    replace_prob_max: float = 0.85

    random_prob_min: float = 0.01
    random_prob_max: float = 0.02

    mask_chars_min: int = 2
    mask_chars_max: int = 2
    mlm_rate_min: float = 0.12
    mlm_rate_max: float = 0.15
    mlm_probability: float = 0.05

    change_mlm_parameter_prob = 0.000007

    def __str__(self) -> str:
        """
        Returns:
            str: A formatted string representation of the collator parameters.
        """
        collator_params = '''\
    Collator Parameters:
    Number of tokens masked to left: {0}
    Number of tokens masked to right: {1}
    Probability of masked token: {2}
    Probability of changing to a random token: {3}
    MLM Probability: {4}
    Default token type: {5}
    Minimum length: {6}
    Probability to replace to padding token: {7}
    Minimum replace probability: {8}
    Maximum replace probability: {9}
    Minimum random probability: {10}
    Maximum random probability: {11}
    Minimum masked characters: {12}
    Maximum masked characters: {13}
    Minimum MLM rate: {14}
    Maximum MLM rate: {15}
    Probability to change MLM parameter: {16}
    '''.format(self.mask_to_left, self.mask_to_right, self.replace_prob, self.random_prob, 
            self.mlm_probability, self.torch_token_dtype, self.minL, self.replace_to_padding_token,
            self.replace_prob_min, self.replace_prob_max, self.random_prob_min, self.random_prob_max,
            self.mask_chars_min, self.mask_chars_max, self.mlm_rate_min, self.mlm_rate_max,
            self.change_mlm_parameter_prob)
        return collator_params

    def set_parameters(self):
        
        print('Setting and sampling a new set of parameters: ')
        kmer = self.tokenizer.tokenization_params['kmer']
        shift = self.tokenizer.tokenization_params['shift']
        nr_chars_to_mask = randint(self.mask_chars_min, self.mask_chars_max)

        mask_to_left, mask_to_right, nr_masktokens =  get_nr_tokens_to_mask(kmer,shift,  nr_chars_to_mask)

        #print(f'kmer: {kmer}')
        #print(f'kmer: {shift}')
        #print(f'kmer: {nr_chars_to_mask}')
        #print(f'mask_to_left: {mask_to_left}')
        #print(f'mask_to_right: {mask_to_right}')
        #print(f'nr_masktokens: {nr_masktokens}')

        self.mask_to_left = mask_to_left
        self.mask_to_right = mask_to_right

        mlm_target_prob = uniform(self.mlm_rate_min, self.mlm_rate_max)
        #print(mlm_target_prob)
        self.mlm_probability = mlm_target_prob/nr_masktokens
        #print('self.mlm_probability ', self.mlm_probability )

        self.replace_prob = uniform(self.replace_prob_min, self.replace_prob_max)
        self.random_prob = uniform(self.random_prob_min, self.random_prob_max)

        print(self.__str__())


    def set_torch_token_dtype(self, torch_token_dtype=torch.long):
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

        if random.random() <= self.change_mlm_parameter_prob:
            self.set_parameters()

        Ncols = inputs.shape[1]
        act_min_L = int(Ncols/1.2)

        #L = random.randint(self.minL, int(self.minL*3))        
        L = random.randint(act_min_L, Ncols)

        inputs = sample_contiguous(inputs, L)
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


    def collate_batch(self, features: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        if isinstance(features[0], dict):
            keys = list(features[0].keys())
            values = list(features[0].values())
            if isinstance(values[0], torch.Tensor):
                inputs = [f[keys[0]] for f in features]
            else:
                raise ValueError(f"Unsupported type for first value in dictionary: {type(values[0])}")
        else:
            inputs = features

        batch_size = len(features)
        max_length = max([len(feature) for feature in features])

        # Padding
        padded_inputs = torch.zeros((batch_size, max_length), dtype=self.torch_token_dtype)
        for i, feature in enumerate(features):
            if isinstance(feature, dict):
                feature = feature[keys[0]]
            padded_inputs[i, : len(feature)] = torch.tensor(feature, dtype=self.torch_token_dtype)

        # Create attention mask
        #attention_mask = (padded_inputs != 0 ).float()
        #attention_mask = (inputs != 0) 
        #attention_mask3 = inputs == 3
        attention_mask = (inputs > 3) |  (inputs == 2) | (inputs == 1)
        attention_mask = attention_mask.float()
        # Mask tokens
        inputs, labels = self.mask_tokens(padded_inputs)
        return {"input_ids": inputs, "labels": labels, "attention_mask": attention_mask}
    






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


    def collate_batch(self, features: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        if isinstance(features[0], dict):
            keys = list(features[0].keys())
            values = list(features[0].values())
            if isinstance(values[0], torch.Tensor):
                inputs = [f[keys[0]] for f in features]
            else:
                raise ValueError(f"Unsupported type for first value in dictionary: {type(values[0])}")
        else:
            inputs = features

        batch_size = len(features)
        max_length = max([len(feature) for feature in features])

        # Padding
        padded_inputs = torch.zeros((batch_size, max_length), dtype=self.torch_token_dtype)
        for i, feature in enumerate(features):
            if isinstance(feature, dict):
                feature = feature[keys[0]]
            padded_inputs[i, : len(feature)] = torch.tensor(feature, dtype=self.torch_token_dtype)

        # Create attention mask
        #attention_mask = (padded_inputs != 0 ).float()
        #attention_mask = (inputs != 0) 
        #attention_mask3 = inputs == 3
        attention_mask = (inputs > 3) |  (inputs == 2) | (inputs == 1)


        attention_mask = attention_mask.float()

        # Mask tokens
        inputs, labels = self.mask_tokens(padded_inputs)

        return {"input_ids": inputs, "labels": labels, "attention_mask": attention_mask}

