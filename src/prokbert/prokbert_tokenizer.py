# coding=utf-8
# Based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/tokenization_bert.py


# ProkBERT tokenizer stuff

import collections
import os
import unicodedata
from typing import List, Optional, Tuple, Union
from copy import deepcopy
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils import _is_control, _is_punctuation, _is_whitespace
from transformers.utils import logging

# These utils contains the tools needed by the ProkBERT tokenizer

from config_utils import *
from sequtils import *

import logging as logger

#logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

# models prokbert-mini-k6s1, prokbert-large-k6s2, prokbert-large-k6s1


PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "prokbert-mini-k6s1": "prokbert-base-dna6/vocab.txt",
        "prokbert-large-k6s1": "prokbert-base-dna6/vocab.txt",
        "prokbert-large-k6s2": "prokbert-base-dna6/vocab.txt"
    }
}


PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "prokbert-mini-k6s1": 1024,
    "prokbert-large-k6s1": 1024,
    "prokbert-large-k6s2": 1024
}

PRETRAINED_INIT_CONFIGURATION = {
    "prokbert-mini-k6s1": {"do_upper_case": True},
    "prokbert-large-k6s1": {"do_upper_case": True},
    "prokbert-large-k6s2": {"do_upper_case": True}

}


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


class ProkBERTTokenizer(PreTrainedTokenizer):
    """Custom tokenizer for ProkBERT."""
    
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    nucleotide_abc = {'A', 'T', 'C', 'G'}
    extended_nucleotide_abc = {'A', 'T', 'C', 'G', '*'}
    sequence_unk_token = 'N'

    def __init__(self, 
                 tokenization_params: Dict = {},
                 segmentation_params: Dict = {},
                 comp_params: Dict = {},
                 operation_space: str = 'kmer',
                 **kwargs):
        """Initialize the ProkBERT tokenizer.
        
        Args:
            tokenization_params (Dict, optional): Tokenization parameters. Defaults to {}.
            segmentation_params (Dict, optional): Segmentation parameters. Defaults to {}.
            comp_params (Dict, optional): Computational parameters. Defaults to {}.
            operation_space (str, optional): Specifies the operation mode. Can be 'kmer' or 'sequence'. Defaults to 'kmer'.
        """
        super().__init__(**kwargs)
        
        self.defconfig = SeqConfig()
        self.tokenization_params = self.defconfig.get_and_set_tokenization_params(tokenization_params)
        self.segmentation_params = self.defconfig.get_set_segmentation_parameters(segmentation_params)
        self.comp_params = self.defconfig.get_set_computational_paramters(comp_params)
        self.operation_space = operation_space

        vocab_file = self.tokenization_params['vocabfile']
        self.vocab = self.tokenization_params['vocabmap']
        self.id2token = {v: k for k, v in self.vocab.items()}
        self.max_len = self.tokenization_params['max_segment_length']

        if self.operation_space == 'sequence':
            token_extension = sorted(list(set(generate_kmers(ProkBERTTokenizer.extended_nucleotide_abc, self.tokenization_params['kmer'])) - \
                 set(generate_kmers(ProkBERTTokenizer.nucleotide_abc, self.tokenization_params['kmer'])) ))
            self.extended_vocab = deepcopy(self.vocab)
            for token in token_extension:
                self.extended_vocab[token] = 4
                            
            self.unk_token = ProkBERTTokenizer.sequence_unk_token * self.tokenization_params['shift']
            self.mask_token = '*'
            full_unk = 'N' * self.tokenization_params['kmer']
            self.vocab[full_unk] = 1
            self.id2token[1] = full_unk

        else:
            self.unk_token = '[UNK]'
        
        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.special_tokens = list(self.special_tokens_map.values())

        
        
    def tokenize(self, text: str, lca_shift: int = 0, all: bool = False) -> Union[List[str], Tuple[List[List[str]], List[List[str]]]]:
        """
        Tokenizes a given segment.

        Args:
            text (str): The DNA segment to tokenize.
            lca_shift (int, optional): Which tokenized vector belonging to the specified LCA offset should be returned. Defaults to 0.
            all (bool, optional): If True, returns all possible tokenizations. Defaults to False.
        
        Returns:
            Union[List[str], Tuple[List[List[str]], List[List[str]]]]: Tokenized segment or tuple of all possible tokenizations.
        
        Usage Example:
            >>> tokenizer = ProkBERTTokenizer(...)
            >>> segment = 'AATCAAGGAATTATTATCGTT'
            >>> tokens, kmers = tokenizer.tokenize(segment, all=True)
            >>> print(tokens)
            ...
        """
        tokenized_segments, kmerized_segments = lca_tokenize_segment(text, self.tokenization_params)
        if all:
            return tokenized_segments, kmerized_segments
        else:
            return kmerized_segments[lca_shift]

        

    def convert_tokens_to_ids(self, tokens):
        """
        Converts tokens to their corresponding IDs.

        Args:
            tokens (List[str]): List of tokens to convert.
        
        Returns:
            List[int]: List of corresponding token IDs.
        
        Usage Example:
            >>> tokenizer = ProkBERTTokenizer(...)
            >>> tokens = ['AATCAA', 'TCAAGG']
            >>> ids = tokenizer.convert_tokens_to_ids(tokens)
            >>> print(ids)
            ...
        """
        
        return [self.vocab.get(token, self.vocab[self.unk_token]) for token in tokens]

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """
        Converts token IDs back to their original tokens.
        
        Args:
            ids (List[int]): List of token IDs to convert.
        
        Returns:
            List[str]: List of corresponding tokens.
        
        Usage Example:
            >>> tokenizer = ProkBERTTokenizer(...)
            >>> ids = [213, 3343]
            >>> tokens = tokenizer.convert_ids_to_tokens(ids)
            >>> print(tokens)
            ...
        """
        if self.operation_space == 'kmer':
            token_list = [self.id2token.get(id, self.unk_token) for id in ids]

        elif self.operation_space == 'sequence':

            token_list = []
            # Handling the sentence start
            if ids[0] == 2:
                pass
            else:
                token_list.append(self.id2token.get(ids[0], self.unk_token))

            print(token_list)
            if len(ids) > 1:
                # if this is a kmer then we add accordingly. 
                true_start_token = self.id2token.get(ids[1], self.unk_token)


                token_list.append(true_start_token)
            print(token_list)
            if len(ids) >2:
                # Adding the other tokens until the end
                for token_id in ids[2:]:
                    mapped_token_id = self.id2token.get(token_id, self.unk_token)
                    if (mapped_token_id in self.special_tokens):
                        act_token_value = ''
                    else:
                        act_token_value = mapped_token_id[-1*self.tokenization_params['shift']:]
                        token_list.append(act_token_value)

        return token_list
    
 
    def save_vocabulary(self, save_directory: str) -> Tuple[str]:
        """Saves the vocabulary to a file.
        
        Args:
            save_directory (str): Directory where the vocabulary will be saved.
        
        Returns:
            Tuple[str]: Path to the saved vocabulary file.
        """
        with open(f"{save_directory}/vocab.txt", "w") as f:
            for token in self.vocab:
                f.write(token + "\\n")
        return (f"{save_directory}/vocab.txt",)
    
    @classmethod
    def from_pretrained(cls, vocab_file: str) -> 'ProkBERTTokenizer':
        """Loads a pre-trained tokenizer.
        
        Args:
            vocab_file (str): Path to the pre-trained tokenizer vocabulary file.
        
        Returns:
            ProkBERTTokenizer: Loaded tokenizer instance.
        """
        return cls(vocab_file)

    def encode_plus(self, text: str, lca_shift: int = 0, **kwargs) -> Dict[str, np.ndarray]:
        """
        Tokenizes a sequence and returns it in a format suitable for model input.
        
        Args:
            text (str): The sequence to tokenize.
            lca_shift (int, optional): LCA offset for tokenization. Defaults to 0.
        
        Returns:
            Dict[str, np.ndarray]: Dictionary containing token IDs and attention masks.
        
        Usage Example:
            >>> tokenizer = ProkBERTTokenizer(...)
            >>> segment = 'AATCAAGGAATTATTATCGTT'
            >>> encoded = tokenizer.encode_plus(segment)
            >>> print(encoded)
            ...
        """
        tokenized_segments, kmerized_segments = lca_tokenize_segment(text, self.tokenization_params)
        input_ids = tokenized_segments[lca_shift]
        attention_mask = [1] * len(input_ids)

        # Padding
        while len(input_ids) < self.max_len:
            input_ids.append(0)
            attention_mask.append(0)

        return {
            "input_ids": np.array(input_ids, dtype=self.comp_params['np_tokentype']),
            "attention_mask": np.array(attention_mask, dtype=self.comp_params['np_tokentype'])
        }
    
    def batch_encode_plus(self, sequences: List[str], lca_shift: int = 0, all: bool = False, **kwargs) -> Dict[str, List[List[int]]]:
        """
        Tokenizes multiple sequences and returns them in a format suitable for model input. It is assumed that sequences 
        have already been preprocessed (i.e., segmented) and quality controlled. 

        Args:
        - sequences (List[str]): A list of DNA sequences to be tokenized.
        - lca_shift (int, default=0): The LCA offset or windows to get the tokenized vector. If the required offset is >= shift, 
        an error is raised.
        - all (bool, default=False): Whether all possible tokenization vectors should be returned. If False, only the specified 
        offset is used. 
        - **kwargs: Additional arguments (like max_length, padding, etc.)

        Returns:
        - Dict[str, List[List[int]]]: A dictionary containing token IDs, attention masks, and token type IDs.
        """
        shift = self.tokenization_params['shift']
        if lca_shift >= shift:
            raise ValueError(f'The required offset {lca_shift} is invalid. The maximum offset should be < {shift}')
        
        # Parallel tokenization. First, create unique IDs for all sequences. 
        sequence_ids = list(range(len(sequences)))
        to_tokenize_data = (sequences, sequence_ids)
        
        # Tokenize each sequence
        tokenization_results = batch_tokenize_segments_with_ids(
            to_tokenize_data, 
            self.tokenization_params, 
            self.comp_params['cpu_cores_for_tokenization'],
            self.comp_params['batch_size_tokenization'],
            self.comp_params['np_tokentype']
        )

        # Generate input ids, token type ids, and attention masks
        input_ids = []
        token_type_ids = []
        attention_masks = []
        
        if all:
            for tokenized_vectors in tokenization_results.values():
                for tokenized_vector in tokenized_vectors:
                    input_ids.append(tokenized_vector)
                    token_type_ids.append([0] * len(tokenized_vector))
                    attention_masks.append([1] * len(tokenized_vector))
        else:
            for tokenized_vectors in tokenization_results.values():
                selected_vector = tokenized_vectors[lca_shift]
                input_ids.append(selected_vector)
                token_type_ids.append([0] * len(selected_vector))
                attention_masks.append([1] * len(selected_vector))

        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_masks
        }
    
    def batch_decode(self, token_ids_list: List[List[int]], **kwargs) -> List[str]:
        """
        Decodes multiple token ID sequences back into their original sequences.
        
        Args:
            token_ids_list (List[List[int]]): List of token ID sequences.
        
        Returns:
            List[str]: List of decoded sequences.
        
        Usage Example:
            >>> tokenizer = ProkBERTTokenizer(...)
            >>> ids = [[2, 213, 3343, 165, 2580, 248, 3905, 978, 3296, 3]]
            >>> sequences = tokenizer.batch_decode(ids)
            >>> print(sequences)
            ...
        """
        return [self.decode(token_ids) for token_ids in token_ids_list]