# coding=utf-8
# Based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/tokenization_bert.py


# ProkBERT tokenizer stuff

import collections
import os
import unicodedata
from typing import List, Optional, Tuple
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

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    nucleotide_abc = {'A', 'T', 'C', 'G'}
    # * refers to the specicial mask token
    extended_nucleotide_abc = {'A', 'T', 'C', 'G', '*'}
    sequence_unk_token = 'N'

    def __init__(self, 
                tokenization_params = {},
                segmentation_params = {},
                 comp_params = {},
                 operation_space = 'kmer',
                   **kwargs):
        super().__init__(**kwargs)
        
        self.defconfig = SeqConfig()
        tokenization_params = self.defconfig.get_and_set_tokenization_params(tokenization_params)
        segmentation_params = self.defconfig.get_set_segmentation_parameters(segmentation_params)
        comp_params = self.defconfig.get_set_computational_paramters(comp_params)

        # Set tokenization params
        self.tokenization_params = tokenization_params
        self.segmentation_params = segmentation_params
        self.comp_params = comp_params
        self.operation_space = operation_space

        vocab_file = tokenization_params['vocabfile']
        self.vocab = tokenization_params['vocabmap']
        self.id2token = {v: k for k, v in self.vocab.items()}
       
        self.max_len = self.tokenization_params['max_segment_length']


        if self.operation_space == 'sequence':

            token_extension = sorted(list(set(generate_kmers(ProkBERTTokenizer.extended_nucleotide_abc, tokenization_params['kmer'])) - \
                 set(generate_kmers(ProkBERTTokenizer.nucleotide_abc, tokenization_params['kmer'])) ))
            print(len(token_extension))
            self.extended_vocab = deepcopy(self.vocab)
            for token in token_extension:
                self.extended_vocab[token] = 4
                            
            self.unk_token = ProkBERTTokenizer.sequence_unk_token*tokenization_params['shift']
            self.mask_token = '*'


        else:
            self.unk_token = '[UNK]'
        
        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        
        
    def tokenize(self, text, lca_shift=0, all=False):
        """ Tokenize a segment. The segment should be smaller then the maximum that the model could handle. 
        If the all=True, then the function returns with a tuple containing all possible tokenization as list described in  lca_tokenize_segment.
        lca_shift: which tokenized vector beloning to the specified lca_offset should be return. It should be smaller then shift. Default only the first one.
    
        """

        tokenized_segments, kmerized_segments = lca_tokenize_segment(text, self.tokenization_params)
        if all:
            return tokenized_segments, kmerized_segments
        else:
            return kmerized_segments[lca_shift]
        

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab.get(token, self.vocab[self.unk_token]) for token in tokens]

    def convert_ids_to_tokens(self, ids):
        if self.operation_space == 'kmer':
            token_list = [self.id2token.get(id, self.unk_token) for id in ids]
        elif self.operation_space == 'sequence':
            act_unknown_token = self.sequence_unk_token*self.tokenization_params['shift']
            start_tok = ids[0:2]
            other_tokens = [self.id2token.get(id, act_unknown_token)[-1*self.tokenization_params['shift']-1:] for id in ids]
            token_list = start_tok + other_tokens

        return token_list
    
 
    def save_vocabulary(self, save_directory):
        with open(f"{save_directory}/vocab.txt", "w") as f:
            for token in self.vocab:
                f.write(token + "\n")
        return (f"{save_directory}/vocab.txt",)

    @classmethod
    def from_pretrained(cls, vocab_file):
        return cls(vocab_file)

    def encode_plus(self, text,lca_shift=0, **kwargs):
        # This is a basic implementation of encode_plus which may need more features
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
    
    
    def batch_encode_plus(self, sequences: List[str], **kwargs) -> Dict[str, List[List[int]]]:
        """
        Tokenizes multiple sequences and returns them in a format suitable for model input.

        Args:
        - sequences (List[str]): A list of sequences to be tokenized.
        - **kwargs: Additional arguments (like max_length, padding, etc.)

        Returns:
        - Dict[str, List[List[int]]]: A dictionary containing token IDs and other tensors.
        """
        # Tokenize each sequence
        tokenized_data = [self.tokenize(seq) for seq in sequences]
        
        # Convert tokens to IDs for each sequence
        input_ids = [self.convert_tokens_to_ids(tokens) for tokens in tokenized_data]
        
        # Depending on kwargs, you might add padding, truncation, etc.
        # For simplicity, only input_ids are returned here.
        return {
            "input_ids": input_ids
        }
    
    def batch_decode(self, token_ids_list: List[List[int]], **kwargs) -> List[str]:
        """
        Decodes multiple token ID sequences back into their original sequences.

        Args:
        - token_ids_list (List[List[int]]): A list of token ID sequences.
        - **kwargs: Additional arguments.

        Returns:
        - List[str]: The decoded sequences.
        """
        # Decode each set of token IDs
        return [self.decode(token_ids) for token_ids in token_ids_list]
    

