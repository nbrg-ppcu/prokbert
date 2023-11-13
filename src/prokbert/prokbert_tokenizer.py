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

from .config_utils import *
from .sequtils import *

import logging as logger

#logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}



PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "prokbert-mini-k6s1": "prokbert-base-dna6/vocab.txt",
        "prokbert-mini-k6s2": "prokbert-base-dna6/vocab.txt",
        "prokbert-mini-k1s1": "prokbert-base-dna1/vocab.txt"
    }
}


PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "prokbert-mini-k6s1": 1024,
    "prokbert-mini-k1s1": 1024,
    "prokbert-mini-k6s2": 2048
}

PRETRAINED_INIT_CONFIGURATION = {
    "prokbert-mini-k6s1": {"do_upper_case": True},
    "prokbert-mini-k1s1": {"do_upper_case": True},
    "prokbert-mini-k6s2": {"do_upper_case": True}

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
    """
    Custom tokenizer for ProkBERT, handling specific tokenization processes required for ProkBERT,
    including LCA tokenization and sequence segmentation.
    ProkBERT employs LCA tokenization, leveraging overlapping k-mers to capture rich local context information, enhancing model 
    generalization and performance. The key parameters are the k-mer size and shift. For instance, with a k-mer size of 6 and a 
    shift of 1, the tokenization captures detailed sequence information, while a k-mer size of 1 represents a basic character-based 
    approach.

    :param tokenization_params: Parameters for tokenization, derived from the 'tokenization' part of the config.
        Expected keys include 'type', 'kmer', 'shift', etc. See below for detailed descriptions.
    :type tokenization_params: dict
    :param segmentation_params: Parameters for segmentation, derived from the 'segmentation' part of the config.
        Includes 'type', 'min_length', 'max_length', etc.
    :type segmentation_params: dict
    :param comp_params: Computation parameters from the 'computation' part of the config, like CPU cores and batch sizes.
    :type comp_params: dict
    :param operation_space: Defines the operation space ('sequence' or 'kmer').
    :type operation_space: str

    Tokenization Parameters:
        - type (str): Tokenization approach, default 'lca' for Local Context Aware.
        - kmer (int): k-mer size for tokenization.
        - shift (int): Shift parameter in k-mer.
        - max_segment_length (int): Maximum number of characters in a segment.
        - token_limit (int): Maximum token count for language model processing.
        - max_unknown_token_proportion (float): Maximum allowed proportion of unknown tokens.
        - vocabfile (str): Path to the vocabulary file.
        - isPaddingToMaxLength (bool): Whether to pad sentences to a fixed length.
        - add_special_token (bool): Whether to add special tokens like [CLS], [SEP].

    Segmentation Parameters:
        - type (str): Segmentation type, 'contiguous' or 'random'.
        - min_length (int): Minimum length for a segment.
        - max_length (int): Maximum length for a segment.
        - coverage (float): Expected average coverage of positions in the sequence.

    Computation Parameters:
        - cpu_cores_for_segmentation (int): Number of CPU cores for segmentation.
        - cpu_cores_for_tokenization (int): Number of CPU cores for tokenization.
        - batch_size_tokenization (int): Batch size for tokenization.
        - batch_size_fasta_segmentation (int): Batch size for fasta file processing.
        - numpy_token_integer_prec_byte (int): Integer precision byte for vectorization.
        - np_tokentype (type): Data type for numpy token arrays.

    Usage Example:
        >>> tokenization_parameters = {'kmer': 6, 'shift': 1}
        >>> tokenizer = ProkBERTTokenizer(tokenization_params=tokenization_parameters)
        >>> encoded = tokenizer('ATTCTTT')
        >>> print(encoded)        
    """
    
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    nucleotide_abc = {'A', 'T', 'C', 'G'}
    extended_nucleotide_abc = {'A', 'T', 'C', 'G', '*'}
    sequence_unk_token = 'N'
    default_unk_token="[UNK]"
    default_sep_token="[SEP]"
    default_pad_token="[PAD]"
    default_cls_token="[CLS]"
    default_mask_token="[MASK]"


    def __init__(self, 
                 tokenization_params: Dict = {},
                 segmentation_params: Dict = {},
                 comp_params: Dict = {},
                 operation_space: str = 'sequence',
                 **kwargs):
        """
        :param tokenization_params: Dictionary containing tokenization parameters such as k-mer size,
            shift, max segment length, and more. Defaults to an empty dictionary.
        :type tokenization_params: Dict

        :param segmentation_params: Dictionary containing segmentation parameters like type,
            min/max length, and coverage. Defaults to an empty dictionary.
        :type segmentation_params: Dict

        :param comp_params: Dictionary containing computational parameters as described above
        :type comp_params: Dict

        :param operation_space: Specifies the operation mode, which can be either 'kmer' or 'sequence'.
            Defaults to 'sequence'.
        :type operation_space: str

        The class supports extended vocabulary and custom unknown tokens for sequence-based operation, 
        and aligns with standard tokenization protocols for language models.

        :return: None



        Example:
            >>> tokenizer = ProkBERTTokenizer(tokenization_params={'kmer': 6, 'shift': 1}, operation_space='sequence')
            >>> tokenizer.tokenize("ACGTACGT")
        """

        
        
        self.defconfig = SeqConfig()
        self.tokenization_params = self.defconfig.get_and_set_tokenization_parameters(tokenization_params)
        self.segmentation_params = self.defconfig.get_and_set_segmentation_parameters(segmentation_params)
        self.comp_params = self.defconfig.get_and_set_computational_parameters(comp_params)
        self.operation_space = operation_space

        vocab_file = self.tokenization_params['vocabfile']
        self.vocab = self.tokenization_params['vocabmap']
        self.id2token = {v: k for k, v in self.vocab.items()}
        self.max_len = self.tokenization_params['max_segment_length']
        super().__init__(cls_token=ProkBERTTokenizer.default_cls_token, **kwargs)

        if self.operation_space == 'sequence':
            token_extension = sorted(list(set(generate_kmers(ProkBERTTokenizer.extended_nucleotide_abc, self.tokenization_params['kmer'])) - \
                 set(generate_kmers(ProkBERTTokenizer.nucleotide_abc, self.tokenization_params['kmer'])) ))
            self.extended_vocab = deepcopy(self.vocab)
            for token in token_extension:
                self.extended_vocab[token] = 4
                            
            self.unk_token = ProkBERTTokenizer.sequence_unk_token * self.tokenization_params['shift']
            self.mask_token = '*'
            self.extended_vocab[self.mask_token] = self.vocab['[MASK]']

            full_unk = 'N' * self.tokenization_params['kmer']
            self.vocab[full_unk] = 1
            self.id2token[1] = full_unk
            self.full_unk_token = full_unk

        else:
            self.extended_vocab = self.vocab 
            self.unk_token = '[UNK]'
        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.mask_token = '[MASK]'
        self.special_tokens = list(self.special_tokens_map.values())      
        

    def __len__(self) -> int:
        return len(self.vocab)-1


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

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)
            

    def depr_convert_ids_to_tokens(self, ids: Union[int, List[int]]) -> List[str]:
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

        if isinstance(ids, int):
            token_ids = self.vocab.get(ids, self.vocab[self.unk_token])


        if self.operation_space == 'sequence':
            token_ids = [self.vocab.get(token, self.vocab[self.full_unk_token]) for token in tokens]
        
        else:
            token_ids = [self.vocab.get(token, self.vocab[self.unk_token]) for token in tokens]
        
        return token_ids

    def convert_ids_to_tokens(self, ids: Union[int, List[int]]) -> Union[str, List[str]]:
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

        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        if isinstance(ids, int):
            ids = [ids]
        if len(ids) == 1: 
            #default_token_list = [self.id2token.get(ids[0], self.unk_token)]
            return self.id2token.get(ids[0], self.unk_token)

        if self.operation_space == 'kmer':
            token_list = [self.id2token.get(id, self.unk_token) for id in ids]

        elif self.operation_space == 'sequence':
            token_list = []
            # Handling the sentence start
            if ids[0] == 2:
                pass
            else:
                token_list.append(self.id2token.get(ids[0], self.unk_token))
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
    
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """Saves the vocabulary to a file."""
        if filename_prefix is None:
            filename_prefix = ""
        vocab_file_path = os.path.join(save_directory, filename_prefix + "vocab.txt")
        with open(vocab_file_path, "w") as f:
            for token in self.vocab:
                f.write(token + "\\n")
        return (vocab_file_path,)

    @classmethod
    def from_pretrained(cls, vocab_file: str) -> 'ProkBERTTokenizer':
        """Loads a pre-trained tokenizer.
        
        Args:
            vocab_file (str): Path to the pre-trained tokenizer vocabulary file.
        
        Returns:
            ProkBERTTokenizer: Loaded tokenizer instance.
        """
        return cls(vocab_file)

    def encode_plus(self, text: str, lca_shift: int = 0, padding_to_max=False, **kwargs) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
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

        # Create attention mask with 1s
        attention_mask = [1] * len(input_ids)

        # Set to 0 where input_ids are 1 or 0
        attention_mask = [0 if id == 1 or id == 0 else mask for id, mask in zip(input_ids, attention_mask)]

        # Padding
        if padding_to_max:
            while len(input_ids) < self.max_len:
                input_ids.append(0)
                attention_mask.append(0)

        if kwargs.get('return_tensors') == 'pt':
            simplified_results = {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
            }
        else: 
            simplified_results = {
                "input_ids": np.array(input_ids, dtype=self.comp_params['np_tokentype']),
                "attention_mask": np.array(attention_mask, dtype=self.comp_params['np_tokentype'])
            }

        return simplified_results    
    def batch_encode_plus(self, batch_text_or_text_pairs: List[str], lca_shift: int = 0, all: bool = False, **kwargs) -> Dict[str, List[List[int]]]:
        """
        Tokenizes multiple sequences and returns them in a format suitable for model input. It is assumed that sequences 
        have already been preprocessed (i.e., segmented) and quality controlled. 

        Args:
        - batch_text_or_text_pairs (List[str]): A list of DNA sequences to be tokenized.
        - lca_shift (int, default=0): The LCA offset or windows to get the tokenized vector. If the required offset is >= shift, 
        an error is raised.
        - all (bool, default=False): Whether all possible tokenization vectors should be returned. If False, only the specified 
        offset is used. 
        - **kwargs: Additional arguments (like max_length, padding, etc.)

        Returns:
        - Dict[str, List[List[int]]]: A dictionary containing token IDs, attention masks, and token type IDs.
        """
        sequences = batch_text_or_text_pairs
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
        #print('BFS batch encode plus')
        #print(tokenization_results)
        expected_max_token = max(len(arr) for arrays in tokenization_results.values() for arr in arrays)
        #print(kwargs)
        #print(f'expected_max_token: {expected_max_token}')

        if kwargs and 'return_tensors' in kwargs and kwargs['return_tensors']=='pt':
            X, _ = get_rectangular_array_from_tokenized_dataset(tokenization_results, 
                                                                    self.tokenization_params['shift'],
                                                                    randomize=False,
                                                                    numpy_dtype=np.int64,
                                                                    max_token_count = expected_max_token)
            X = torch.tensor(X, dtype=torch.long)
            token_type_ids = torch.zeros_like(X)
            attention_masks = torch.ones_like(X)
            attention_masks[X == 1] = 0
            attention_masks[X == 0] = 0

            return_data = {
                "input_ids": X,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_masks
            }            

        else:
            # Generate input ids, token type ids, and attention masks
            input_ids = []
            token_type_ids = []
            attention_masks = []
            if all:
                for tokenized_vectors in tokenization_results.values():
                    for tokenized_vector in tokenized_vectors:
                        input_ids.append(list(tokenized_vector))
                        token_type_ids.append([0] * len(tokenized_vector))
                        mask = [1 if token != 1 and token != 0 else 0 for token in tokenized_vector]
                        attention_masks.append(mask)

            else:
                for tokenized_vectors in tokenization_results.values():
                    selected_vector = list(tokenized_vectors[lca_shift])
                    input_ids.append(selected_vector)
                    token_type_ids.append([0] * len(selected_vector))
                    mask = [1 if token != 1 and token != 0 else 0 for token in selected_vector]
                    attention_masks.append(mask)
            return_data = {
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_masks
            }

        return return_data
    


    
    def encode(self, segment: str,  lca_shift: int = 0, all: bool = False, add_special_tokens: bool = True, **kwargs) -> List[int]:
        """
        Encode a DNA sequence into its corresponding token IDs.
        
        Args:
            text (str): The DNA segment to encode.
            add_special_tokens (bool, optional): Whether to add special tokens like [CLS] and [SEP]. Defaults to True.
        
        Returns:
            List[int]: Encoded token IDs.
        
        Usage Example:
            >>> tokenizer = ProkBERTTokenizer(...)
            >>> segment = 'AATCAAGGAATTATTATCGTT'
            >>> ids = tokenizer.encode(segment)
            >>> print(ids)
            ...
        """
        shift = self.tokenization_params['shift']
        if lca_shift >= shift:
            raise ValueError(f'The required offset {lca_shift} is invalid. The maximum offset should be < {shift}')
        
        tokenized_segments, _ = lca_tokenize_segment(segment, self.tokenization_params)

        new_tokenized_segments = []
        if kwargs and 'return_tensors' in kwargs:
            ## print('Converting the results into torch.long')
            for tokenized_segment in tokenized_segments:
                new_tokenized_segment = torch.tensor(tokenized_segment, dtype=torch.long)
                new_tokenized_segments.append(new_tokenized_segment)
            tokenized_segments = new_tokenized_segments


        # if all is set to True, then we return all the possible ids as a list
        if all:
            token_ids = tokenized_segments
            if not add_special_tokens:
                new_token_ids = []
                for token_id_set in tokenized_segments:
                    new_token_ids.append(token_id_set[1:len(token_id_set)-1])
                token_ids = new_token_ids

        else:
            token_ids = tokenized_segments[lca_shift]
            # Convert tokens to their corresponding IDs
            # Add special tokens if needed
            if not add_special_tokens:
                token_ids = token_ids[1:len(token_ids)-1]

        return token_ids
    
    def decode(self, ids):
        tokens = self.convert_ids_to_tokens(ids)
        return ''.join(tokens)
    
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

    def get_vocab(self):

        return self.vocab
    
    
    def get_positions_tokens(self, sequence: str, position: int) -> List[str]:
        """
        Get tokens containing the nucleotide at the given position.

        Args:
            sequence (str): Sequence
            position (int): Position of the character.

        Returns:
            List[str]: List of tokens containing the character at the specified position.

        Usage Example:
            >>> tokenizer = ProkBERTTokenizer(...)
            >>> position = 8
            >>> sequence = "AACTGTGATCTGA"
            >>> tokens = tokenizer.get_positions_tokens(sequence, position)
            >>> print(tokens)
            ...
        """
        all_tokens = []
        sequence_w_pos = sequence
        char = sequence_w_pos[position]
        positions = []
        sequence_w_pos = sequence_w_pos[:position] + '0' + sequence_w_pos[position + 1:]
        print("You look for nucleotide {0} at position {1}".format(char, position))
        ids, kmers_w_0 = self.tokenize(sequence_w_pos, all=True)
        #print(kmers_w_0)
        if position > len(sequence):
            raise ValueError('Given position is higher than the lenght of the sequence!')
        if len(kmers_w_0[0]) == 0 :
            raise ValueError('No kmers could be made from the sequence!')    
        print('All kmers:' , self.tokenize(sequence, all=True)[1])
        # Iterate through token IDs to find tokens containing the character at the given position
        for kmers in kmers_w_0:
            tokens_at_position = []
            pos_in_kmers = []
            for i in range(len(kmers)):
                if '0' in kmers[i]:
                    tok = kmers[i].replace('0', char)
                    tokens_at_position.append(tok)
                    pos_in_kmers.append(i)
            if tokens_at_position:
                all_tokens.append(tokens_at_position)
                positions.append(pos_in_kmers)
        return [all_tokens, positions]
    #STILL NEED WHICH KMER LIST HAS THE POS IF NOT ALL