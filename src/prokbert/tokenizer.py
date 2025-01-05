import collections
import os
import json
from copy import deepcopy
from typing import List, Optional, Tuple, Dict
from transformers import PreTrainedTokenizer
from transformers.utils.hub import cached_file, hf_hub_url

from .config_utils import SeqConfig
from .sequtils import generate_kmers, lca_kmer_tokenize_segment

# Define the names of the vocabulary files
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

# Define the mapping for pretrained vocabulary files
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "lca-mini-k6s1": "lca-base-dna6/vocab.txt",
        "lca-mini-k6s2": "lca-base-dna6/vocab.txt",
        "lca-mini-k1s1": "lca-base-dna1/vocab.txt",
    }
}

# Define positional embedding sizes for pretrained models
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "lca-mini-k6s1": 1024,
    "lca-mini-k1s1": 1024,
    "lca-mini-k6s2": 2048,
}

# Define initial configuration for pretrained models
PRETRAINED_INIT_CONFIGURATION = {
    "lca-mini-k6s1": {"do_upper_case": True},
    "lca-mini-k1s1": {"do_upper_case": True},
    "lca-mini-k6s2": {"do_upper_case": True},
}

# Utility function to load vocabulary from a file
def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        vocab[token.rstrip("\n")] = index
    return vocab

class LCATokenizer(PreTrainedTokenizer):
    """
    Custom tokenizer for LCA (Local Context Aware) tasks.
    Handles specific tokenization processes, including k-mer tokenization with configurable shifts.

    Attributes:
        vocab_files_names (dict): Mapping of vocabulary file names.
        pretrained_vocab_files_map (dict): Mapping of pretrained vocabulary files.
        pretrained_init_configuration (dict): Initial configuration for pretrained models.
        max_model_input_sizes (dict): Maximum input sizes for pretrained models.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    
    nucleotide_abc = {"A", "T", "C", "G"}
    extended_nucleotide_abc = {"A", "T", "C", "G", "*"}
    sequence_unk_token = 'N'

    default_unk_token = "[UNK]"
    default_sep_token = "[SEP]"
    default_pad_token = "[PAD]"
    default_cls_token = "[CLS]"
    default_mask_token = "[MASK]"

    def __init__(
        self,
        config: Dict = {},
        operation_space: str = "kmer",
        **kwargs,
    ):
        """
        Initializes the LCATokenizer with configuration and operation space.

        Args:
            config (dict): Tokenization parameters like k-mer size and shift.
            operation_space (str): Defines operation mode ('kmer' or 'sequence').
            kwargs: Additional arguments for PreTrainedTokenizer.
        """
        self.defconfig = SeqConfig()
        config = self.defconfig.get_and_set_tokenization_parameters(config)
        self.config = config
        self.operation_space = operation_space

        # Set default tokens
        kwargs.setdefault("cls_token", self.default_cls_token)
        kwargs.setdefault("unk_token", self.default_unk_token)
        kwargs.setdefault("sep_token", self.default_sep_token)
        kwargs.setdefault("pad_token", self.default_pad_token)
        kwargs.setdefault("mask_token", self.default_mask_token)

        # Load vocabulary
        vocab_file = self.config["vocabfile"]
        self.vocab = self.config["vocabmap"]
        self.id2token = {v: k for k, v in self.vocab.items()}
        self.max_len = self.config["max_segment_length"]

        super().__init__(**kwargs)

        # Handle extended vocabulary for sequence mode
        if self.operation_space == 'sequence':
            token_extension = sorted(list(set(generate_kmers(LCATokenizer.extended_nucleotide_abc, self.config['kmer'])) - \
                 set(generate_kmers(LCATokenizer.nucleotide_abc, self.config['kmer'])) ))
            self.extended_vocab = deepcopy(self.vocab)
            for token in token_extension:
                self.extended_vocab[token] = 4
                            
            self.unk_token = LCATokenizer.sequence_unk_token * self.config['shift']
            self.mask_token = '*'
            self.extended_vocab[self.mask_token] = self.vocab['[MASK]']

            full_unk = 'N' * self.config['kmer']
            self.vocab[full_unk] = 1
            self.id2token[1] = full_unk
            self.full_unk_token = full_unk

        else:
            self.extended_vocab = self.vocab 
            self.unk_token = '[UNK]'

        self.unkown_tokenid = self.vocab['[UNK]']
        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.mask_token = '[MASK]'
        self.special_tokens = list(self.special_tokens_map.values())     



    def _tokenize(self, text, **kwargs):
        """
        Tokenizes the input text using LCA tokenization with an optional offset.

        Args:
            text (str): The input DNA sequence to tokenize.
            kwargs: Additional arguments, including:
                - offset (int): The starting position for tokenization. Default is 0.

        Returns:
            List[str]: A list of tokens generated from the input text.
        """
        offset = kwargs.get("offset", 0)
        #if offset < 0 or offset >= self.config.get("shift", 1):
        #    raise ValueError(f"Invalid offset: {offset}. Must be between 0 and {self.config['shift'] - 1}.")

        return lca_kmer_tokenize_segment(text, offset, self.config)

    def _convert_token_to_id(self, token: str) -> int:
        """
        Converts a token to its corresponding ID using the vocabulary.

        Args:
            token (str): The token to convert.

        Returns:
            int: Token ID, or the unknown token ID if the token is not in the vocabulary.
        """
        return self.extended_vocab.get(token, self.unkown_tokenid)

    def _convert_id_to_token(self, index: int) -> str:
        """
        Converts an ID to its corresponding token using the vocabulary.

        Args:
            index (int): The ID to convert.

        Returns:
            str: Corresponding token, or the unknown token if the ID is not in the vocabulary.
        """


        return self.id2token.get(index, self.unk_token)
    
    def __len__(self) -> int:
        """
        Returns the length of the tokenizer's vocabulary.

        The length returned is one less than the actual number of items in the vocabulary
        to account for a specific offset or adjustment in token indexing.

        :return: The adjusted length of the vocabulary.
        :rtype: int
        """
        return len(self.vocab)



    def tokenize(self, text: str, **kwargs) -> List[str]:
        """
        Tokenizes the input text using LCA tokenization.

        Args:
            text (str): The input DNA sequence to tokenize.
            kwargs: Additional arguments, including:
                - offset (int): The starting position for tokenization. Default is 0.

        Returns:
            List[str]: A list of tokens generated from the input text.
        """
        return self._tokenize(text, **kwargs)

    def encode(self, text: str,  **kwargs) -> List[int]:
        """
        Extends the base `encode` method to support an `offset` parameter for custom tokenization logic.

        Args:
            text (str): Input text (DNA sequence).
            offset (int): Offset parameter for the LCA tokenization. Defaults to 0.
            kwargs: Additional arguments passed to the base `encode` method.

        Returns:
            List[int]: Encoded token IDs.
        """
        # Inject the offset into kwargs for the tokenizer
        offset = kwargs.get("offset", 0)
        kwargs["offset"] = offset
        return super().encode(text, **kwargs)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Builds inputs by adding special tokens to a sequence or pair of sequences.

        Args:
            token_ids_0 (List[int]): List of token IDs for the first sequence.
            token_ids_1 (List[int], optional): List of token IDs for the second sequence.

        Returns:
            List[int]: Input IDs with special tokens.
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        return [self.cls_token_id] + token_ids_0 + [self.sep_token_id] + token_ids_1 + [self.sep_token_id]
    
    def batch_encode_plus(self, *args, **kwargs):
        """
        Extends the base `batch_encode_plus` method to add custom functionality if needed.

        Args:
            *args: Positional arguments passed to the base method.
            **kwargs: Keyword arguments passed to the base method.

        Returns:
            dict: A dictionary containing the results of batch encoding.
        """
        # Call the parent method to handle the batch encoding
        #print('Running batch encoding with ids')
        act_outputs = super().batch_encode_plus(*args, **kwargs)
        return act_outputs


    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Saves the tokenizer's vocabulary to a file.

        Args:
            save_directory (str): Directory to save the vocabulary file.
            filename_prefix (str, optional): Prefix for the filename. Default is None.

        Returns:
            Tuple[str]: Path to the saved vocabulary file.
        """
        if filename_prefix is None:
            filename_prefix = ""
        vocab_file_path = os.path.join(save_directory, filename_prefix + "vocab.txt")
        with open(vocab_file_path, "w") as f:
            for token in self.vocab:
                f.write(token + "\n")
        return (vocab_file_path,)

    def save_pretrained(self, save_directory: str, **kwargs):
        """
        Saves the tokenizer configuration and vocabulary to a directory.

        Args:
            save_directory (str): Directory to save the tokenizer files.
        """
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        super().save_pretrained(save_directory, **kwargs)

        tokenizer_config_path = os.path.join(save_directory, "tokenizer_config.json")
        if os.path.exists(tokenizer_config_path):
            with open(tokenizer_config_path, "r") as f:
                tokenizer_config = json.load(f)
        else:
            tokenizer_config = {}

        tokenizer_config.update({
            "kmer": self.config.get("kmer", 6),
            "shift": self.config.get("shift", 1),
        })

        with open(tokenizer_config_path, "w") as f:
            json.dump(tokenizer_config, f, indent=2)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        Loads a tokenizer from the pretrained model directory or Hugging Face Hub.

        Args:
            pretrained_model_name_or_path (str): Path or model name on Hugging Face Hub.
            kwargs: Additional arguments for initialization.

        Returns:
            LCATokenizer: The loaded tokenizer instance.
        """
        tokenizer_config_file = hf_hub_url(
            pretrained_model_name_or_path, filename="tokenizer_config.json"
        )
        resolved_tokenizer_config_file = cached_file(
            pretrained_model_name_or_path, filename="tokenizer_config.json"
        )

        with open(resolved_tokenizer_config_file, "r") as f:
            tokenizer_config = json.load(f)

        kmer = tokenizer_config.pop("kmer", 6)
        shift = tokenizer_config.pop("shift", 1)
        base_tokenization_config = {'kmer': kmer, 'shift': shift}
        defconfig = SeqConfig()
        config = defconfig.get_and_set_tokenization_parameters(base_tokenization_config)

        tokenizer = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        tokenizer.config = config

        return tokenizer


