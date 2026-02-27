from typing import Any, List, Optional, Tuple, Dict, Set, Union

import os
import json
import collections
from copy import deepcopy
from itertools import product

from transformers import PreTrainedTokenizer
from transformers.utils import logging

logger = logging.get_logger(__name__)


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

def generate_kmers(abc: Set[str], k: int) -> List[str]:
    """
    Generates all possible k-mers from a given alphabet.

    :param abc: The alphabet.
    :type abc: Set[str]
    :param k: Length of the k-mers.
    :type k: int
    :return: List of all possible k-mers.
    :rtype: List[str]
    """
    return [''.join(p) for p in product(abc, repeat=k)]


# Utility function to load vocabulary from a file
def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        vocab[token.rstrip("\n")] = index
    return vocab


def resolve_vocab_file(vocab_file: Optional[str], kmer) -> str:
    """
    Resolves the path to the vocabulary file. If not provided, tries to load it
    from the installed prokbert package or download it from the GitHub repository.

    Args:
        vocab_file (str, optional): Path to the vocabulary file.

    Returns:
        str: Path to the resolved vocabulary file.

    Raises:
        FileNotFoundError: If the vocabulary file cannot be resolved.
    """
    if vocab_file and os.path.exists(vocab_file):
        return vocab_file

    # Attempt 1: Check if prokbert is installed
    try:
        import prokbert
        package_dir = os.path.dirname(prokbert.__file__)
        vocab_path = os.path.join(package_dir, 'data/prokbert_vocabs/', f'prokbert-base-dna{kmer}', 'vocab.txt')

        print(vocab_path)


        if os.path.exists(vocab_path):
            logger.info(f"Loaded vocab file from installed prokbert package: {vocab_path}")
            return vocab_path
    except ImportError:
        logger.info("Prokbert package not installed, proceeding to download vocab.txt.")

    # Attempt 2: Download from GitHub repository
    github_url = "https://raw.githubusercontent.com/username/prokbert/main/vocab.txt"
    temp_vocab_path = os.path.join(os.getcwd(), "vocab.txt")
    try:
        import requests

        response = requests.get(github_url, timeout=10)
        response.raise_for_status()  # Raise an error for HTTP failures
        with open(temp_vocab_path, "w", encoding="utf-8") as f:
            f.write(response.text)
        logger.info(f"Downloaded vocab.txt from GitHub to: {temp_vocab_path}")
        return temp_vocab_path
    except requests.RequestException as e:
        raise FileNotFoundError(
            "Could not find or download vocab.txt. Ensure prokbert is installed or "
            "provide a valid vocab file path. Error: {e}"
        ) from e


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

    vocab_files_names = {"vocab_file": "vocab.txt"}



    def __init__(
        self,
        vocab_file: Optional[str] = None,
        kmer: int = 6,
        shift: int = 1,
        operation_space: str = "kmer",
        **kwargs,
    ):
        """
        Initializes the LCATokenizer.

        Args:
            vocab_file (str): Path to the vocabulary file.
            kmer (int): K-mer size for tokenization.
            shift (int): Shift size for tokenization.
            operation_space (str): Defines operation mode ('kmer' or 'sequence').
            kwargs: Additional arguments for PreTrainedTokenizer.
        """
        # load vocabulary directly from the vocab file
        self.config: Dict[str, Any] = {}
        resolved_vocab_file = resolve_vocab_file(vocab_file, kmer)
        self.vocab = load_vocab(resolved_vocab_file)
        self.id2token = {v: k for k, v in self.vocab.items()}
        self.kmer = kmer
        self.shift = shift
        self.operation_space = operation_space

        self.config["kmer"] = kmer
        self.config["shift"] = shift
        self.config["operation_space"] = operation_space

        # special tokens
        kwargs.setdefault("cls_token", "[CLS]")
        kwargs.setdefault("sep_token", "[SEP]")
        kwargs.setdefault("pad_token", "[PAD]")
        kwargs.setdefault("unk_token", "[UNK]")
        kwargs.setdefault("mask_token", "[MASK]")
        self.special_tokens = [kwargs["cls_token"], kwargs["sep_token"], kwargs["pad_token"], kwargs["unk_token"], kwargs["mask_token"]]
        super().__init__(**kwargs)
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


    def get_vocab(self) -> Dict[str, int]:
        return self.vocab


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

        return self.lca_kmer_tokenize_segment(text, offset)

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

    def lca_kmer_tokenize_segment(self, segment: str, offset: int):
        # calculate the tokenization for one offset value
        shift = self.shift
        kmer = self.kmer

        kmers = [segment[i:i + kmer] for i in range(offset, len(segment) - kmer + 1, shift)]

        return kmers

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

        input_ids = [self.cls_token_id] + token_ids_0 + [self.sep_token_id] + token_ids_1 + [self.sep_token_id]
        return input_ids

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create the token type IDs corresponding to the sequences passed. [What are token type
        IDs?](../glossary#token-type-ids)

        Should be overridden in a subclass if the model has a special way of building those.

        Args:
            token_ids_0 (`List[int]`): The first tokenized sequence.
            token_ids_1 (`List[int]`, *optional*): The second tokenized sequence.

        Returns:
            `List[int]`: The token type ids.
        """
        if token_ids_1 is None:
            return (len(token_ids_0)+2) * [0]
        return [0] * len(token_ids_0) + [1] * len(token_ids_1)

    def batch_encode_plus(self, *args, **kwargs):
        """
        Extends the base `batch_encode_plus` method to add custom functionality if needed.

        Args:
            *args: Positional arguments passed to the base method.
            **kwargs: Keyword arguments passed to the base method.

        Returns:
            dict: A dictionary containing the results of batch encoding.
        """
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


    @property
    def vocab_size(self) -> int:
        """
        Returns the size of the vocabulary (number of tokens in `vocab.txt`).

        Returns:
            int: The size of the vocabulary.
        """
        return len(self.vocab)

    def save_pretrained(
            self,
            save_directory: Union[str, os.PathLike],
            legacy_format: Optional[bool] = None,
            filename_prefix: Optional[str] = None,
            push_to_hub: bool = False,
            **kwargs,
        ) -> tuple[str, ...]:
        # overwrite the base `save_pretrained` method to ensure
        # that custom tokenizer parameters are also saved.

        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        save_files = super().save_pretrained(save_directory, legacy_format, filename_prefix, push_to_hub, **kwargs)

        tokenizer_config_path = os.path.join(save_directory, "tokenizer_config.json")

        if os.path.exists(tokenizer_config_path):
            with open(tokenizer_config_path, "r", encoding="utf-8") as f:
                tokenizer_config = json.load(f)
        else:
            tokenizer_config = {}

        tokenizer_config["kmer"] = self.kmer
        tokenizer_config["shift"] = self.shift
        tokenizer_config["operation_space"] = self.operation_space

        with open(tokenizer_config_path, "w", encoding="utf-8") as f:
            json.dump(tokenizer_config, f, indent=2)

        return save_files
