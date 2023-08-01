# coding=utf-8

""" Library for sequence processing """

#sequence: whole, not-segmented DNA sequence
#sentence: sequence block/chunk - 512, 1024, etc
#kmer: token
#tokenizes: vectorized 

#KERDESEK
#-

#TODO
#def for getting params
#KESZ  - segmentate- DataFrame-s is legyen!
#KESZ  - shift=2 eseten 2 tokenizalt vector! 0, 1 start.poz!
#tokenization default padding=False

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool
import multiprocessing
from os.path import join, isfile, splitext
from os import listdir
import random
from Bio import SeqIO
import numpy as np
import math
import gzip
from mimetypes import guess_type
from functools import partial
import operator
import pathlib

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}



def load_contigs(fasta_files_list, adding_reverse_complement=True, IsAddHeader=False, AsDataFrame=False):
    """ 
    Load contigs from a list of fasta files.

    Parameters:
    fasta_files_list (list): List of paths to fasta files. Compressed (gz) fasta files are accepted as well.
    adding_reverse_complement (bool, optional): If True, add the reverse complement of each sequence. Defaults to True.
    IsAddHeader (bool, optional): If True, include the fasta ID and description in the output. Defaults to False.
    AsDataFrame (bool, optional): If True, return the sequences as a pandas DataFrame. Defaults to False.

    Returns:
    list or DataFrame: The loaded sequences. Each sequence is represented as a string if IsAddHeader is False, 
    or as a list [fasta_id, description, source_file, sequence, orientation] if IsAddHeader is True. 
    If AsDataFrame is True, the sequences are returned as a DataFrame.
    """
    
    print('Loading sequence data into memory!')
    sequences = []
    df_cols = ['fasta_id', 'description', 'source_file', 'sequence', 'orientation']
    for act_assembly in fasta_files_list:
        # Determine the file encoding based on the file extension
        encoding = guess_type(act_assembly)[1]
        _open = partial(gzip.open, mode='rt') if encoding == 'gzip' else open
        with _open(act_assembly) as f_assembly:
            # Parse the fasta file
            contigs = list(SeqIO.parse(f_assembly, "fasta"))
        for contig in contigs:
            act_seq = str(contig.seq)[:]
            act_header = str(contig.id)
            act_description = str(contig.description)
            if adding_reverse_complement:
                # Compute the reverse complement of the sequence
                act_reverse_complement = str(contig.seq.reverse_complement())

            if IsAddHeader:
                # Include the fasta ID, description, source file, sequence, and orientation in the output
                new_record = [act_header, act_description,act_assembly, act_seq, 'forward']
                sequences.append(new_record)

                if adding_reverse_complement:
                    new_record = [act_header, act_description,act_assembly, act_reverse_complement, 'reverse']
                    sequences.append(new_record)
            else:
                # Only include the sequence in the output
                sequences.append(act_seq)
                if adding_reverse_complement:
                    sequences.append(act_reverse_complement)
    if AsDataFrame:
        # Convert the sequences to a DataFrame
        if IsAddHeader:
            sequences = pd.DataFrame(sequences, columns = df_cols)
        else:
            print('Are you sure do you want to use DataFrame for the list of sequences?')
            sequences = pd.DataFrame(sequences, columns = ['sequence'])
    
    return sequences

def segmentate_single_sequence(sequence, params, AsDataFrame=False):
    """ 
    Cuts a single sequence into segments.

    Parameters:
    sequences (string/list): Each sequence is represented as a string or as a list[fasta_id, description, source_file, sequence, orientation].
    params (dict): dictionary with parameters.
    AsDataFrame (bool, optional): If True, return the segments as a pandas DataFrame. Defaults to False.

    Returns:
    list<list> or DataFrame: The segmentated sequences, represented as lists if AsDataFrame is False, 
                       or if AsDataFrame is True, and the input is a list, the segments are returned as a DataFrame with columns corresponding to the elements of the input list and the last columns with the segments.
    """
    segmentation_type = params['segmentation']['segmentation_type']
    shifts = params['segmentation']['shifts']
    kmer = params['segmentation']['kmer']
    minSeqLen = params['segmentation']['minSeqLen']
    df_cols = ['fasta_id', 'description', 'source_file', 'sequence', 'orientation', 'segments']
    
    for v in params['segmentation']:
        print(v, ': ', params['segmentation'][v])
        
    if isinstance(sequence, str):
        act_seq = sequence
    elif isinstance(sequence, list):
        act_seq = sequence[3]  # Get the sequence from the input list
    else:
        raise ValueError("Invalid input type. The input should be either a string or a list.")

    all_segments = []

    if len(act_seq) >= minSeqLen:

        if segmentation_type == 'contigous':
            for i in range(kmer):
                segments = []
                for i in range(i, len(act_seq) - kmer + 1, kmer):
                    segment = act_seq[i:i + kmer]
                    segments.append(segment)
                all_segments.append(segments)
                    
        elif segmentation_type == 'covering':
            for shift in range(shifts): #segmentating with diffferent starting positions
                segments = []
                for i in range(shift, len(act_seq) - kmer + 1, shifts):
                    segment = act_seq[i:i + kmer]
                    segments.append(segment)
                all_segments.append(segments)
        
        if AsDataFrame:
            # Convert the segments to a DataFrame
            print('Are you sure you want to use DataFrame for the list of sequences?')
            if isinstance(sequence, str):
                all_segments = pd.DataFrame({'segments': [all_segments]})
            else:
                all_segments = pd.DataFrame([sequence + [all_segments]], columns = df_cols)

    else:
        print("Sequence ignored due to length constraint:", act_seq)

    
    return all_segments


def segmentate_sequences_from_list(sequences, params, AsDataFrame=False):
    """ 
    Cuts sequences into segments.

    Parameters:
    sequences (list): List of sequences. Each sequence is represented as a string if AddedHeader is False, or as a list[fasta_id, description, source_file, sequence, orientation] if AddedHeader is True.
    params (dict): dictionary with parameters.
    AsDataFrame (bool, optional): If True, return the sequences as a pandas DataFrame. Defaults to False.

    Returns:
    list<list>: List of segmentated sequences, each represented as a list of segments.
    """
    segmentation_type = params['segmentation']['segmentation_type']
    shifts = params['segmentation']['shifts']
    kmer = params['segmentation']['kmer']
    minSeqLen = params['segmentation']['minSeqLen']
    
    for v in params['segmentation']:
        print(v, ': ', params['segmentation'][v])

    segmentated_sequences = []
   
    for sequence in sequences:   
        if isinstance(sequence, str):
            act_seq = sequence
        elif isinstance(sequence, list):
            act_seq = sequence[3]  # Get the sequence from the input list
        else:
            raise ValueError("Invalid input type. The input should be either a string or a list.")

    
        if len(act_seq) >= minSeqLen:
            all_segments = []
            if segmentation_type == 'contigous':
                for i in range(kmer):
                    segments = []
                    for i in range(i, len(act_seq) - kmer + 1, kmer):
                        segment = act_seq[i:i + kmer]
                        segments.append(segment)
                    all_segments.append(segments)

            elif segmentation_type == 'covering':
                for shift in range(shifts): #segmentating with diffferent starting positions
                    segments = []
                    for i in range(shift, len(act_seq) - kmer + 1, shifts):
                        segment = act_seq[i:i + kmer]
                        segments.append(segment)
                    all_segments.append(segments)

            segmentated_sequences.append(all_segments)


        else:
            print("Sequence ignored due to length constraint:", act_seq)

    if AsDataFrame:
        # Convert the segments to a DataFrame
        print('Are you sure you want to use DataFrame for the list of sequences?')
        if isinstance(sequence, str):
            segmentated_sequences = pd.DataFrame({'segments': [segmentated_sequences]})
        else:
            df_cols = ['fasta_id', 'description', 'source_file', 'sequence', 'orientation', 'segments']
            df_sequence = pd.DataFrame([seq for seq in sequences if len(seq[3])>minSeqLen], columns=df_cols[:-1])
            df_sequence['segments'] = segmentated_sequences
            segmentated_sequences = df_sequence

            #segmentated_sequences = df_sequence
            
    return segmentated_sequences
    
    
def tokenize_sentence_from_list(sequences, params):
    """ 
    Tokenizes segmentated sequences.

    Parameters:
    sequences (list): List of sequences. Each sequence is represented as a list of kmers.
    params (dict): dictionary with parameters.

    Returns:
    list<list>: List of segmentated sequences, each represented as a list of segments.
    """
    
    vocabmap = params['tokenization']['vocabmap']
    sentence_length = params['tokenization']['sentence_length']
    min_sentence_size = params['tokenization']['min_sentence_size']
    unkwon_tsh = params['tokenization']['unkwon_tsh']
    
    sentence_tokens = []
    unkw_tsh_count = int(sentence_length*unkwon_tsh)
    print(unkw_tsh_count)
    for act_seq in sequences:
        sentence = [vocabmap['[CLS]']]
        unkcount=0
        unkw_tsh_count = int(len(act_seq)*unkwon_tsh)

        if len(act_seq) < min_sentence_size:
            print('too short sent')
            continue
        for kmer in act_seq:
            try:
                sentence.append(vocabmap[kmer.upper()])
            except KeyError:
                sentence.append(vocabmap['[UNK]'])
                unkcount+=1
        if unkcount > unkw_tsh_count:
            #print('skip sentence')
            #print(act_seq)
            continue
        sentence.append(vocabmap['[SEP]'])
        if len(act_seq) < sentence_length-2:
            extra_padding_tokens_ct = sentence_length - len(act_seq) -2
            for j in range(extra_padding_tokens_ct):
                sentence.append(vocabmap['[PAD]'])
        sentence_tokens.append(sentence)
    return sentence_tokens
    
        
    
def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab

def pretty_print_overlapping_sequence(segment, segment_kmers, params):
    """
    Format the sequence for pretty printing with overlapping k-mers.

    Parameters:
    segment (str): DNA sequence.
    segment_kmers (list): List of k-mers in the segment.
    tokenizer_params (dict): Dictionary containing tokenization parameters.

    Returns:
    list: List of formatted strings representing the sequence with overlapping k-mers.
    """
        
    shift = params['segmentation']['shifts']
    k = params['segmentation']['kmer']
    sep_c = 2
    lines = []
    base_offset = len(str( int((k+3)/shift))) + 3
    first_line = ' '*base_offset + segment
    lines.append(first_line)
    nr_lines = int(np.ceil((k+sep_c)/shift))
    print('Nr. line to cover the seq:  {0}'.format(nr_lines))

    for line_id in range(nr_lines):

        line_mers = [k_mer for j, k_mer in enumerate(segment_kmers) if j%nr_lines== line_id]
        act_line = str(line_id) + '.  ' + ' '*(line_id*shift)  + (' '*(sep_c)).join(line_mers)
        lines.append(act_line)
    lines = '\n'.join(lines)
    return lines



def get_default_segmentation_params(params):
    print('Get default parameters for a segmentation')

    kmer = get_default_value(params['segmentation'], 'kmer', 6)
    #prokbert_base_path = get_default_value(actparams, 'prokbert_base_path', '.')
    minSeqLen = get_default_value(params['segmentation'], 'minSeqLen', 10)
    shifts = get_default_value(params['segmentation'], 'shifts', 2)
    segmenetation_type = get_default_value(params['segmentation'], 'segmenetation_type', 'contigous')

    params['segmentation'] = {'kmer' : kmer,
                    'minSeqLen': minSeqLen,
                    'shifts': shifts,
                    'segmenetation_type': segmenetation_type}
    
    return params

def get_default_tokenization_params(params):
    print('Get default parameters for tokenization')

    sentence_length = get_default_value(params['tokenization'], 'sentence_length', 512)
    min_sentence_size = get_default_value(params['tokenization'], 'min_sentence_size', 2)
    unknown_tsh = get_default_value(params['tokenization'], 'unknown_tsh', 0)
    prokbert_base_path = get_default_lca_tokenizer_get_default_value(params['tokenization'], 'prokbert_base_path', '.')
    
    token_vocab_file = join(prokbert_base_path, 'data/tokenizer/vocabs/bert-base-dna{0}/vocab.txt'.format(kmer))
    vocabmap = {line.strip(): i for i, line in enumerate(open(token_vocab_file))}

    
    params['tokenization'] = {'vocabmap' : vocabmap,
                    'sentence_length': sentence_length,
                    'min_sentence_size': min_sentence_size,
                    'unknown_tsh': unknown_tsh,
                    'prokbert_base_path': prokbert_base_path}
    
    return params

def get_default_value(tokenizer_params, var_name, var_def_value = None):
    if var_name in tokenizer_params:
        var_value=tokenizer_params[var_name]
    else:
        var_value=var_def_value
    return var_value



def get_default_lca_tokenizer_get_default_tokenizer_params(actparams):
    print('Get default parameters for a tokenizer and its preprocessor')

    Ls = get_default_lca_tokenizer_get_default_value(actparams, 'Ls', 1024)
    kmer = get_default_lca_tokenizer_get_default_value(actparams, 'kmer', 6)
    prokbert_base_path = get_default_lca_tokenizer_get_default_value(actparams, 'prokbert_base_path', '.')
    lca_shift = get_default_lca_tokenizer_get_default_value(actparams, 'lca_shift', 1)
    minSeqLen = get_default_lca_tokenizer_get_default_value(actparams, 'minSeqLen', 2)
    unkwon_tsh = get_default_lca_tokenizer_get_default_value(actparams, 'unkwon_tsh', 0)
    shifts = get_default_lca_tokenizer_get_default_value(actparams, 'shifts', [0])
    nr_repetation = get_default_lca_tokenizer_get_default_value(actparams, 'nr_repetation', 1)
    coverage = get_default_lca_tokenizer_get_default_value(actparams, 'coverage', 1)
    P_short = get_default_lca_tokenizer_get_default_value(actparams, 'P_short', 0)
    tokenization_method = get_default_lca_tokenizer_get_default_value(actparams, 'tokenization_method', 'lcas')
    lca_shift = get_default_lca_tokenizer_get_default_value(actparams, 'lca_shift', 1)
    segmenetation_type = get_default_lca_tokenizer_get_default_value(actparams, 'segmenetation_type', 'random')
    lca_left = get_default_lca_tokenizer_get_default_value(actparams, 'lca_left', 0)
    lca_right = get_default_lca_tokenizer_get_default_value(actparams, 'lca_right', 0)

    token_vocab_file = join(prokbert_base_path, 'data/tokenizer/vocabs/bert-base-dna{0}/vocab.txt'.format(kmer))
    vocabmap = {line.strip(): i for i, line in enumerate(open(token_vocab_file))}
    max_sentence_length = kmer + (Ls-2)*lca_shift

    tokenization_params = {'kmer' : kmer,
                    'Ls' : Ls,
                    'minSeqLen': minSeqLen,
                    'unkwon_tsh': unkwon_tsh,
                    'token_vocab_file': token_vocab_file,
                    'vocabmap': vocabmap,
                    'shifts': shifts,
                    'nr_repetation': nr_repetation,
                    'coverage': coverage,
                    'P_short': P_short,
                    'tokenization_method': tokenization_method,
                    'lca_shift': lca_shift,
                    'segmenetation_type': segmenetation_type,
                    'lca_left' : lca_left,
                    'lca_right': lca_right,
                    'max_sentence_length': max_sentence_length}
    
    return tokenization_params






'''

class BertTokenizer(PreTrainedTokenizer):
    r"""
    Construct a BERT tokenizer. Based on WordPiece.
    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.
    Args:
        vocab_file (`str`):
            File containing the vocabulary.
        do_upper_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
        do_basic_tokenize (`bool`, *optional*, defaults to `True`):
            Whether or not to do basic tokenization before WordPiece.
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(
            self,
            vocab_file,
            do_upper_case=True,
            do_basic_tokenize=True,
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]",
            **kwargs
    ):
        super().__init__(
            do_upper_case=do_upper_case,
            do_basic_tokenize=do_basic_tokenize,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )

        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"
                " model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.do_basic_tokenize = do_basic_tokenize
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(
                do_upper_case=do_upper_case
            )

    @property
    def do_lower_case(self):
        return self.basic_tokenizer.do_lower_case

    @property
    def do_upper_case(self):
        return self.basic_tokenizer.do_upper_case
    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)

    def _tokenize(self, text):
        split_tokens = []
        split_tokens = self.basic_tokenizer.tokenize(text)
        return split_tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def build_inputs_with_special_tokens(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A BERT sequence has the following format:
        - single sequence: `[CLS] X [SEP]`
        - pair of sequences: `[CLS] A [SEP] B [SEP]`
        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None,
            already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.
        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.
        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A BERT sequence
        pair mask has the following format:
        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```
        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).
        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        index = 0
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )
        else:
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)
'''

'''
class BasicTokenizer(object):
    """
    Constructs a BasicTokenizer that will run basic tokenization ().
    Args:
        do_upper_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
    """

    def __init__(self, do_upper_case=True):
        self.do_upper_case = do_upper_case

    def tokenize(self, text):
        """
        Basic Tokenization of a piece of text. Split on "white spaces" only, for sub-word tokenization, see
        WordPieceTokenizer.
        Args:

        """
        text = self._clean_text(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_upper_case:
                token = token.upper()
            split_tokens.append(token)
        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)
'''