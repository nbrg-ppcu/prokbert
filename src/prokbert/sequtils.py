
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# coding=utf-8

""" Library for sequence processing """


import os
import sys
import pandas as pd
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
#from typing import Dict, List, Type, Tuple
from itertools import product
from typing import List, Union, Dict, Any, Optional, Tuple, Type
from .general_utils import *


import h5py

def load_contigs(
    fasta_files_list: Union[List[str], str],
    adding_reverse_complement: bool = True,
    IsAddHeader: bool = False,
    AsDataFrame: bool = False,
    to_uppercase: bool = False,
    is_add_sequence_id: bool = False
) -> Union[List[Union[str, List[str]]], pd.DataFrame]:
    """
    Loads contigs from a list of FASTA files.

    :param fasta_files_list: List of paths to FASTA files or a single file path. Compressed (gz) FASTA files are accepted.
    :type fasta_files_list: Union[List[str], str]
    :param adding_reverse_complement: If True, adds the reverse complement of each sequence. Defaults to True.
    :type adding_reverse_complement: bool
    :param IsAddHeader: If True, includes the FASTA ID and description in the output. Defaults to False.
    :type IsAddHeader: bool
    :param AsDataFrame: If True, returns the sequences as a pandas DataFrame. Defaults to False.
    :type AsDataFrame: bool
    :param to_uppercase: If True, converts sequences to uppercase. Defaults to False.
    :type to_uppercase: bool
    :param is_add_sequence_id: If True, adds a unique integer sequence ID to each sequence. Defaults to False.
    :type is_add_sequence_id: bool
    :return: The loaded sequences. Each sequence is represented as a string if IsAddHeader is False, or as a list 
             [sequence_id, fasta_id, description, source_file, sequence, orientation] if IsAddHeader is True and is_add_sequence_id is True. 
             If AsDataFrame is True, the sequences are returned as a DataFrame.
    :rtype: Union[List[Union[str, List[str]]], pd.DataFrame]

    Example:
        >>> fasta_files = ['path/to/file1.fasta', 'path/to/file2.fasta.gz']
        >>> load_contigs(fasta_files, adding_reverse_complement=False, IsAddHeader=True, AsDataFrame=True, to_uppercase=True, is_add_sequence_id=True)
        # Returns a DataFrame with the sequences from the specified FASTA files, all in uppercase, with unique sequence IDs.
    """

    logging.info('Loading sequence data into memory!')
    if isinstance(fasta_files_list, str):
        logging.info('Since the fasta_files_list is a string, not a list, we convert it to a list.')
        fasta_files_list = [fasta_files_list]

    sequences = []
    sequence_id = 0
    df_cols = ['sequence_id', 'fasta_id', 'description', 'source_file', 'sequence', 'orientation'] if (IsAddHeader and is_add_sequence_id) else ['fasta_id', 'description', 'source_file', 'sequence', 'orientation'] if IsAddHeader else ['sequence']
    for act_assembly in fasta_files_list:
        # Determine the file encoding based on the file extension
        encoding = guess_type(act_assembly)[1]
        _open = partial(gzip.open, mode='rt') if encoding == 'gzip' else open
        with _open(act_assembly) as f_assembly:
            # Parse the fasta file
            contigs = list(SeqIO.parse(f_assembly, "fasta"))
        for contig in contigs:
            act_seq = str(contig.seq)[:] if not to_uppercase else str(contig.seq).upper()[:]
            act_header = str(contig.id)
            act_description = str(contig.description)
            if adding_reverse_complement:
                # Compute the reverse complement of the sequence
                act_reverse_complement = str(contig.seq.reverse_complement()) if not to_uppercase else str(contig.seq.reverse_complement()).upper()

            if IsAddHeader:
                # Include sequence ID (if applicable), fasta ID, description, source file, sequence, and orientation in the output
                entry = [sequence_id] if is_add_sequence_id else []
                entry.extend([act_header, act_description, act_assembly, act_seq, 'forward'])
                sequences.append(entry)
                if adding_reverse_complement:
                    entry = [sequence_id + 1] if is_add_sequence_id else []
                    entry.extend([act_header, act_description, act_assembly, act_reverse_complement, 'reverse'])
                    sequences.append(entry)
                    if is_add_sequence_id:
                        sequence_id += 2
            else:
                # Only include the sequence in the output
                sequences.append(act_seq)
                if adding_reverse_complement:
                    sequences.append(act_reverse_complement)

    if AsDataFrame:
        # Convert the sequences to a DataFrame
        sequences = pd.DataFrame(sequences, columns=df_cols)
    return sequences


def segment_sequence_contiguous(
    sequence: str, 
    params: Dict[str, Any], 
    sequence_id: Optional[Any] = np.NaN
) -> List[Dict[str, Any]]:
    """
    Creates end-to-end, disjoint segments of a sequence without overlaps.

    Segments smaller than the predefined minimum length will be discarded.
    This function returns a list of segments along with their positions in the original sequence.

    :param sequence: The input nucleotide sequence to be segmented.
    :type sequence: str
    :param params: Dictionary containing the segmentation parameters. Must include 'min_length' and 'max_length' keys
                   specifying the minimum and maximum lengths of the segments, respectively. Can contain other parameters.
    :type params: Dict[str, Any]
    :param sequence_id: An identifier for the sequence, optional. Defaults to NaN.
    :type sequence_id: Optional[Any]
    :return: A list of dictionaries, each representing a segment. Each dictionary contains the segment's sequence,
             start position, end position, and sequence ID.
    :rtype: List[Dict[str, Any]]

    Example:
        >>> params = {'min_length': 0, 'max_length': 100}
        >>> segment_sequence_contiguous('ATCGATCGA', params)
        [{'segment': 'ATCGATCGA', 'segment_start': 0, 'segment_end': 9, 'sequence_id': np.NaN}]
    """
    
    # Extract segmentation parameters
    min_segment_len = params['min_length']
    max_segment_len = params['max_length']
    
    # Ensure the sequence is treated as a string
    if isinstance(sequence, str):
        act_seq = sequence
    L = len(sequence)
    
    segments = [] 
    for i in range(0, L, max_segment_len):
        act_start_pos = i
        act_end_pos = min(i + max_segment_len, L)
        act_segment = sequence[act_start_pos:act_end_pos]



        # Add segment to the list if it's longer than the minimum length
        if len(act_segment) >= min_segment_len:
            new_record = {
                'segment': act_segment,
                'segment_start': act_start_pos,
                'segment_end': act_end_pos,
                'sequence_id': sequence_id
            }
            segments.append(new_record)
    
    return segments



def segment_sequences_random(
    sequences: Union[pd.DataFrame, List[str]],
    params: Dict[str, Union[int, float, str, Dict, List, Tuple]]
) -> List[Dict[str, Union[int, str]]]:
    """
    Randomly segments the input sequences.

    This function accepts either a list of sequences or a DataFrame containing sequences.
    If a DataFrame is provided, it's assumed to have preprocessed sequences with "sequence" and "sequence_id" columns,
    where "sequence_id" is a valid primary key. The function returns a list of dictionaries,
    each containing details of a segment including its sequence, start position, end position,
    associated sequence ID, and a segment ID (not generated in this function).

    :param sequences: A DataFrame containing sequences with "sequence" and "sequence_id" columns or a list of sequences.
    :type sequences: Union[pd.DataFrame, List[str]]
    :param params: Dictionary containing segmentation parameters such as 'coverage', 'min_length', and 'max_length'.
    :type params: Dict[str, Union[int, float, str, Dict, List, Tuple]]
    :return: A list of dictionaries with each containing details of a segment.
    :rtype: List[Dict[str, Union[int, str]]]

    Notes:
        - The actual number of segments may differ from the expected number due to random sampling and sequences
          being shorter than the specified segment size.
        - Segment IDs are not generated by this function.
    """
    
    # Calculate sequence lengths and cumulative sum of lengths
    sequences['seq_lengths'] = sequences.apply(lambda x: len(x['sequence']), axis=1)
    sequences['lenght_cum_sum'] = sequences['seq_lengths'].cumsum()
    Lseqs = sum(sequences['seq_lengths'])
    
    # Calculate the number of segments to sample based on expected coverage.
    # Note: The actual number might be biased if many sequences are "short" compared to the segment sizes.
    N_segments = int(np.ceil(params['coverage'] * Lseqs / params['max_length']))
    logging.info(f'Sampling {N_segments} segments from {len(sequences)} sequences.')
    
    # Generate random starting coordinates for segments
    start_coords = list(np.sort(np.int64(np.random.uniform(0, sequences['lenght_cum_sum'].max(), N_segments))))
    segmentdb = []
    
    for sid, act_sampling_coord in enumerate(start_coords):

        diff = act_sampling_coord - sequences['lenght_cum_sum']

        # Find the sequence in which the current segment starts
        for i in range(len(sequences['lenght_cum_sum'])):
            if diff[i] < 0:
                break

        act_sequence_id = sequences['sequence_id'].iloc[i]
        rel_coord = act_sampling_coord - sequences['lenght_cum_sum'].iloc[i] + sequences['seq_lengths'].iloc[i]
        
        segment_end = min(rel_coord + params['max_length'], sequences['seq_lengths'].iloc[i])
        
        # Skip the segment if it's shorter than the minimum segment length
        if segment_end - rel_coord < params['min_length']:
            pred_seqgment = sequences['sequence'].iloc[i][rel_coord:segment_end]
            minimum_len = params['min_length']
            logging.info(f'Too short segment, skip! Sampled segment: {pred_seqgment},  Segment end coordinate: {segment_end}, relative coordinate: {rel_coord}, minimum length is: {minimum_len}')
            continue
        
        new_segment = sequences['sequence'].iloc[i][rel_coord:segment_end]
        new_record = {
            'sequence_id': act_sequence_id,
            'segment_start': rel_coord,
            'segment_end': segment_end,
            'segment': new_segment,
            'segment_id': str(sid)
        }
        
        segmentdb.append(new_record)

    return segmentdb

def segment_sequences(
    sequences: Union[List[str], pd.DataFrame],
    params: Dict[str, Union[int, float, str, ]],
    AsDataFrame: bool = False
) -> Union[List[str], pd.DataFrame]:
    """
    Segments sequences based on the provided parameters.

    This function assumes that the sequence is quality controlled and preprocessed, i.e., it is a valid nucleotide sequence.
    If sequences are provided as a DataFrame, then it is assumed that there is a "sequence_id" and
    a "sequence" attribute. The "sequence_id" should be a valid primary key.
    If the output is requested as a DataFrame, then the IDs are added as well.

    :param sequences: A list of sequences or a DataFrame containing sequences.
                      If a DataFrame, it must have "sequence_id" and "sequence" attributes.
    :type sequences: Union[List[str], pd.DataFrame]
    :param params: Dictionary containing the segmentation parameters.
        - 'type' (str): The type of segmentation ('contiguous' or 'random').
        - 'min_length' (int): Minimum length of a segment.
        - 'max_length' (int): Maximum length of a segment.
        - 'coverage' (float): Coverage percentage for random segmentation.
    :type params: Dict[str, Union[int, float, str, Dict[str, int], List[int], Tuple[int, int]]]
    :param AsDataFrame: If True, the output will be a DataFrame. If False, it will be a list. Defaults to False.
    :type AsDataFrame: bool
    :return: List of segmented sequences or a DataFrame with segmented sequences and their corresponding information based on the `AsDataFrame` parameter.
    :rtype: Union[List[str], pd.DataFrame]
    :raises ValueError: If the provided sequences DataFrame does not have the required attributes.
    :raises ValueError: If the "sequence_id" column is not a valid primary key.

    Examples:
        >>> segment_sequences(['AATCAATTTTATTT', 'AGCCGATTCAATTGCATTATTT'], {'type': 'contiguous', 'min_length': 1, 'max_length': 1000, 'coverage': 1.0})
    """
    
    segmentation_type = params['type']

    # Checking for primary key and sequence attribute???
    expected_attributes = ['sequence_id', 'sequence']
    return_cols = ['segment_id', 'sequence_id', 'segment_start', 'segment_end', 'segment']

    if isinstance(sequences, list):
        logging.info('Sequences is a list, therefore ignoring ids and tracking information. ')
        IsSequenceId = None
        IsSeqList = True
    elif isinstance(sequences, pd.DataFrame):
        #logging.info('Sequences is a list, therefore adding tracking information.')
        logging.info('Checking input DataFrame!')
        check_expected_columns(sequences, expected_attributes)
        logging.info('Checking input sequence_id is valid primary key in the DataFrame')
        is_valid_primary_key(sequences, 'sequence_id')
        IsSequenceId = True
        IsSeqList=False

    segments = []
    if segmentation_type == 'contiguous':
        if IsSeqList:
            if IsSequenceId:
                for act_seq_id, seq in enumerate(sequences):
                    act_segments = segment_sequence_contiguous(seq, params, act_seq_id)
                    segments.extend(act_segments)
            else:
                for seq in sequences:
                    act_segments = segment_sequence_contiguous(seq, params)
                    segments.extend(act_segments)
        else:
            for _, rec in sequences.iterrows():
                act_seq = rec['sequence']
                act_seq_id = rec['sequence_id']
                act_segments = segment_sequence_contiguous(act_seq, params, act_seq_id)
                segments.extend(act_segments)
        
    elif segmentation_type == 'random':
        if IsSeqList:
            seqeunce_df = pd.DataFrame(sequences,
                                        columns = ['sequence'])
            seqeunce_df['sequence_id'] = list(range(len(sequences)))
            segments = segment_sequences_random(seqeunce_df, params)

        else:
            segments = segment_sequences_random(sequences, params)
    if AsDataFrame:
        #logging.info('Creating a DataFrame from the segments. ')
        segment_db = pd.DataFrame(segments)
        segment_ids = list(range(len(segment_db)))
        segment_db['segment_id'] = segment_ids
        segment_db = segment_db[return_cols]

    else:
        segment_db = [seg['segment'] for seg in segments]
    return segment_db


def lca_tokenize_segment(
    segment: str, 
    params: Dict[str, Dict[str, int] | int | float]
) -> Tuple[List[List[int]], List[List[str]]]:
    """
    Tokenizes a single segment using Local Context Aware (LCA) tokenization.
    The segment is first split into k-mers with specified shifts and then tokenized into token vectors.

    :param segment: The input nucleotide sequence segment to be tokenized.
    :type segment: str
    :param params: Dictionary containing the tokenization parameters.
        - 'shift' (int): The k-mer shift parameter.
        - 'max_segment_length' (int): Maximum allowable segment length.
        - 'max_unknown_token_proportion' (float): Maximum allowable proportion of unknown tokens in a segment.
        - 'kmer' (int): Size of the k-mer.
        - 'token_limit' (int): Maximum number of tokens allowed in the tokenized output.
        - 'vocabmap' (dict[str, int]): Dictionary mapping k-mers to their respective token values.
    :type params: dict
    :returns: A tuple containing:
        - list[list[int]]: List of tokenized segments (each segment as a list of integers).
        - list[list[str]]: List of k-merized segments with different shifts (each segment as a list of strings).
    :rtype: Tuple[List[List[int]], List[List[str]]]
    :raises ValueError: If the segment length exceeds the `max_segment_length`.

    Examples:
        >>> vocabmap_example = {"[CLS]": 2, "[SEP]": 3, "[UNK]": 0, "TCTTT": 4, "CTTTG": 5, "TTTGC": 6, "TTGCT": 7}
        >>> segment_example = 'TCTTTGCTAAG'
        >>> params_example = {'shift': 1, 'max_segment_length': 512, 'max_unknown_token_proportion': 0.2, 'kmer': 5, 'token_limit': 10, 'vocabmap': vocabmap_example}
        >>> lca_tokenize_segment(segment_example, params_example)
        ([[2, 4, 5, 6, 7, 3]], [['TCTTT', 'CTTTG', 'TTTGC', 'TTGCT']])
    """


    #logging.info('Tokenizing a segment')
    shift = params['shift']
    max_segment_length = params['max_segment_length']
    max_unknown_token_proportion = params['max_unknown_token_proportion']
    kmer = params['kmer']
    token_limit = params['token_limit']
    vocabmap = params['vocabmap']
    add_special_token = params['add_special_token']
    if len(segment) > max_segment_length:
        raise(ValueError(f'The segment is longer {len(segment)} then the maximum allowed segment length ({max_segment_length}). '))
    
    kmers_offset = []
    # For every pssoble offset and window we should get a k-mer vector. 
    # If the segmen is too short or non-existent, then we might have a problem. So, please ensure the segment
    for offset in range(shift):
        kmers = [segment[i:i + kmer] for i in range(offset, len(segment) - kmer + 1, shift)]
        kmers_offset.append(kmers)
    # Mapping the k-mers into numbers
    tokenized_segments = tokenize_kmerized_segment_list(kmers_offset, vocabmap, token_limit, max_unknown_token_proportion, add_special_token)
    return tokenized_segments, kmers_offset
    
    
    
def tokenize_kmerized_segment_list(kmerized_segments: List[List[str]], 
                                   vocabmap: Dict[str, int], 
                                   token_limit: int, 
                                   max_unknown_token_proportion: float, 
                                   add_special_tokens: bool = True) -> List[List[int]]:
    """Tokenizes or vectorizes a list of k-merized segments into a list of token vectors. If the expected number of
    tokens in a segment exceeds the maximum allowed tokens (`token_limit`), the function raises an error. For segments
    where unknown k-mers exceed the proportion set by `max_unknown_token_proportion`, the output is a special token
    sequence indicating an empty sentence.

    :param kmerized_segments: List containing k-merized segments.
    :type kmerized_segments: List[List[str]]
    :param vocabmap: Dictionary that maps k-mers to their respective token values.
    :type vocabmap: Dict[str, int]
    :param token_limit: Maximum number of tokens allowed in the tokenized output.
    :type token_limit: int
    :param max_unknown_token_proportion: Maximum allowable proportion of unknown tokens in a segment.
    :type max_unknown_token_proportion: float
    :param add_special_tokens: Whether to add special tokens (`[CLS]` and `[SEP]`) to the tokenized segments.
    :type add_special_tokens: bool, optional (default=True)
    :returns: List containing tokenized segments.
    :rtype: List[List[int]]
    :raises ValueError: If the expected number of tokens in a segment exceeds `token_limit`.
    
    Examples
    --------

    >>> vocabmap_example = {"[CLS]": 2, "[SEP]": 3, "[UNK]": 0, "TCTTTG": 4, "CTTTGC": 5, "TTTGCT": 6, "TTGCTA": 7}
    >>> kmerized_segment_example = [['TCTTTG', 'CTTTGC', 'TTTGCT', 'TTGCTA']]
    >>> tokenize_kmerized_segment_list(kmerized_segment_example, vocabmap_example, 10, 0.2)
    [[2, 4, 5, 6, 7, 3]]
    """
    
    tokenized_segments = []
    if add_special_tokens:
        empty_sentence = [2, 3]
    else:
        empty_sentence = []

    for act_kmer_list in kmerized_segments:
        if add_special_tokens:
            tokenized_kmerized_segment = [vocabmap['[CLS]']]
        else:
            tokenized_kmerized_segment = []
        unkcount=0
        L_kmerized_segment = len(act_kmer_list)
        unkw_tsh_count = int(L_kmerized_segment*max_unknown_token_proportion)
        if len(act_kmer_list)+2 > token_limit:
            raise(ValueError(f'The expected number of tokens in the segment ({L_kmerized_segment+2}) is larger, then the maximum allowed number of tokens = ({token_limit}). '))
        if L_kmerized_segment == 0:
            logging.info('Its and empty sentence')
            tokenized_kmerized_segment = empty_sentence
            tokenized_segments.append(empty_sentence)
            continue
        for kmer in act_kmer_list:
            try:
                tokenized_kmerized_segment.append(vocabmap[kmer.upper()])
            except KeyError:
                tokenized_kmerized_segment.append(vocabmap['[UNK]'])
                unkcount+=1
        if unkcount > unkw_tsh_count:
            tokenized_segments.append(empty_sentence)
            continue
        if add_special_tokens:
            tokenized_kmerized_segment.append(vocabmap['[SEP]'])
        tokenized_segments.append(tokenized_kmerized_segment)
    
    return tokenized_segments

def process_batch_tokenize_segments_with_ids(
    segments: List[str],
    segment_ids: List[Any],
    tokenization_params: Dict[str, Any],
    np_token_type: type = np.uint16
) -> Dict[Any, List[np.ndarray]]:
    """
    Tokenizes a batch of segments and associates them with their provided IDs.

    This function generates a vector representation for a collection of segments, assuming the segments
    have undergone quality control. The result is a dictionary where the keys are segment IDs, and the values
    are lists of potential vector representations for the segment, with each list element corresponding to
    a specific shift.

    The vector representations are converted to numpy arrays. The output is not a 2D rectangular array but
    a list of arrays.

    :param segments: A list of preprocessed and validated segments.
    :type segments: List[str]
    :param segment_ids: A list of segment IDs corresponding to each segment in `segments`.
    :type segment_ids: List[Any]
    :param tokenization_params: A dictionary containing tokenization parameters.
    :type tokenization_params: Dict[str, Any]
    :param np_token_type: Numpy data type for the tokenized segments, defaults to np.uint16.
    :type np_token_type: type, optional
    :return: A dictionary with segment IDs as keys and lists of numpy arrays representing tokenized segments as values.
    :rtype: Dict[Any, List[np.ndarray]]

    Example:
        >>> segments = ['ACTG', 'TGCA']
        >>> segment_ids = [1, 2]
        >>> tokenization_params = {'max_segment_length': 50, ...}
        >>> process_batch_tokenize_segments_with_ids(segments, segment_ids, tokenization_params)
        {1: [np.array([...], dtype=np.uint16), ...], 2: [np.array([...], dtype=np.uint16), ...]}
    """
    


    #logging.info('Tokenization of a list of segments')
    tokenized_segments_with_ids = {}
    for i, segment in enumerate(segments):
        act_id = segment_ids[i]
        tokenized_segments_with_ids[act_id] = []
        max_segment_length = tokenization_params['max_segment_length']
        if len(segment) > max_segment_length:
            raise(ValueError(f'The segment is longer {len(segment)} then the maximum allowed segment length ({max_segment_length}). '))
        
        tokenized_segment,_ = lca_tokenize_segment(segment, tokenization_params)
        tokenized_segment = [np.array(act_segment, dtype=np_token_type) for act_segment in tokenized_segment]
        tokenized_segments_with_ids[act_id] = tokenized_segment
    return tokenized_segments_with_ids
   
def batch_tokenize_segments_with_ids(segment_data, tokenization_params, num_cores=1, batch_size = 10000, np_token_type=np.uint16):
    """Parallel tokenization of segments. If the segments are provided as DataFrame then it is splitted into junks specified in the paramaters
    The default number of cores are the maximum available ones. If the segment data is a tuple, then it is expected the first element is the list segments, while the second elements are the ids.
    Please note that the segment_ids should be unique. The segments should quality controlloed.

    :param segment_data: param tokenization_params:
    :param num_cores: Default value = 1)
    :param batch_size: Default value = 10000)
    :param np_token_type: Default value = np.uint16)
    :param tokenization_params: 

    """

    if isinstance(segment_data, tuple) or isinstance(segment_data, list):
        segments = segment_data[0]
        segment_ids = segment_data[1]
    elif isinstance(segment_data, pd.DataFrame):
        segments = list(segment_data['segment'])
        segment_ids = list(segment_data['segment_id'])
    else:
        raise(ValueError(f'The input should be either pandas DataFrame or a tuple instead of {segment_data.__class__}'))

    Ndata = len(segments)
    batch_intervals = [(i, min( i+batch_size, Ndata)) for i in range(0, Ndata, batch_size)]
    params = [(segments[interval[0]:interval[1]], 
               segment_ids[interval[0]:interval[1]],
               tokenization_params,
               np_token_type) for interval in batch_intervals]
    with Pool(processes=num_cores) as pool:
        result_list = pool.starmap(process_batch_tokenize_segments_with_ids, params)

    tokenized_sets = {}
    for d in result_list:
        tokenized_sets.update(d)
    

    return tokenized_sets


def get_rectangular_array_from_tokenized_dataset(tokenized_segments_data: Dict[int, List[np.ndarray]], 
                                                 shift: int, 
                                                 max_token_count: int, 
                                                 truncate_zeros: bool = True, 
                                                 randomize: bool = True, 
                                                 numpy_dtype: Type = np.uint16) -> Tuple[np.ndarray, pd.DataFrame]:
    """Create a rectangular numpy array that can be used as input to a Language Model (LM) from tokenized segment data.

    :param tokenized_segments_data: A dictionary where keys are segment ids and values are lists of possible LCA tokenized vectors.
    :type tokenized_segments_data: Dict[int, List[np.ndarray]]

    :param shift: Number of LCA offsets.
    :type shift: int

    :param max_token_count: Maximum allowed token count in the output numpy array.
    :type max_token_count: int

    :param truncate_zeros: If True, truncate columns from the end of the numpy array that only contain zeros. (default=True)
    :type truncate_zeros: bool, optional

    :param randomize: If True, randomize the order of the rows in the output numpy array. (default=True)
    :type randomize: bool, optional

    :param numpy_dtype: Data type of the values in the output numpy array. (default=np.uint16)
    :type numpy_dtype: Type, optional

    :returns: A rectangular numpy array suitable for input to an LM.
    :rtype: np.ndarray

    :returns: A dataframe that describes which row in the numpy array corresponds to which segment and its LCA offset.
        Columns are: ['torch_id', 'segment_id', 'offset']
    :rtype: pd.DataFrame
    
    """


    expected_length = len(tokenized_segments_data)*shift
    X=np.full((expected_length,max_token_count),0, dtype=numpy_dtype)
    torch_db = [] 
    torch_id = 0
    for segment_id, tokenized_vectors in tokenized_segments_data.items():
        for offset in range(shift):
            segment_vector = tokenized_vectors[offset]
            X[torch_id,0:segment_vector.shape[0]] = segment_vector
            torch_db.append([torch_id, segment_id, offset])
            torch_id+=1
    torch_tokenized_segment_db = pd.DataFrame(torch_db,
                                            columns = ['torch_id', 'segment_id', 'offset'])
    
    if randomize:
        logging.info('Doing randomization!')
        perm = np.random.permutation(expected_length)        
        X = X[perm,:]
        torch_tokenized_segment_db.rename({'torch_id': 'original_torch_id'}, axis=1, inplace=True)
        torch_tokenized_segment_db = torch_tokenized_segment_db.iloc[perm,:].reset_index().drop('index', axis=1).reset_index().rename({'index' : 'torch_id'}, axis=1)

    if truncate_zeros:
        logging.info('Tuncating all zeros column')
    X = truncate_zero_columns(X)
    return X, torch_tokenized_segment_db

        
def pretty_print_overlapping_sequence(segment, segment_kmers, tokenizer_params):
    """
    Format the sequence for pretty printing with overlapping k-mers.

    :param segment: DNA sequence.
    :type segment: str

    :param segment_kmers: List of k-mers in the segment.
    :type segment_kmers: list

    :param tokenizer_params: Dictionary containing tokenization parameters.
    :type tokenizer_params: dict

    :return: List of formatted strings representing the sequence with overlapping k-mers.
    :rtype: list
    """
        
    shift = tokenizer_params['shift']
    k = tokenizer_params['kmer']
    sep_c = 2
    lines = []
    base_offset = len(str( int((k+3)/shift))) + 3
    first_line = ' '*base_offset + segment
    lines.append(first_line)
    nr_lines = int(np.ceil((k+sep_c)/shift))
    logging.info('Nr. line to cover the seq:  {0}'.format(nr_lines))

    for line_id in range(nr_lines):

        line_mers = [k_mer for j, k_mer in enumerate(segment_kmers) if j%nr_lines== line_id]
        act_line = str(line_id) + '.  ' + ' '*(line_id*shift)  + (' '*(sep_c)).join(line_mers)
        lines.append(act_line)
    lines = '\n'.join(lines)
    return lines


def generate_kmers(abc, k):
    """Generates all possible k-mers from a given alphabet.

    :param abc: The alphabet.
    :type abc: set
    :param k: Length of the k-mers.
    :type k: int
    :returns: List of all possible k-mers.
    :rtype: List[str]

    """
    return [''.join(p) for p in product(abc, repeat=k)]

def save_to_hdf(X: np.ndarray, hdf_file_path: str, 
                   database: pd.DataFrame = None, 
                   compression: bool = False, 
                   pd_chunksize: int = 10_000_000) -> None:
    """Save a numpy array and an optional pandas DataFrame to an HDF5 file.

    :param X: 2D numpy array to be saved.
    :type X: np.ndarray
    :param hdf_file_path: Path to the HDF5 file.
    :type hdf_file_path: str
    :param database: Pandas DataFrame to be saved. Defaults to None.
    :type database: pd.DataFrame
    :param compression: Whether to apply compression. Defaults to False.
    :type compression: bool
    :param pd_chunksize: Number of rows per chunk for saving the DataFrame. Defaults to 10,000,000.
    :type pd_chunksize: int
    :raises ValueError: If the provided numpy array is not 2D.
    :raises OSError: If there's an error creating the directory structure or removing an existing HDF5 file.
    Example:

    >>> import numpy as np
        >>> import pandas as pd
        >>> array = np.random.random((100, 100))
        >>> df = pd.DataFrame({'A': range(1, 101), 'B': range(101, 201)})
        >>> save_to_hdf(array, "sample.hdf5", database=df, compression=True)
    """
    
    # Check if X is a 2D numpy array
    if len(X.shape) != 2:
        raise ValueError("The provided numpy array is not 2D.")
    
    # If HDF5 file exists, attempt to delete it
    if os.path.exists(hdf_file_path):
        try:
            os.remove(hdf_file_path)
            logging.info(f"Existing HDF5 file {hdf_file_path} removed successfully.")
        except Exception as e:
            raise OSError(f"Error removing existing HDF5 file {hdf_file_path}. Error: {e}")
    
    # Create directory structure for HDF5 file
    create_directory_for_filepath(hdf_file_path)
    
    # Save the numpy array to HDF5
    with h5py.File(hdf_file_path, 'w') as hdf:
        try:
            grp = hdf.create_group("training_data")
        except ValueError:
            del hdf['training_data']

        if compression:
            grp.create_dataset("X", data=X, compression="lzf", chunks=True)
        else:
            grp.create_dataset("X", data=X, chunks=True)
    
    logging.info(f"Numpy array saved to {hdf_file_path} successfully.")

    # Save the pandas DataFrame to HDF5, if provided
    if database is not None:
        logging.info("Adding database into the HDF5 file!")
        num_chunks = int(np.ceil(len(database) / pd_chunksize))
        logging.info(f'Number of chunks: {num_chunks}')
        chunk_grouping = np.arange(len(database)) // pd_chunksize
        chunkseqs = database.groupby(chunk_grouping)
        for i, (_, chunk) in enumerate(chunkseqs):
            logging.info(f'Writing database chunk {i} into {hdf_file_path}')
            if compression:
                chunk.to_hdf(hdf_file_path, f'database_{i}', format='table', data_columns=True,  mode='a', complib='lzo')
            else:
                chunk.to_hdf(hdf_file_path, f'database_{i}', format='table', data_columns=True,  mode='a')

        logging.info('Database addition finished!')
