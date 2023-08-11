
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
from typing import Dict, List, Type, Tuple

from general_utils import *
# Ezt a felhasználónak kellene biztosatania 
# VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

import logging



def load_contigs(fasta_files_list, adding_reverse_complement=True, IsAddHeader=False, AsDataFrame=False):
    """ 
    Load contigs from a list of fasta files.

    :param fasta_files_list: List of paths to fasta files. Compressed (gz) fasta files are accepted as well.
    :type fasta_files_list: list

    :param adding_reverse_complement: If True, add the reverse complement of each sequence. Defaults to True.
    :type adding_reverse_complement: bool, optional

    :param IsAddHeader: If True, include the fasta ID and description in the output. Defaults to False.
    :type IsAddHeader: bool, optional

    :param AsDataFrame: If True, return the sequences as a pandas DataFrame. Defaults to False.
    :type AsDataFrame: bool, optional

    :return: The loaded sequences. Each sequence is represented as a string if IsAddHeader is False,
             or as a list [fasta_id, description, source_file, sequence, orientation] if IsAddHeader is True.
             If AsDataFrame is True, the sequences are returned as a DataFrame.
    :rtype: list or DataFrame
    """
    
    logging.info('Loading sequence data into memory!')
    if isinstance(fasta_files_list, str):
        logging.info('Since the fasta_files_list is a string, not list, we convert to a list.')
        fasta_files_list = [fasta_files_list]


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
            logging.info('Are you sure do you want to use DataFrame for the list of sequences?')
            sequences = pd.DataFrame(sequences, columns = ['sequence'])
    
    return sequences


def segment_sequence_contiguous(sequence, params, sequence_id=np.NaN):
    """
    Create end-to-end, disjoint segments of a sequence without overlaps.

    Segments smaller than the predefined minimum length will be discarded. 
    This function returns a list of segments along with their positions in the original sequence.

    Parameters
    ----------
    sequence : str
        The input nucleotide sequence to be segmented.
    params : dict
        Dictionary containing the segmentation parameters. Must have 'min_length' 
        and 'max_length' keys specifying the minimum and maximum lengths of the segments, respectively.
    sequence_id : numeric, optional
        An identifier for the sequence. Defaults to NaN.

    Returns
    -------
    list of dict
        Each dictionary in the list represents a segment and contains the segment's sequence, 
        start position, end position, and sequence ID.


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

def segment_sequences_random(sequences, params):
    """
    Randomly segment the input sequences.

    The input can be a list of sequences or a DataFrame containing sequences.
    If a DataFrame is provided, it's assumed to be preprocessed: 
    the "sequence" column stores the sequences to be segmented, and "sequence_id" is a valid primary key.
    
    The actual coverage might differ from the expected one. 
    The output is a list of dictionaries. Note that segment IDs are not generated in this function.

    Parameters
    ----------
    sequences : pd.DataFrame
        DataFrame containing sequences in the "sequence" column and their associated IDs in "sequence_id".
    params : dict
        Dictionary containing segmentation parameters including 'coverage', 'min_length', and 'max_length'.

    Returns
    -------
    list of dict
        Each dictionary contains information about a segment including its sequence, start position, 
        end position, associated sequence ID, and a segment ID.

    Notes
    -----
    The actual number of segments might differ from the expected number due to the random sampling nature 
    and the presence of sequences shorter than the segment size.

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
            logging.info('Too short segment, skip!')
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



def segment_sequences(sequences, params, AsDataFrame=False):
    """
    Segment sequences based on the provided parameters.
    
    We assume that the sequence is quality controlled and preprocessed, 
    i.e., is a valid nucleotide sequence, etc. If sequences are provided 
    as a DataFrame, then it is assumed that there is a "sequence_id" and 
    a "sequence" attribute. The "sequence_id" should be a valid primary key. 
    If the output is requested as a DataFrame, then the IDs are added as well.

    Parameters
    ----------
    sequences : list or pd.DataFrame
        A list of sequences or a DataFrame containing sequences. 
        If a DataFrame, it must have "sequence_id" and "sequence" attributes.
    params : dict
        Dictionary containing the segmentation parameters. 
        The 'type' key in the dictionary can be 'contiguous' or 'random'.
    AsDataFrame : bool, optional
        If True, the output will be a DataFrame. If False, it will be a list. 
        Defaults to False.

    Returns
    -------
    list or pd.DataFrame
        List of segmented sequences or a DataFrame with segmented sequences 
        and their corresponding information based on the `AsDataFrame` parameter.

    Raises
    ------
    ValueError
        If the provided sequences DataFrame does not have the required attributes.
    ValueError
        If the "sequence_id" column is not a valid primary key.

    Notes
    -----
    If the segmentation type is 'random', the functionality is yet to be implemented.

    Examples
    --------
    TODO: Add examples after finalizing the function's behavior and output.

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


def lca_tokenize_segment(segment, params):
    """
    Tokenizes a single segment using Local Context Aware (LCA) tokenization. 
    The segment is first split into k-mers with specified shifts and then tokenized into token vectors.

    Parameters
    ----------
    segment : str
        The input nucleotide sequence segment to be tokenized.
    params : dict
        Dictionary containing the tokenization parameters:
            - 'shift' (int): The k-mer shift parameter.
            - 'max_segment_length' (int): Maximum allowable segment length.
            - 'max_unknown_token_proportion' (float): Maximum allowable proportion of unknown tokens in a segment.
            - 'kmer' (int): Size of the k-mer.
            - 'token_limit' (int): Maximum number of tokens allowed in the tokenized output.
            - 'vocabmap' (dict[str, int]): Dictionary that maps k-mers to their respective token values.

    Returns
    -------
    tuple
        - list[list[int]]: List containing tokenized segments.
        - list[list[str]]: List containing k-merized segments with different shifts.

    Raises
    ------
    ValueError
        If the segment length exceeds the `max_segment_length`.

    Examples
    --------
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
    if len(segment) > max_segment_length:
        raise(ValueError(f'The segment is longer {len(segment)} then the maximum allowed segment length ({max_segment_length}). '))
    
    kmers_offset = []
    # For every pssoble offset and window we should get a k-mer vector. 
    # If the segmen is too short or non-existent, then we might have a problem. So, please ensure the segment
    for offset in range(shift):
        kmers = [segment[i:i + kmer] for i in range(offset, len(segment) - kmer + 1, shift)]
        kmers_offset.append(kmers)
    # Mapping the k-mers into numbers
    tokenized_segments = tokenize_kmerized_segment_list(kmers_offset, vocabmap, token_limit, max_unknown_token_proportion)
    return tokenized_segments, kmers_offset
    
    
    
def tokenize_kmerized_segment_list(kmerized_segments, vocabmap, token_limit, max_unknown_token_proportion):
    """ 
    Tokenizes or vectorizes a list of k-merized segments into a list of token vectors. If the expected number of 
    tokens in a segment exceeds the maximum allowed tokens (`token_limit`), the function raises an error. For segments
    where unknown k-mers exceed the proportion set by `max_unknown_token_proportion`, the output is a special token 
    sequence indicating an empty sentence.

    Parameters
    ----------
    kmerized_segments : list[list[str]]
        List containing k-merized segments.
    vocabmap : dict[str, int]
        Dictionary that maps k-mers to their respective token values.
    token_limit : int
        Maximum number of tokens allowed in the tokenized output.
    max_unknown_token_proportion : float
        Maximum allowable proportion of unknown tokens in a segment.

    Returns
    -------
    list[list[int]]
        List containing tokenized segments.

    Raises
    ------
    ValueError
        If the expected number of tokens in a segment exceeds `token_limit`.

    Examples
    --------
    >>> vocabmap_example = {"[CLS]": 2, "[SEP]": 3, "[UNK]": 0, "TCTTTG": 4, "CTTTGC": 5, "TTTGCT": 6, "TTGCTA": 7}
    >>> kmerized_segment_example = [['TCTTTG', 'CTTTGC', 'TTTGCT', 'TTGCTA']]
    >>> tokenize_kmerized_segment_list(kmerized_segment_example, vocabmap_example, 10, 0.2)
    [[2, 4, 5, 6, 7, 3]]
    """
    
    tokenized_segments = []
    empty_sentence = [2, 3]

    for act_kmer_list in kmerized_segments:
        tokenized_kmerized_segment = [vocabmap['[CLS]']]
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
        tokenized_kmerized_segment.append(vocabmap['[SEP]'])
        tokenized_segments.append(tokenized_kmerized_segment)
    
    return tokenized_segments

def process_batch_tokenize_segments_with_ids(segments, segment_ids, tokenization_params, np_token_type=np.uint16):
    """
    Tokenizes a batch of segments and associates them with their provided IDs.

    This function generates a vector representation for a collection of segments. It presumes that 
    the segments have undergone quality control. The result is a dictionary where the keys represent
    the provided segment IDs, and the values are lists of potential vector representations for the segment.
    Each list element corresponds to a specific shift (e.g., 0-shifted, 1-shifted, etc.).
    
    The vector representations are converted to numpy arrays. Note that the output isn't a 2D rectangular
    array but a list of arrays.

    Parameters
    ----------
    segments : list
        A list of preprocessed and validated segments.
    segment_ids : list
        A list of segment IDs corresponding to each segment in the `segments` list.
    tokenization_params : dict
        A dictionary containing tokenization parameters.

    Returns
    -------
    dict
        A dictionary where keys are segment IDs and values are lists of numpy arrays representing 
        tokenized segments.

    Raises
    ------
    ValueError
        If the length of a segment exceeds the maximum permissible segment length defined in `tokenization_params`.

    """
    logging.info('Tokenization of a list of segments')
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
    """ Parallel tokenization of segments. If the segments are provided as DataFrame then it is splitted into junks specified in the paramaters
    The default number of cores are the maximum available ones. If the segment data is a tuple, then it is expected the first element is the list segments, while the second elements are the ids.
    Please note that the segment_ids should be unique. The segments should quality controlloed. 
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
    """
    Create a rectangular numpy array that can be used as input to a Language Model (LM) from tokenized segment data.

    Parameters:
    ----------
    tokenized_segments_data : Dict[int, List[np.ndarray]]
        A dictionary where keys are segment ids and values are lists of possible LCA tokenized vectors.
        
    shift : int
        Number of LCA offsets.
        
    max_token_count : int
        Maximum allowed token count in the output numpy array.
        
    truncate_zeros : bool, optional (default=True)
        If True, truncate columns from the end of the numpy array that only contain zeros.
        
    randomize : bool, optional (default=True)
        If True, randomize the order of the rows in the output numpy array.
        
    numpy_dtype : Type, optional (default=np.uint16)
        Data type of the values in the output numpy array.

    Returns:
    -------
    np.ndarray
        A rectangular numpy array suitable for input to an LM.
        
    pd.DataFrame
        A dataframe that describes which row in the numpy array corresponds to which segment and its LCA offset.
        Columns are: ['torch_id', 'segment_id', 'offset']

    """
    # ... [rest of the function code]


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

    TODO: Pretty print: mindegyik lca tokenizálást kellene vizualizálni, most pl. az offset hiányzik, így a shift=2 esetre csak a 0. offset-re vizualizál jól. 
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

