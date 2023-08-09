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

from general_utils import *
# Ezt a felhasználónak kellene biztosatania 
# VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}



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
    
    print('Loading sequence data into memory!')
    if isinstance(fasta_files_list, str):
        print('Since the fasta_files_list is a string, not list, we convert to a list.')
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
            print('Are you sure do you want to use DataFrame for the list of sequences?')
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
        print('Sequences is a list, therefore ignoring ids and tracking information. ')
        IsSequenceId = None
        IsSeqList = True
    elif isinstance(sequences, pd.DataFrame):
        print('Sequences is a list, therefore adding tracking information.')
        print('Checking input DataFrame!')
        check_expected_columns(sequences, expected_attributes)
        print('Checking input sequence_id is valid primary key in the DataFrame')
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
        print('TODO ....')
        segments = []

    
    
    if AsDataFrame:
        print('Creating a DataFrame from the segments. ')
        segment_db = pd.DataFrame(segments)
        segment_ids = list(range(len(segment_db)))
        segment_db['segment_id'] = segment_ids
        segment_db = segment_db[return_cols]

    else:
        segment_db = [seg['segment'] for seg in segments]

    return segment_db





def segmentate_single_sequence(sequence, params, AsDataFrame=False):
    """ 

    Cuts a single sequence into segments.

    :param sequences: Each sequence is represented as a string or as a list [fasta_id, description, source_file, sequence, orientation].
    :type sequences: string or list

    :param params: Dictionary with parameters.
    :type params: dict

    :param AsDataFrame: If True, return the segments as a pandas DataFrame. Defaults to False.
    :type AsDataFrame: bool, optional

    :return: The segmented sequences, represented as lists if AsDataFrame is False,
             or if AsDataFrame is True and the input is a list, the segments are returned as a DataFrame
             with columns corresponding to the elements of the input list and the last columns with the segments.
    :rtype: list<list> or DataFrame
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

    :param sequences: List of sequences. Each sequence is represented as a string if AddedHeader is False,
                      or as a list [fasta_id, description, source_file, sequence, orientation] if AddedHeader is True.
    :type sequences: list

    :param params: Dictionary with parameters.
    :type params: dict

    :param AsDataFrame: If True, return the sequences as a pandas DataFrame. Defaults to False.
    :type AsDataFrame: bool, optional

    :return: List of segmented sequences, each represented as a list of segments.
    :rtype: list<list>
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

    :param sequences: List of sequences. Each sequence is represented as a list of kmers.
    :type sequences: list

    :param params: Dictionary with parameters.
    :type params: dict

    :return: List of tokenized sequences, each represented as a list of tokens (segmented sequences).
    :rtype: list<list>
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

    :param segment: DNA sequence.
    :type segment: str

    :param segment_kmers: List of k-mers in the segment.
    :type segment_kmers: list

    :param tokenizer_params: Dictionary containing tokenization parameters.
    :type tokenizer_params: dict

    :return: List of formatted strings representing the sequence with overlapping k-mers.
    :rtype: list
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


#### Kind of general utils

