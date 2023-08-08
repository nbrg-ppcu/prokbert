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


def segment_sequence(sequence, params,  AsDataFrame=False):

    segmentation_type = params['type']
    min_segment_len = params['min_length']
    max_segment_len = params['max_length']
    if isinstance(sequence, str):
        act_seq = sequence
    
    




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

