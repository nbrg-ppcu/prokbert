# coding=utf-8

# Lib for tokenization 

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
import random
import numpy as np
import h5py
import math
import gzip
from mimetypes import guess_type
from functools import partial
import operator
import prokbert.dnadatasets, prokbert.dnatokenizer
import pathlib
from distutils.dir_util import copy_tree
import os
import pathlib
import torch

def get_non_empty_files(start_path, extension='.fasta'):
    for dirpath, _, filenames in os.walk(start_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if filename.endswith(extension) and os.path.getsize(filepath) > 0:
                yield (filename)
            


def check_model_existance_and_checkpoint(model_name, output_path):
# Milyen modell van meg, milyen checkpoint-tal, ami nem üres

    model_path = join(output_path,model_name)
    path_exists = pathlib.Path.exists(pathlib.Path(model_path))
    largest_checkpoint_dir = None
    largest_checkpoint = None
    chekcpoint_nr = None
    print(model_path)
    if path_exists:
        try:
            subfolders = [ f for f in os.scandir(model_path) if f.is_dir() ]
            subfolders = [sf for sf in subfolders if len(os.listdir(subfolders[0])) > 1]
            chekcpoint_nr = sorted([int(f.name[11:]) for f in subfolders if f.name.startswith('checkpoint-')])
            largest_checkpoint = chekcpoint_nr[-1]
            largest_checkpoint_dir = join(model_path, 'checkpoint-' + str(largest_checkpoint))

        except IndexError:
            print('Something is wrong, set default valies')
            print(str(subfolders))
            path_exists =False
            largest_checkpoint_dir = None
            largest_checkpoint = None
        
    
    return path_exists, largest_checkpoint_dir, largest_checkpoint, chekcpoint_nr

def copy_model_dir_into_another(checkpoint_dir, output_folder, act_model):
    
    copy_output_dir = join(output_folder, act_model)
    copy_output_dir_with_cp = join(copy_output_dir, pathlib.PurePath(checkpoint_dir).name)


    print('Creating: ' + copy_output_dir_with_cp)
    os.makedirs(copy_output_dir_with_cp, exist_ok=True)

    print('Copy from: {0} TO {1}'.format(checkpoint_dir, copy_output_dir_with_cp))
    copy_tree(checkpoint_dir, copy_output_dir_with_cp)


def copy_specific_cp_model_data(act_model, input_dir_folder, output_folder,  cp_to_copy = None):
    
    path_exists, largest_checkpoint_dir, largest_checkpoint, chekcpoint_nr = check_model_existance_and_checkpoint(act_model, input_dir_folder)
    copy_model_dir_into_another(largest_checkpoint_dir, output_folder, act_model)
    
    if cp_to_copy and len(cp_to_copy)>0:
    
        for cp in cp_to_copy:
            expected_input_folder =  join(input_dir_folder, act_model, 'checkpoint-' + str(cp))
            print('expected_input_folder: ', expected_input_folder)
            cp_path_exists = pathlib.Path.exists(pathlib.Path(expected_input_folder))
            print(cp_path_exists)
            if cp_path_exists:
                print('Path exists!')
                copy_model_dir_into_another(expected_input_folder, output_folder, act_model)
            else:
                print('ERROR, skip, Path is not exist!')

    
    


#Hard coded cuts, only 51x hosszú
def cut_no_overlap_old(length, kmer=6, max_prob=0.5):
    cuts = []
    while length:
        if length <= 509+kmer:
            cuts.append(length)
            break
        else:
            if random.random() > max_prob:
                cut = max(int(random.random()*(509+kmer)), 6)
            else:
                cut = 509+kmer
            cuts.append(cut)
            length -= cut
    return cuts

def get_segment_sizes_without_overlap(length, kmer=6, max_prob=2, max_length=512):
    cuts = []
    offset = max_length - kmer  - 2 
    while length:
        if length <= offset+kmer:
            cuts.append(length)
            break
        else:
            if random.random() > max_prob:
                print('Random cut')
                cut = max(int(random.random()*(offset)), kmer)
            else:
                cut = offset+kmer
            cuts.append(cut)
            length -= cut
    return cuts



def load_contigs(fasta_files_list, adding_reverse_complement=True):
    
    print('Loading sequence data into memory!')
    sequences = []
    for act_assembly in fasta_files_list:
        encoding = guess_type(act_assembly)[1]  # uses file extension
        _open = partial(gzip.open, mode='rt') if encoding == 'gzip' else open
        with _open(act_assembly) as f_assembly:
            contigs = list(SeqIO.parse(f_assembly, "fasta"))
            
        for contig in contigs:
            sequences.append(str(contig.seq)[:])
            #print(contig)
            if adding_reverse_complement:
                sequences.append(str(contig.seq.reverse_complement()))
    #print('Shuffle sequences')
    #random.shuffle(sequences)
    #print('Finished')
    return sequences


def process_line(line, kmer_size, Ls=512, max_prob=1):
    line_length = len(line)
    cuts = get_segment_sizes_without_overlap(length=line_length, kmer=kmer_size,max_prob=max_prob, max_length=Ls)
    start = 0
    tokenized_outputs = []
    for cut in cuts:
        new_line = line[start:start+cut]
        sentence = get_kmer_sentence_list(new_line, kmer=kmer_size)
        #sentence = DnaBertlib.get_kmer_sentence(new_line, kmer=kmer_size)
        start += cut
        tokenized_outputs.append(sentence)
    return tokenized_outputs        
    

def kmerize_contigs(shuffled_seqs,vocabmap, kmer_size=6, nr_shuffling=3, Ls=512, max_prob=1):
    print('Kmerizing!')
    tokenized_outputs_all = []
    for act_shuffle in range(nr_shuffling):
        print('Shuffling: ' + str(act_shuffle))
        random.shuffle(shuffled_seqs)
        for line in shuffled_seqs:
            kmers = process_line(line, kmer_size, Ls=Ls, max_prob=max_prob)
            tokens = tokenize_sentence(kmers, vocabmap, sentence_length = Ls)
            tokenized_outputs_all.extend(tokens)
    print('Shuffling the segments!')
    random.shuffle(tokenized_outputs_all)
    print('Finished')
    return tokenized_outputs_all

def get_kmer_sentence_list(original_string, kmer=6, stride=1):
    if kmer == -1:
        return original_string

    new_sentence = []
    for i in range(len(original_string)-kmer+1):
        new_sentence.append(original_string[i:i+kmer])
    
    return new_sentence


def tokenize_sentence(kmers, vocabmap, sentence_length = 512, min_sentence_size = 80, unkwon_tsh = 0.05):
    
    sentence_tokens = []
    unkw_tsh_count = int(sentence_length*unkwon_tsh)
    for kmer_sentence in kmers:
        sentence = [vocabmap['[CLS]']]
        unkcount=0
        unkw_tsh_count = int(len(kmer_sentence)*unkwon_tsh)

        if len(kmer_sentence) < min_sentence_size:
            continue
        for kmer in kmer_sentence:
            try:
                sentence.append(vocabmap[kmer.upper()])
            except KeyError:
                sentence.append(vocabmap['[UNK]'])
                unkcount+=1
        if unkcount > unkw_tsh_count:
            #print('skip sentence')
            #print(kmer_sentence)
            continue
        sentence.append(vocabmap['[SEP]'])
        if len(kmer_sentence) < sentence_length-2:
            extra_padding_tokens_ct = sentence_length - len(kmer_sentence) -2
            for j in range(extra_padding_tokens_ct):
                sentence.append(vocabmap['[PAD]'])
        sentence_tokens.append(sentence)
    return sentence_tokens
        
    
    
    
def get_genome_batches(input_fasta_files, block_size = 100):
    
    #block_size = 100
    N = len(input_fasta_files)
    Nbatches = int(np.ceil(N/block_size))
    print('Number of batches {0}'.format(Nbatches))
    ranges = []
    genome_batches = []
    for i in range(Nbatches):
        new_range = (i*block_size, min((i+1)*block_size, N) )
        ranges.append(new_range)
        genome_batches.append(input_fasta_files[new_range[0]:new_range[1]])
    return genome_batches, ranges


def process_batch_tokenize_hdf5(output_folder, kmer_size,vocabmap, act_batch_id, genome_batches,Ls=512, train_prob=1, max_prob=1):
    
    
    if random.random()>train_prob:
        tt_prefix = 'test'
    else:
        tt_prefix = 'train'
    
    output_tokenized_padded_file = join(output_folder, tt_prefix + '_test_tokenized_batch_{0}.txt'.format(act_batch_id))
    output_tokenized_padded_file_hdf = join(output_folder, 'hdf5_' + tt_prefix + '_test_tokenized_batch_{0}.h5'.format(act_batch_id))

    act_batch = genome_batches[act_batch_id]
    #print(act_batch[0:2])
    
    print('Tokeneizing batch {0}'.format(act_batch_id))
    print(' output file: {0}'.format(output_tokenized_padded_file))

    print('Parsing genome data!')
    shuffled_seqs = load_contigs(act_batch)
    print('K-merization and tokenization fist round!!')
    tokenized_outputs_all = kmerize_contigs(shuffled_seqs,vocabmap,kmer_size=kmer_size, nr_shuffling=1, Ls=Ls,max_prob=max_prob )
  
    print('Collecting info!')
    nr_tokens = sum([len(sentence) for sentence in tokenized_outputs_all])
    nr_lines = len(tokenized_outputs_all)
    
    print('Number of tokens: {0}'.format(nr_tokens))
    print('Number of lines: {0}'.format(nr_lines))
    tokenized_outputs_all_array = np.array(tokenized_outputs_all,dtype="uint16")
    hdf_file = h5py.File(output_tokenized_padded_file_hdf, 'a')
    try:
        grp = hdf_file.create_group("training_data")
    except ValueError:
        del hdf_file['training_data']
    #ds = grp.create_dataset("sentences", data = tokenized_outputs_all_array, chunks=True, compression="gzip", maxshape=(None, tokenized_outputs_all_array.shape[1]))
    ds = grp.create_dataset("sentences", data = tokenized_outputs_all_array, chunks=True, maxshape=(None, tokenized_outputs_all_array.shape[1]))
    grp.create_dataset("dataset_size", data = (nr_lines, tokenized_outputs_all_array.shape[1]))
    hdf_file.close()

    

def concatenate_hdf5_files(hdf5_folder, outputfile, input_prefix, output_prefix, IsCompression=False):
    
    
    train_hdf_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(hdf5_folder) for f in filenames if f.startswith(input_prefix) and os.path.splitext(f)[1] == '.h5']
    
    concatenated_hdf_File = outputfile
    #print(train_hdf_files)
    hdf_files = [h5py.File(hdffile) for hdffile in train_hdf_files]
    

    dataset_sizes = [list(hdf_file['training_data']['dataset_size']) for hdf_file in hdf_files]
    
    final_size_sentence_lenghts = [dataset_size[1] for dataset_size in dataset_sizes]
    final_sentence_counts = [dataset_size[0] for dataset_size in dataset_sizes]

    
    sentence_lenght = int(np.mean(final_size_sentence_lenghts))
    sentence_counts = sum(final_sentence_counts)

    print('Creating a concatenated HDF file:\n  Sentence length: {0}\n  \
Sentence counts: {1}\n\
 Nr files: {2}'.format(sentence_lenght, sentence_counts, len(dataset_sizes)))
    print('Creating the large dataset file {0}'.format(concatenated_hdf_File))   
    with h5py.File(concatenated_hdf_File, 'w') as hout:
        try:
            grp = hout.create_group("training_data")
        except ValueError:
            del hout['training_data']
        print('Addin modell size')
        grp.create_dataset("dataset_size", data = (sentence_counts, sentence_lenght))
        print('Adding first dataset!')
        datasetname = list(hdf_files[0]['training_data'])[1]


        #print(list(hdf_files[0]['training_data']))
        #print(list(hdf_files[0]['training_data'])[1]  )

        first_dataset = np.array(hdf_files[0]['training_data'][datasetname])
        
        if IsCompression:
            ds = grp.create_dataset("sentences", data = first_dataset, 
                                    chunks=True, compression="gzip", 
                                    maxshape=(sentence_counts, sentence_lenght))
        else:
            ds = grp.create_dataset("sentences", data = first_dataset, 
                                    chunks=True, #compression="gzip", 
                                    maxshape=(sentence_counts, sentence_lenght))
        act_start_pos = dataset_sizes[0][0]
        for i, dataseth in enumerate(hdf_files[1:]):
            print('Adding data form {0}'.format(train_hdf_files[i+1]))
            datasetname = list(dataseth['training_data'])[1]
            act_ds = dataseth['training_data'][datasetname]
            act_ds_length = dataset_sizes[i+1][0]
            ds.resize( (act_start_pos + act_ds_length,sentence_lenght))
            ds[act_start_pos:,:] = act_ds
            act_start_pos = act_start_pos + act_ds_length

    [hdf_file.close() for hdf_file in hdf_files]
    
    hout.close()

def get_non_overlapping_contigous_largest_cuts(act_seq, params):

    Lc = len(act_seq)
    Ls = params['tokenization']['Ls'] 
    kmer = params['tokenization']['kmer']
    L = kmer*(Ls-2) #
    nr_cuts = math.ceil(Lc/L)
    cut_intervals = [ (min(i*L, Lc), min( (i+1)*L,Lc)) for i in range(nr_cuts)]

    return cut_intervals



######## Non-overlapping part ############


def generate_random_intervals_with_length(act_seq, params):
    Lc = len(act_seq)
    Ls = params['tokenization']['Ls'] 
    kmer = params['tokenization']['kmer']

    L = kmer*(Ls-2) #
    #Mi van akkor, ha a szekvencia rövidebb, mint a a lehetséges ablak
    if L >= Lc:
        sampled_intervals = [(0,Lc)]
        return sampled_intervals
 
    minLs = params['tokenization']['minLs'] 
    coverage = params['tokenization']['coverage'] 
    P_short = params['tokenization']['P_short'] 
    P_full = 1 - P_short
    expected_sentence_length = P_short*( (L- minLs*kmer)*0.5) + P_full*L
    N = math.ceil((coverage * Lc)/expected_sentence_length)

    sampled_intervals = [] 
    for i in range(N):
        start_pos = int((Lc-L)*random.random())
        if random.random() > P_short:
            end_pos = start_pos + L+4
        else:
            end_pos = start_pos + int((Ls-2)*random.random())*kmer
        sampled_intervals.append( [start_pos, end_pos])
    return sorted(sampled_intervals, key=operator.itemgetter(0))



def generate_random_started_intervals(act_seq, params):
    kmer = params['tokenization']['kmer']
    Ls = params['tokenization']['Ls'] 
    minLs = params['tokenization']['minLs'] 
    P_short = params['tokenization']['P_short'] 
    L = len(act_seq)
    cut_intervals = []
    start_pos = int((Ls+1)*random.random())
    Lsbp = Ls*kmer

    N_sentences = math.floor((L-start_pos)/Lsbp)
    Nr = (L-start_pos) % Lsbp
    #print('Generating {0} sentences, rest is: {1}'.format(N_sentences, Nr))
    i=0
    for i in range(N_sentences):
        cut_intervals.append((start_pos+i*Lsbp, start_pos+(i+1)*Lsbp))
    if Nr > minLs:
        cut_intervals.append((start_pos+(i+1)*Lsbp,L))
    return cut_intervals
    

def get_non_overlapping_k_mers(act_seq, cuts, params, act_shift=0):
    
    contig_kmers = [] 
    
    for act_cut in cuts:
        act_sentence = act_seq[act_shift + act_cut[0]:act_cut[1]]
        Ls = len(act_sentence)
        kmer = params['tokenization']['kmer']
        Nk = math.floor((Ls-act_shift)/kmer)
        k_mers = [act_sentence[act_shift +i*kmer: act_shift +(i+1)*kmer] for i in range(Nk)]
        
        contig_kmers.append(k_mers)
    return contig_kmers

def get_lca_tokenized_segments(act_seq,sampling_intervals, params):
    k = params['tokenization']['kmer']
    lca_left = params['tokenization']['lca_left']
    lca_right = params['tokenization']['lca_right']
    minLs = params['tokenization']['minLs']
    IsAddingFlanking = True
    contig_kmers = []
    for act_interval in sampling_intervals:
        if IsAddingFlanking:
            act_segment = 'AA' + act_seq[act_interval[0]:act_interval[1]]
        else:
            act_segment = act_seq[act_interval[0]:act_interval[1]]
        Ntok = int((len(act_segment)-lca_left-lca_right)/3)
        #tokenized_segment = [act_segment[0+i*k-3:(i+1)*k+1] for i in range(1,Ntok+1)]
        tokenized_segment = [act_segment[0+i*k-3:(i+1)*k+1] for i in range(1,Ntok+1)]
        if len(tokenized_segment) > minLs:
            contig_kmers.append(tokenized_segment)
    return contig_kmers


def get_non_overlapping_sentence_non_overlapping_kmers(act_seq, params):
    
    kmer = params['tokenization']['kmer']
    shifts = params['tokenization']['shifts']
    nr_repetation = params['tokenization']['nr_repetation']
    vocabmap = params['tokenization']['vocabmap']
    tokenization_method = params['tokenization']['tokenization_method']

    
    tokenized_contigs = []
    for repeat_count in range(nr_repetation):
        #sampling_intervals = generate_random_intervals_with_length(act_seq, params)
        sampling_intervals = get_non_overlapping_contigous_largest_cuts(act_seq, params)

        for shift_id in shifts:
            if tokenization_method=='lca':
                #print('using local context aware tokenization!')
                contig_kmers = get_lca_tokenized_segments(act_seq,sampling_intervals, params)
            else:
                contig_kmers = get_non_overlapping_k_mers(act_seq, sampling_intervals, params, act_shift=shift_id)
            # Itt hozzá lehet adni a számokat
            contig_tokenized = tokenize_sentence(contig_kmers, vocabmap, 
                                                 sentence_length = params['tokenization']['Ls'], 
                                                 min_sentence_size = params['tokenization']['minLs'], 
                                                 unkwon_tsh = params['tokenization']['unkwon_tsh'])
            tokenized_contigs.extend(contig_tokenized)
    return tokenized_contigs

def get_non_overlapping_sentence_non_overlapping_kmers_contigs(contigs, params):
    
    tokenized_contigs = []
    for act_seq in contigs:
        tokenized_contigs.extend( get_non_overlapping_sentence_non_overlapping_kmers(act_seq, params))
    print('Shuffling the contigs!')
    random.shuffle(tokenized_contigs)
    return tokenized_contigs

def get_non_overlapping_tokenization_long(output_folder, act_batch_id, genome_batches, params, train_prob=1):
    pass
    if random.random()>train_prob:
        tt_prefix = 'test'
    else:
        tt_prefix = 'train'
    
    tokenization_method = params['tokenization']['tokenization_method']

    output_tokenized_padded_file = join(output_folder, tt_prefix + '_test_tokenized_batch_{0}.txt'.format(act_batch_id))
    output_tokenized_padded_file_hdf = join(output_folder, 'hdf5_' + tt_prefix + '_test_tokenized_batch_{0}.h5'.format(act_batch_id))
    act_batch = genome_batches[act_batch_id]
    print('Tokeneizing batch {0}'.format(act_batch_id))
    print(' output file: {0}'.format(output_tokenized_padded_file))

    shuffled_seqs = load_contigs(act_batch)
    if tokenization_method == 'lcas':
        print('Running Shifted LCA tokenization!')
        tokenized_contigs = lca_tokenize_contigs(shuffled_seqs, params)
    else:
        tokenized_contigs = get_non_overlapping_sentence_non_overlapping_kmers_contigs(shuffled_seqs, params)

    print('Collecting info!')
    nr_tokens = sum([len(sentence) for sentence in tokenized_contigs])
    nr_lines = len(tokenized_contigs)
    
    print('Number of tokens: {0}'.format(nr_tokens))
    print('Number of lines: {0}'.format(nr_lines))
    tokenized_outputs_all_array = np.array(tokenized_contigs,dtype="uint16")
    hdf_file = h5py.File(output_tokenized_padded_file_hdf, 'a')
    try:
        grp = hdf_file.create_group("training_data")
    except ValueError:
        del hdf_file['training_data']
    #ds = grp.create_dataset("sentences", data = tokenized_outputs_all_array, chunks=True, compression="gzip", maxshape=(None, tokenized_outputs_all_array.shape[1]))
    ds = grp.create_dataset("sentences", data = tokenized_outputs_all_array, chunks=True, maxshape=(None, tokenized_outputs_all_array.shape[1]))
    grp.create_dataset("dataset_size", data = (nr_lines, tokenized_outputs_all_array.shape[1]))
    hdf_file.close()

    
    
def convert_dataset_into_hdf_file(input_ds, output_file, IsCompression=False):
    print('Converting a dataset into HDF file!')
    print('Convert into numpy array!')
    numpydata = np.array(input_ds[:], dtype=np.int16)
    sentence_counts = numpydata.shape[0]
    sentence_lenght = numpydata.shape[1]
    
    print('Creating HDF5 file!')
    with h5py.File(output_file, 'w') as hout:
        try:
            grp = hout.create_group("training_data")
        except ValueError:
            del hout['training_data']
        print('Addin modell size')
        grp.create_dataset("dataset_size", data = (sentence_counts, sentence_lenght))
        print('Adding dataset!')
        if IsCompression:
            ds = grp.create_dataset("sentences", data = numpydata, 
                                    chunks=True, compression="gzip", 
                                    maxshape=(sentence_counts, sentence_lenght))
        else:
            ds = grp.create_dataset("sentences", data = numpydata, 
                                    chunks=True, #compression="gzip", 
                                    maxshape=(sentence_counts, sentence_lenght))


                
def get_random_segment_coordinates(Lseq,segment_length, expected_cov, min_segment_lenght):
    
    if Lseq< min_segment_lenght:
        #print('Segment is too short! Discarding')
        starting_coords=[]
    elif Lseq < segment_length:
        starting_coords = [0]
    else:
        nr_samples = math.ceil(expected_cov*Lseq/segment_length)
        #print('Number of segments: {0}'.format(nr_samples))
        if nr_samples > 1:
            Lseq_virt = Lseq-min_segment_lenght
            starting_coords = [int(Lseq_virt*random.random()) for i in range(nr_samples-1)]
            starting_coords.append(0)
            starting_coords = sorted(starting_coords)
        else:
            starting_coords = [0]
    return starting_coords   


def get_overlap_kmer_tokenization_seq(act_seq, params):
    """ Getting the overlaped tokenization of a sequence
    """
    kmer_size = params['tokenization']['kmer']
    Ls = params['tokenization']['Ls']
    max_prob = params['tokenization']['P_short']
    vocabmap = params['tokenization']['vocabmap']

    kmers = process_line(act_seq, kmer_size, Ls=Ls, max_prob=max_prob)
    tokens = tokenize_sentence(kmers, vocabmap, sentence_length = Ls)

    return tokens, kmers

#### LCA #####
def lca_tokenize_contig_contigous(act_seq, params):

    #print('Processing segment: {0}'.format(len(act_seq)))
    k = params['tokenization']['kmer']
    start_pos = 0
    act_shift = params['tokenization']['lca_shift']
    max_prob =  1-params['tokenization']['P_short']
    Ls = params['tokenization']['Ls']
    vocabmap = params['tokenization']['vocabmap']
    minLs = params['tokenization']['minLs']
    unkwon_tsh = params['tokenization']['unkwon_tsh']
    max_length = (Ls-1)*act_shift + k-act_shift
    cuts = get_segment_sizes_without_overlap(len(act_seq), k, max_prob=max_prob, max_length=max_length)
    #print('cuts', cuts)
    tokenized_outputs = []
    kmers_list = []
    Lseq=len(act_seq)
    for act_cut in cuts:
        for window in range(act_shift):
            act_segment=act_seq[start_pos+window:min(start_pos+window+act_cut,Lseq)]
            L = len(act_segment)
            expected_length = L-1*(k-act_shift)
            Ns = math.floor((L-1*(k-act_shift))/act_shift)
            kmers = [act_segment[i+i*(act_shift-1):i+i*(act_shift-1)+k] for i in range(Ns)]
            kmers_list.append(kmers)
        start_pos+=act_cut
    tokenized = tokenize_sentence(kmers_list, vocabmap, sentence_length = Ls, min_sentence_size = minLs, unkwon_tsh = unkwon_tsh)
    return tokenized, kmers_list

def lca_tokenize_contig_random_sampling(act_seq, params):
    k = params['tokenization']['kmer']
    start_pos = 0
    act_shift = params['tokenization']['lca_shift']
    max_prob =  1-params['tokenization']['P_short']
    Ls = params['tokenization']['Ls']
    vocabmap = params['tokenization']['vocabmap']
    minLs = params['tokenization']['minLs']
    unkwon_tsh = params['tokenization']['unkwon_tsh']
    expected_cov = params['tokenization']['coverage']
    max_length = (Ls-2)*act_shift + k-act_shift
    min_segment_lenght = (minLs-2)*act_shift + k-act_shift
    #print('max_length', max_length, 'min_segment_lenght', min_segment_lenght)
    Lseq = len(act_seq)

    tokenized = []
    kmers_list = []

    coords = get_random_segment_coordinates(Lseq,max_length, expected_cov, min_segment_lenght)
    #print(coords)
    for act_start_coord in coords:
        for window in range(act_shift):
            act_segment=act_seq[act_start_coord+window:min(act_start_coord+window+max_length,Lseq)]
            L = len(act_segment)
            #print('L:', L)
            expected_length = L-1*(k-act_shift)
            #print(expected_length)
            Ns = math.floor((L-1*(k-act_shift))/act_shift)
            kmers = [act_segment[i+i*(act_shift-1):i+i*(act_shift-1)+k] for i in range(Ns)]
            kmers_list.append(kmers)
    tokenized = tokenize_sentence(kmers_list, vocabmap, sentence_length = Ls, min_sentence_size = minLs, unkwon_tsh = unkwon_tsh)
    return tokenized, kmers_list



def lca_tokenize_contigs(contigs, params):
    
    tokenized_ds = []
    
    if 'segmenetation_type' in params['tokenization']:
        act_segmentation_type = params['tokenization']['segmenetation_type']
    else:
        act_segmentation_type = 'contigous'
    #print(act_segmentation_type, 'act_segmentation_type')
    for act_seq in contigs:
        if act_segmentation_type == 'contigous':        
            tokenized,kmers_list = lca_tokenize_contig_contigous(act_seq, params)
        else: 
            tokenized,kmers_list = lca_tokenize_contig_random_sampling(act_seq, params)
        tokenized_ds.extend(tokenized)
    
    print('Randomizing the tokenized segments!')
    random.shuffle(tokenized_ds)
    print('Finished')
    return tokenized_ds


def tokenize_set_of_fasta_file(files_to_process, Nbs, base_tokenout_folder, params):
    
    genome_batches, ranges = get_genome_batches(files_to_process, block_size = Nbs)
    tok_params = [(base_tokenout_folder, 
                act_batch_id,
                genome_batches,
                params,
                1) for act_batch_id in range(len(genome_batches[:]))]

    with multiprocessing.Pool(processes=12) as pool:
        pool.starmap(get_non_overlapping_tokenization_long, tok_params)
    len(files_to_process)
    pool.close()

    hdf5_folder = base_tokenout_folder
    outputfile_train = join(hdf5_folder, 'train.h5')
    input_prefix_train = 'hdf5_train'
    output_prefix = None
    concatenate_hdf5_files(hdf5_folder, outputfile_train, input_prefix_train, output_prefix, IsCompression=False)

    outputfile_train_randomozed = join(hdf5_folder, 'trainr.h5')
    with h5py.File(outputfile_train) as fin:
        numpydata = np.array(fin['training_data']['sentences'], dtype=np.uint16)

    sentence_counts = numpydata.shape[0]
    sentence_lenght = numpydata.shape[1]
    #random.shuffle(numpydata)
    print('randomize data!')
    rng = np.random.default_rng()
    rng.shuffle(numpydata)

    print('Creating HDF5 file!')
    IsCompression=False
    with h5py.File(outputfile_train_randomozed, 'w') as hout:
        try:
            grp = hout.create_group("training_data")
        except ValueError:
            del hout['training_data']
        print('Addin modell size')
        grp.create_dataset("dataset_size", data = (sentence_counts, sentence_lenght))
        print('Adding dataset!')
        if IsCompression:
            ds = grp.create_dataset("sentences", data = numpydata, 
                                    chunks=True, compression="gzip", 
                                    maxshape=(sentence_counts, sentence_lenght))
        else:
            ds = grp.create_dataset("sentences", data = numpydata, 
                                    chunks=True, #compression="gzip", 
                                    maxshape=(sentence_counts, sentence_lenght))


    cleanup_cmd = 'rm -rf {0}/hdf5_train_test_tokenized_batc*; mv {2} {1}'.format(hdf5_folder,outputfile_train, outputfile_train_randomozed) 
    print(cleanup_cmd)
    os.system(cleanup_cmd)
    
def sampling_from_bacterial_dataset(bacterial_ds_path, tokenizer, Npos):
    train_ds =dnadatasets.BabDataset(tokenizer, bacterial_ds_path, 10000, False)
    sampled_data = []
    for i, record in enumerate(train_ds):
        sampled_data.append(record)
        if i==Npos-1:
            break
    Npos
    len(sampled_data)
    sampled_data = np.stack(sampled_data)
    return sampled_data   

def get_finetune_dataset_data(phage_hdf_file, bac_data):
    """ Input: phage (hdf path), bac data (numpy array), balanced way.
    Output: randomized numpy arrays.
    """
    with h5py.File(phage_hdf_file) as fin:
        phage_data = np.array(fin['training_data']['sentences'], dtype=np.uint16)

    phage_labels = np.ones(phage_data.shape[0], dtype=np.int32)
    bac_labels = np.ones(bac_data.shape[0], dtype=np.int32)*0
    y = np.concatenate([phage_labels, bac_labels])
    x = np.concatenate((phage_data, bac_data))

    rs = np.random.permutation(x.shape[0])
    x=x[rs,:]
    y=y[rs]
    
    return x, y

def creating_finetune_hdf_file(x, y, output_filename):
    
    print('Creating HDF file: {0}'.format(output_filename))
    sentence_counts = x.shape[0]
    sentence_lenght = x.shape[1]

    print('  sentence_counts:  {0}\n  sentence_lenght:  {1}'.format(sentence_counts, sentence_lenght))

    with h5py.File(output_filename, 'w') as hout:
        try:
            grp = hout.create_group("training_data")
        except ValueError:
            del hout['training_data']
        print('Addin modell size')
        grp.create_dataset("dataset_size", data = (sentence_counts, sentence_lenght))

        dsx = grp.create_dataset("x", data = x, 
                                    chunks=True, 
                                    maxshape=(sentence_counts, sentence_lenght))
        dsy = grp.create_dataset("y", data = y, 
                                chunks=True,
                                maxshape=(sentence_counts))

    hout.close()
    print('Finished')
    
def convert_ids2seq(x, vocabmap, lca_shift, kmer):
    id2token = {v : k for k,v in vocabmap.items()}
    seqs = []

    for i in range(x.shape[0]):
        act_x =  x[i,:]
        last_nonzero_index = len(act_x) - np.argmax(act_x[::-1] != 0) - 1
        # Truncate the array by slicing up to the last non-zero index
        act_x = act_x[:last_nonzero_index + 1]
        act_tokens = [id2token[token_id] for token_id in act_x][1:-1]
        act_seq = act_tokens[0] + ''.join([act_token[kmer-lca_shift:kmer] for act_token in act_tokens[1:]])
        
        seqs.append(act_seq)
    return seqs

def get_split_intevals(sequence, segment_length):
    """
    Splits the sequence into intervals of specified length.

    Parameters:
    sequence (str): DNA sequence.
    segment_length (int): Length of the segment.

    Returns:
    list: List of tuples where each tuple represents an interval in the sequence.
    """

    return [(i, min( i+segment_length, len(sequence))) for i in range(0, len(sequence), segment_length)]
    


################## Tokeninization of sequences with ids and labels #####################


def tokenize_sequences_batch_with_ids(seqs, ids, class_params, pos_offset=0, interval_size=1024):

    tokenized_set_with_ids = {}
    for i, seq in enumerate(seqs):
        act_id = ids[i]
        tokenized_set_with_ids[act_id] = {}

        intervals = get_split_intevals(seq, interval_size)
        #intervals
        for act_interval in intervals:
            #act_interval
            act_interval_start = act_interval[0]
            act_interval_end = act_interval[1]
            act_sequence = seq[act_interval_start:act_interval_end]
            tokenizer_params = class_params['tokenization']
            max_sentence_length = tokenizer_params['max_sentence_length']
            if len(act_sequence) > max_sentence_length:
                raise ValueError('The lenght of the sequence is larger then the max allowed sentence length! {0}'.format(max_sentence_length))
            tokenized,kmers_list = lca_tokenize_contig_contigous(act_sequence, class_params)
            if len(tokenized)>0:
                act_segment = np.array(tokenized[pos_offset], dtype=np.int32)
                last_nonzero_index = len(act_segment) - np.argmax(act_segment[::-1] != 0) - 1
                act_segment = act_segment[:last_nonzero_index + 1]
                tokenized_set_with_ids[act_id][act_interval] = act_segment
            #act_segment = np.array(tokenized[pos_offset])
    
    return tokenized_set_with_ids

def parallell_tokenization_of_batch_of_sequences(data,class_params,pos_offset=0, interval_size=81, num_workers=10, batch_size=1000):

    seqs = list(data[:,-1]) #Last columns
    ids = list([i for i in range(len(seqs))])
    Ndata = len(data)


    batch_intervals = [(i, min( i+batch_size, Ndata)) for i in range(0, Ndata, batch_size)]
    params = [(seqs[interval[0]:interval[1]], 
            ids[interval[0]:interval[1]],
            class_params, 
            pos_offset,
            interval_size) for interval in batch_intervals]
    with Pool(processes=num_workers) as pool:
        result_list = pool.starmap(tokenize_sequences_batch_with_ids, params)
        
    tokenized_sets = {}
    for d in result_list:
        tokenized_sets.update(d)
    
    tokenized_meta = {}
    tokenized_meta['tokenizer_params'] =  class_params['tokenization']
    tokenized_meta['pos_offset'] = pos_offset
    tokenized_meta['interval_size'] = interval_size
    tokenized_meta['seqs'] = seqs
    tokenized_meta['ids'] = ids

    return tokenized_sets, tokenized_meta

def get_labeled_dataset_from_tokenized_dataset_only_specific_segment(data, tokenized_seqs, interval_size):

    X = []
    used_seq_ids = []
    for act_seq_pos_id in tokenized_seqs:
        try:
            act_vector = tokenized_seqs[act_seq_pos_id][(0,interval_size)]
            #act_vector
            X.append(act_vector)
            used_seq_ids.append(act_seq_pos_id)
        except KeyError:
            pass
        #break
    x=np.array(X, dtype=np.int32)
    zeros = np.zeros((x.shape[0], 1), dtype=np.int32)
    x = np.concatenate((x, zeros), axis=1)
    dataset_data = data[data.index.isin(used_seq_ids)]

    y = [0 for i in range(x.shape[0])]
    xgpu = torch.tensor(x)
    ygpu = torch.tensor(y)
    attention_mask = torch.full(x.shape, 1)
    token_type_ids = torch.full(x.shape, 0)

    return dataset_data, xgpu, ygpu, attention_mask, token_type_ids




"""
def process_batch_tokenize(output_folder, kmer_size, act_batch_id, genome_batches, train_prob=0.5):
    
    
    if random.random()>train_prob:
        tt_prefix = 'test'
    else:
        tt_prefix = 'train'
    
    output_tokenized_padded_file = join(output_folder, tt_prefix + '_test_tokenized_batch_{0}.txt'.format(act_batch_id))
    output_tokenized_padded_info_file = join(output_folder, 'info_' + tt_prefix + 'test_tokenized_batch_{0}.txt'.format(act_batch_id))
    
    act_batch = genome_batches[act_batch_id]
    
    print(act_batch[0:2])
    
    print('Tokeneizing batch {0}'.format(act_batch_id))
    print(' output file: {0}'.format(output_tokenized_padded_file))

    print('Parsing genome data!')
    shuffled_seqs = load_contigs(act_batch)
    print('K-merization and tokenization fist round!!')
    tokenized_outputs_all = kmerize_contigs(shuffled_seqs, nr_shuffling=2)
    print('Finished')

    print('Collecting info!')
    nr_tokens = sum([len(sentence) for sentence in tokenized_outputs_all])
    nr_lines = len(tokenized_outputs_all)
    with open(output_tokenized_padded_info_file, 'w') as fout:
        fout.write('{0},{1}\n'.format(nr_tokens, nr_lines))
    
    
    
    print('Writing into file!')
    with open(output_tokenized_padded_file, 'w') as fout:
        fout.write('\n'.join([' '.join([str(token) for token in sentence]) for sentence in tokenized_outputs_all]))
    fout.close()
"""