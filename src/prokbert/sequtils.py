# coding=utf-8

""" Library for sequence processing """

#KERDESEK
#-

#TODO
#def for getting params

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

import os
import pathlib

from mimetypes import guess_type
from functools import partial
import gzip

import pandas as pd

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

def segmentate_single_sequence(sequence, params, AddedHeader=False): #  1 db szekvencia, test esettel
    """ 
    Cuts a single sequence into segments.

    Parameters:
    sequences (string/list): Each sequence is represented as a string if AddedHeader is False, or as a list[fasta_id, description, source_file, sequence, orientation] if AddedHeader is True.
    params (dict): dictionary with parameters.
    AddedHeader (bool, optional): If True, the fasta ID and description in the input is included. Defaults to False.

    Returns:
    list: The segmentated sequence, represented as a string.
    """
    segmentation_type = params['segmentation']['segmentation_type']
    shifts = params['segmentation']['shifts']
    kmer = params['segmentation']['kmer']
    minLs = params['segmentation']['minLs']
    
    for v in params['segmentation']:
        print(v, ': ', params['segmentation'][v])
        
    if AddedHeader:
        act_seq = sequence[3]  # Get the sequence from the input list
    else:
        act_seq = sequence
    segments = []

    if len(act_seq) >= minLs:

        if segmentation_type == 'contigous':
            for i in range(0, len(act_seq) - kmer + 1, kmer):
                #if (i+kmer>len(act_seq))
                segment = act_seq[i:i + kmer]
                segments.append(segment)
                    
        elif segmentation_type == 'covering':
            for i in range(0, len(act_seq) - kmer + 1, shifts):
                segment = act_seq[i:i + kmer]
                segments.append(segment)

    else:
        print("Sequence ignored due to length constraint:", act_seq)


    return segments


def segmentate_sequences_from_list(sequences, params, AddedHeader=False):
    """ 
    Cuts sequences into segments.

    Parameters:
    sequences (list): List of sequences. Each sequence is represented as a string if AddedHeader is False, or as a list[fasta_id, description, source_file, sequence, orientation] if AddedHeader is True.
    params (dict): dictionary with parameters.
    AddedHeader (bool, optional): If True, the fasta ID and description in the input is included. Defaults to False.

    Returns:
    list<list>: List of segmentated sequences, each represented as a list of segments.
    """
    segmentation_type = params['segmentation']['segmentation_type']
    shifts = params['segmentation']['shifts']
    kmer = params['segmentation']['kmer']
    minLs = params['segmentation']['minLs']
    
    for v in params['segmentation']:
        print(v, ': ', params['segmentation'][v])

    segmentated_sequences = []
    
    for sequence in sequences:   
        if AddedHeader:
            act_seq = sequence[3]  # Get the sequence from the input list
        else:
            act_seq = sequence
        segments = []
    
        if len(act_seq) >= minLs:
            if segmentation_type == 'contigous':
                for i in range(0, len(act_seq) - kmer + 1, kmer):
                    #if (i+kmer>len(act_seq))
                    segment = act_seq[i:i + kmer]
                    segments.append(segment)
                        
            elif segmentation_type == 'covering':
                for i in range(0, len(act_seq) - kmer + 1, shifts):
                    segment = act_seq[i:i + kmer]
                    segments.append(segment)
                    
            segmentated_sequences.append(segments)
        
        else:
            print("Sequence ignored due to length constraint:", act_seq)
    
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
    
    
    
    
    
    
    
    
'''
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


def tokenize_sentence(sequences):
    """ 
    Tokenizes sentences.

    Parameters:
    sequences (list<list>): List of sequences. Each sequence is represented as a list containing the previously got kmers.
    
    vocabmap (map):
    Ls (int, optional):
    minLS (int, optional): minimum length of each sequence, the shorter ones will be ignored. Defaults to 80.
    unknown_tsh (float, optional):
    Returns:
    list: The tokenized sequences.
    """
    
    vocabmap = params['tokenization']['vocabmap']
    Ls = params['tokenization']['Ls']
    minLs = params['tokenization']['minLs']
    unknown_tsh = params['tokenization']['unknown_tsh']
    
    sentence_tokens = []
    unkw_tsh_count = int(sentence_length*unknown_tsh)
    for kmer_sentence in kmers:
        sentence = [vocabmap['[CLS]']]
        unkcount=0
        unkw_tsh_count = int(len(kmer_sentence)*unknown_tsh)

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
        
    
def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


def process_batch_tokenize_hdf5(output_folder, kmer_size,vocabmap, act_batch_id, genome_batches,Ls=512, train_prob=1, max_prob=1):
    
    
    if random.random()>train_prob:
        tt_prefix = 'test'
    else:
        tt_prefix = 'train'
    
    output_tokenized_padded_file = join(output_folder, tt_prefix + '_test_tokenized_batch_{0}.txt'.format(act_batch_id))

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
    unknown_tsh = params['tokenization']['unknown_tsh']
    max_length = (Ls-1)*act_shift + k-act_shift
    cuts = get_segment_sizes_without_overlap(len(act_seq), k, max_prob=max_prob, max_length=max_length)
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
    tokenized = tokenize_sentence(kmers_list, vocabmap, sentence_length = Ls, min_sentence_size = minLs, unknown_tsh = unknown_tsh)
    return tokenized, kmers_list

def lca_tokenize_contigs(contigs, params):
    
    tokenized_ds = []
    for act_seq in contigs:
        tokenized,kmers_list = lca_tokenize_contig_contigous(act_seq, params)
        tokenized_ds.extend(tokenized)
    
    print('Randomizing the tokenized segments!')
    random.shuffle(tokenized_ds)
    print('Finished')
    return tokenized_ds





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
"""'''
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
