import numpy as np
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool
from os.path import join, isfile, splitext
from os import listdir
import random
from transformers import LineByLineTextDataset
import tokenizers
from os.path import expanduser
localhome = expanduser("~")
print(localhome)
from copy import deepcopy
from prokbert.DataCollatorForDNA import *
#import prokbert.dnadatasets
#import prokbert.dnatokenizer 

from prokbert import dnatokenizer
from prokbert import dnadatasets

#import get_default_lca_tokenizer
import prokbert.prokbert_preproclib
from sklearn.metrics import top_k_accuracy_score
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, matthews_corrcoef
# Loading the data
import bz2
import pickle
import _pickle as cPickle
from transformers import AutoTokenizer, MegatronBertForSequenceClassification
from transformers import Trainer, TrainingArguments, BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling, BertModel, AutoModelForMaskedLM, AutoTokenizer, MegatronBertForMaskedLM
from transformers import AutoModelForSequenceClassification
import math
import re
import shutil
from prokbert.prokbert_preproclib import *
from torch.utils.data import TensorDataset, DataLoader


def get_topk_acc(prediction_data, act_index, masked_pos2token_ids, act_top_k=10):
    pos_index = masked_pos2token_ids[act_index]

    act_topk_val = prediction_data[act_index]['top_values'][0:act_top_k]
    act_topk_idx = prediction_data[act_index]['top_idx'][0:act_top_k]
    topk_acc = np.count_nonzero(act_topk_idx==pos_index)
    
    
    return topk_acc
    

def get_score_and_rank(prediction_data, act_index, masked_pos2token_ids):
    
    score = prediction_data[act_index]['hs']
    
    sorted_idx = score.argsort()[::-1]
    ranks = np.empty_like(sorted_idx)
    ranks[sorted_idx] = np.arange(len(score))
    act_rank = ranks[masked_pos2token_ids[act_index]]
    act_score = score[masked_pos2token_ids[act_index]]
    #act_score = score[masked_pos2token_ids[act_index]]
    
    return act_rank, act_score

    
def evaluation_masking_segments(prediction_data,masked_pos2token_ids,masked_pos2tokens, top_scores = [1,3,10,100]):
    
    act_index = 10
    
    eval_results = []
    
    for act_index in prediction_data:
    
        true_token_id = masked_pos2token_ids[act_index]
        true_token = masked_pos2tokens[act_index]
        score = prediction_data[act_index]['hs']
        Nt = score.shape[0]
        y_true = np.zeros(Nt)
        y_true[masked_pos2token_ids[act_index]]=1

        top_accs = [get_topk_acc(prediction_data, act_index, masked_pos2token_ids, act_top_k=top_k) for top_k in top_scores]

        act_rank, act_score = get_score_and_rank(prediction_data, act_index, masked_pos2token_ids)
        auc = roc_auc_score(y_true,score)

        eval_results.append([act_index,true_token_id,true_token, act_rank, act_score, auc] + top_accs)
        
    return eval_results

    
def build_masked_data(tokenized_sentence, masked_indeces, tokenizer, device):
    
    masked_tokenized_sentence = np.array(deepcopy(tokenized_sentence))
    mask_token = tokenizer.mask_token
    mask_token_id = tokenizer.mask_token_id
    
    masked_pos2token_ids = {}
    masked_pos2tokens = {}
    for masked_index in masked_indeces:
        masked_tokenized_sentence[masked_index] = mask_token
        masked_pos2tokens[masked_index] = tokenized_sentence[masked_index]
        try:
            masked_pos2token_ids[masked_index] = tokenizer.vocab[tokenized_sentence[masked_index]]
        except KeyError:
            print(masked_index)
            masked_pos2token_ids[masked_index] = tokenizer.unk_token_id

    token_ids = torch.tensor([tokenizer.encode(list(masked_tokenized_sentence))]).to(device)
    #token_ids = tokenizer.encode(list(masked_tokenized_sentence))
    
    
    
    return masked_tokenized_sentence, masked_pos2token_ids, masked_pos2tokens, token_ids
    

def get_prediction_last_layer_data(model, token_ids, masked_indeces, top_k=100):

    inputs= {'input_ids': token_ids}
    output = model(**inputs)
    last_hidden_state = output[0].squeeze()
    prediction_data = {} 
    for mask_index in masked_indeces:
        mask_hidden_state = last_hidden_state[mask_index+1]
        values, idx = torch.topk(mask_hidden_state, k=top_k, dim=0)
        
        prediction_data[mask_index] = {'hs': mask_hidden_state.detach().cpu().numpy(),
                                      'top_idx': idx.detach().cpu().numpy(),
                                      'top_values': values.detach().cpu().numpy()}
    return prediction_data


def eval_model_segment_masked_pos(model, tokenizer, masked_ids, tokenized_sentence, eval_id, device):
    results = []
    print('Evaluating masking task for segment.')
    for masked_indeces in masked_ids:
        idx_set = '(' + ', '.join([str(index) for index in masked_indeces]) + ')'
        #print('  idx_set:  ' + idx_set)
        masked_tokenized_sentence, masked_pos2token_ids, masked_pos2tokens, token_ids = build_masked_data(tokenized_sentence, masked_indeces, tokenizer, device)
        token_ids.to(device)
        prediction_data = get_prediction_last_layer_data(model, token_ids, masked_indeces, top_k=100)
        act_results = evaluation_masking_segments(prediction_data,masked_pos2token_ids,masked_pos2tokens)
        
        act_results2=[act_result.append(idx_set) for act_result in act_results]
        results.extend(act_results)

    results = pd.DataFrame(results)
    results.columns = ['Masking_position', 'masked_token_id', 'masked_token', 'pred_pos', 'logit', 'auc', 'acc_top1', 'acc_top3', 'acc_top110', 'acc_top100', 'index_set']
    results['eval_id'] = eval_id
    
    return results

    
    
    
def get_dnaK():


    ecoli_gene = """atgGGTAAAA TAATTGGTAT CGACCTGGGT ACTACCAACT CTTGTGTAGC GATTATGGAT
    GGCACCACTC CTCGCGTGCT GGAGAACGCC GAAGGCGATC GCACCACGCC TTCTATCATT
    GCCTATACCC AGGATGGTGA AACTCTAGTT GGTCAGCCGG CTAAACGTCA GGCAGTGACG
    AACCCGCAAA ACACTCTGTT TGCGATTAAA CGCCTGATTG GTCGCCGCTT CCAGGACGAA
    GAAGTACAGC GTGATGTTTC CATCATGCCG TTCAAAATTA TTGCTGCTGA TAACGGCGAC
    GCATGGGTCG AAGTTAAAGG CCAGAAAATG GCACCGCCGC AGATTTCTGC TGAAGTGCTG
    AAAAAAATGA AGAAAACCGC TGAAGATTAC CTGGGTGAAC CGGTAACTGA AGCTGTTATC
    ACCGTACCGG CATACTTTAA CGATGCTCAG CGTCAGGCAA CCAAAGACGC AGGCCGTATC
    GCTGGTCTGG AAGTAAAACG TATCATCAAC GAACCGACCG CAGCTGCGCT GGCTTACGGT
    CTGGACAAAG GCACTGGCAA CCGTACTATC GCGGTTTATG ACCTGGGTGG TGGTACTTTC
    GATATTTCTA TTATCGAAAT CGACGAAGTT GACGGCGAAA AAACCTTCGA AGTTCTGGCA
    ACCAACGGTG ATACCCACCT GGGGGGTGAA GACTTCGACA GCCGTCTGAT CAACTATCTG
    GTTGAAGAAT TCAAGAAAGA TCAGGGCATT GACCTGCGCA ACGATCCGCT GGCAATGCAG
    CGCCTGAAAG AAGCGGCAGA AAAAGCGAAA ATCGAACTGT CTTCCGCTCA GCAGACCGAC
    GTTAACCTGC CATACATCAC TGCAGACGCG ACCGGTCCGA AACACATGAA CATCAAAGTG
    ACTCGTGCGA AACTGGAAAG CCTGGTTGAA GATCTGGTAA ACCGTTCCAT TGAGCCGCTG
    AAAGTTGCAC TGCAGGACGC TGGCCTGTCC GTATCTGATA TCGACGACGT TATCCTCGTT
    GGTGGTCAGA CTCGTATGCC AATGGTTCAG AAGAAAGTTG CTGAGTTCTT TGGTAAAGAG
    CCGCGTAAAG ACGTTAACCC GGACGAAGCT GTAGCAATCG GTGCTGCTGT TCAGGGTGGT
    GTTCTGACTG GTGACGTAAA AGACGTACTG CTGCTGGACG TTACCCCGCT GTCTCTGGGT
    ATCGAAACCA TGGGCGGTGT GATGACGACG CTGATCGCGA AAAACACCAC TATCCCGACC
    AAGCACAGCC AGGTGTTCTC TACCGCTGAA GACAACCAGT CTGCGGTAAC CATCCATGTG
    CTGCAGGGTG AACGTAAACG TGCGGCTGAT AACAAATCTC TGGGTCAGTT CAACCTAGAT
    GGTATCAACC CGGCACCGCG CGGCATGCCG CAGATCGAAG TTACCTTCGA TATCGATGCT
    GACGGTATCC TGCACGTTTC CGCGAAAGAT AAAAACAGCG GTAAAGAGCA GAAGATCACC
    ATCAAGGCTT CTTCTGGTCT GAACGAAGAT GAAATCCAGA AAATGGTACG CGACGCAGAA
    GCTAACGCCG AAGCTGACCG TAAGTTTGAA GAGCTGGTAC AGACTCGCAA CCAGGGCGAC
    CATCTGCTGC ACAGCACCCG TAAGCAGGTT GAAGAAGCAG GCGACAAACT GCCGGCTGAC
    GACAAAACTG CTATCGAGTC TGCGCTGACT GCACTGGAAA CTGCTCTGAA AGGTGAAGAC
    AAAGCCGCTA TCGAAGCGAA AATGCAGGAA CTGGCACAGG TTTCCCAGAA ACTGATGGAA
    ATCGCCCAGC AGCAACATGC CCAGCAGCAG ACTGCCGGTG CTGATGCTTC TGCAAACAAC
    GCGAAAGATG ACGATGTTGT CGACGCTGAA TTTGAAGAAG TCAAAGACAA AAAAtaa"""
    dnaK = ''.join(ecoli_gene.split()).upper()
    
    return dnaK


###### Evaluation of pretraiing #####

def get_shift_counts(tokenizer_params, Nm):
    """
    Calculate the number of token required for masking at leasn Nm nucleotide position

    :param tokenizer_params: Dictionary containing tokenization parameters.
    :type: dict

    :param Nm: Number of masking nucleotides.
    :type: int

    return: Number of shifts required.
    :rtype: int

    Parameters:
    tokenizer_params (dict): Dictionary containing tokenization parameters.
    Nm (int): Number of masking nucleotides.

    Returns:
    int: Number of shifts required.
    """
    #tokenizer_params['lca_shift']*1*X-tokenizer_params['kmer']=Nm
    nr_shifts = (Nm+tokenizer_params['kmer']) / (tokenizer_params['lca_shift'])-1
    nr_shifts_ceil = int(np.ceil(nr_shifts))
    
    if nr_shifts !=nr_shifts_ceil:
        raise ValueError('It is not possible to exactly masking {0} nt with parameters: k={1}, shift={2}.'.format(Nm,
                                                                                                            tokenizer_params['kmer'],
                                                                                                            tokenizer_params['lca_shift']))
    return nr_shifts_ceil

def get_kmers_for_seq_positions(tokenizer_params, position):
    """
    Determine the starting and ending positions of tokens (k-mers) that covers a certain position in the sequence. (Only works for symmetric, invariant tokenization params!)

    Parameters:
    tokenizer_params (dict): Dictionary containing tokenization parameters.
    position (int): Position in the sequence.

    Returns:
    tuple: Starting and ending positions of k-mers in the sequence.
    """
        
    k=tokenizer_params['kmer']
    shift=tokenizer_params['lca_shift']

    pos_first_token = max(0, int(np.ceil( (position-k)/shift)))+1
    pos_last_token = int( (position-k)/shift + np.ceil(k/shift))+1
        
    #pos_last_token =  pos_first_token + int(k/shift)
    return pos_first_token, pos_last_token


def pretty_print_overlapping_sequence(segment, segment_kmers, tokenizer_params):
    """
    Format the sequence for pretty printing with overlapping k-mers.

    Parameters:
    segment (str): DNA sequence.
    segment_kmers (list): List of k-mers in the segment.
    tokenizer_params (dict): Dictionary containing tokenization parameters.

    Returns:
    list: List of formatted strings representing the sequence with overlapping k-mers.
    """
        
    k=tokenizer_params['kmer']
    shift = tokenizer_params['lca_shift']
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
    return lines
    
def get_all_masked_vectors(act_segment,act_segment_kmers, tokenizer, N_masked_token, tokenizer_params):
    """
    Generate masked vectors for the sequence, at all possible positions. It covers at least N_masked_token

    Parameters:
    act_segment (str): DNA sequence.
    act_segment_kmers (list): List of k-mers in the segment.
    tokenizer (Tokenizer): Tokenizer object.
    N_masked_token (int): Number of masking nucleoiteds.

    Returns:
    tuple: Array of masked vectors, DataFrame with information about masked vectors, actual segment sequence.
    """
    masked_token_id = tokenizer.mask_token_id
    k = tokenizer_params['kmer']
    lca_shift = tokenizer_params['lca_shift']
    
    sequences = convert_ids2seq(np.array([act_segment], dtype=np.int32), tokenizer_params['vocabmap'], 
                tokenizer_params['lca_shift'], 
                tokenizer_params['kmer'])
    act_segment_seq = sequences[0]
    
    Ni = get_shift_counts(tokenizer_params, N_masked_token)
    
    masking_all_positions = []
    segment_info = []

    last_nonzero_index = len(act_segment) - np.argmax(act_segment[::-1] != 0) - 1
    #last_nonzero_index
    act_segment = act_segment[:last_nonzero_index + 1]
    masking_all_positions = []
    segment_info = []
    #print('len(act_segment_kmers):', len(act_segment_kmers))
    #print('Ni', Ni)
    act_maskvector_id = 0
    for i in range(2, len(act_segment_kmers)-Ni+2):
        new_segment = deepcopy(act_segment)
        for j in range(i,i+Ni):
            new_segment[j]=masked_token_id

        
        masked_seq_pos_end = (i-1)*lca_shift + k
        masked_seq_pos_start = (i-2)*lca_shift + k
        masked_seq_pos_end = masked_seq_pos_start + N_masked_token        
        #masked_seq_pos_end =  ((i-2)+Ni-1)*lca_shift+k-lca_shift
        #masked_seq_pos_end = masked_seq_pos_start + lca_shift
        
        masked_seq = act_segment_seq[masked_seq_pos_start:masked_seq_pos_end]
        segment_info_act = [act_maskvector_id, i, masked_seq_pos_start, masked_seq_pos_end, masked_seq]
        segment_info.append(segment_info_act)
        masking_all_positions.append(new_segment)
        act_maskvector_id+=1


    descr = pd.DataFrame(segment_info,
                columns = ['act_maskvector_id', 'mask_index_start', 'masked_seq_pos_start', 'masked_seq_pos_end', 'masked_nts'])
    
    return np.array(masking_all_positions),descr, act_segment_seq



def get_pretraining_evaluation_batch_size_megatron_bert_eval_load_model(model_path, device):
    """
    Load Megatron BERT model and prepare for evaluation.

    Parameters:
    model_path (str): Path to the model.
    device (str): Device to load the model onto.

    Returns:
    MegatronBertForMaskedLM: Loaded model.
    """
    torch.cuda.empty_cache()
    model = MegatronBertForMaskedLM.from_pretrained(model_path, output_attentions=False)
    model.to(device)
    model.eval()
    num_gpus = torch.cuda.device_count()
    print('num_gpus: ', num_gpus)
    print('No of parameters: ', model.num_parameters()/1000000)
    if num_gpus > 1 and torch.cuda.device_count() >= num_gpus:
        model = torch.nn.DataParallel(model, device_ids=list(range(num_gpus)))
    
    
    return model

def get_mask_predictions(model, masked_vectors, batch_size=128):
    """
    Get predictions for masked vectors.

    Parameters:
    model (MegatronBertForMaskedLM): Loaded model.
    masked_vectors (array): Array of masked vectors.
    batch_size (int, optional): Batch size for the DataLoader. Defaults to 128.

    Returns:
    array: Final logits from the model.
    """
    torch.cuda.empty_cache()
    dataset = TensorDataset(masked_vectors)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    inputs = {'input_ids': None}  # Placeholder for inputs
    final_logits_list = []
    print('Running evaluation!')
    print('Number of batches:  {0}'.format(len(dataloader)))
    with torch.no_grad():
        for batch in dataloader:
            batch[0].shape
            inputs['input_ids'] = batch[0]
            outputs = model(**inputs)
            final_logits = outputs[0]
            final_logits_list.append(final_logits)
    del inputs
    final_logits = torch.concatenate(final_logits_list)
    torch.cuda.empty_cache()
    final_logits = final_logits.detach().cpu().numpy()

    
    return final_logits

def get_metric_for_masking_position(act_segment, masked_vectors, final_logits, id2token):
    """
    Compute metrics for each masking position in the sequence.

    Parameters:
    act_segment (str): DNA sequence.
    masked_vectors (array): Array of masked vectors.
    final_logits (array): Array of final logits from the model.
    id2token (dict): Dictionary mapping token IDs to tokens.

    Returns:
    DataFrame: DataFrame containing prediction results.
    """
        
    prediction_results = []

    for act_maskvector_id in range(masked_vectors.shape[0]):
        act_vectors = masked_vectors[act_maskvector_id,:]
        #act_vectors
        indices = np.where(act_vectors == 4)[0]
        #print('Number of masked indeces:  ', len(indices))

        for submaskpos, mask_index in enumerate(indices):
            mask_hidden_state = final_logits[act_maskvector_id, mask_index, :]

            top_indices = np.argsort(mask_hidden_state)[::-1]
            #top_indices.shape
            #values, idx = np.topk(mask_hidden_state, k=top_k, dim=0)
            #top_indices = np.argpartition(mask_hidden_state, -top_k)
            #mask_hidden_state[top_indices[-10:-1]]
            #topk_values = mask_hidden_state[topk_indices]

            true_token_id = act_segment[mask_index]
            true_token = id2token[true_token_id]
            #true_token_id, true_token
            score = mask_hidden_state
            Nt = score.shape[0]
            y_true = np.zeros(Nt)
            y_true[true_token_id] = 1
            sorted_idx = score.argsort()[::-1]
            ranks = np.empty_like(sorted_idx)
            ranks[sorted_idx] = np.arange(len(score))
            act_rank = ranks[true_token_id]
            act_score = score[true_token_id]
            auc = roc_auc_score(y_true,score)

            data = [act_maskvector_id, mask_index, submaskpos, act_rank, act_score, auc]
            prediction_results.append(data)

        #act_rank
        #act_score
        #break
    segment_pred_results = pd.DataFrame(prediction_results,
                columns = ['act_maskvector_id', 'mask_index', 'mask_index_pos', 'rank', 'score', 'roc-auc'])
    
    
    return segment_pred_results

def get_prediction_results_for_segment(act_gene, basename,  act_interval,N_masked_token, class_params, model,tokenizer, batch_size, model_name, cp):
    """
    Get prediction results for a specific segment of the DNA sequence.

    Parameters:
    act_gene (str): DNA sequence.
    basename (str): Base name for identification.
    act_interval (tuple): Interval to consider in the sequence.
    N_masked_token (int): Number of masking tokens.
    class_params (dict): Dictionary containing class parameters.
    model (MegatronBertForMaskedLM): Loaded model.
    tokenizer (Tokenizer): Tokenizer object.
    batch_size (int): Batch size for the DataLoader.
    model_name (str): Name of the model used for prediction.
    cp (str): Checkpoint for the model.

    Returns:
    DataFrame: DataFrame containing prediction results. The columns include information about the masked vectors, 
    actual sequence, masked positions, ID of the segment, starting and ending interval, positional offset, absolute
    positions of the masked sequence, context size, model name, checkpoint, k-mer size, and LCA shift.
    """
    # Your code
    masked_token_id = tokenizer.mask_token_id
    pred_masked_indeces_tokens = list([i for i in range(N_masked_token)])

    act_interval_start = act_interval[0]
    act_interval_end = act_interval[1]

    act_sequence = act_gene[act_interval_start:act_interval_end]

    # Kellene egy gyors check a max hosszról,
    tokenizer_params = class_params['tokenization']
    id2token = {v : k for k,v in tokenizer_params['vocabmap'].items()}
    max_sentence_length = tokenizer_params['max_sentence_length']
    if len(act_sequence) > max_sentence_length:
        raise ValueError('The lenght of the sequence is larger then the max allowed sentence length! {0}'.format(max_sentence_length))


    #print(act_sequence)

    tokenized,kmers_list = lca_tokenize_contig_contigous(act_sequence, class_params)
    print('Number of tokenized segments:  {0}'.format(len(tokenized)))
    print('Number of k-mers:              {0}'.format(len(kmers_list)))
    segment_pred_results_ls = []
    for pos_offset in range(len(tokenized)):
        print('Act offset:   {0}'.format(pos_offset))

        act_segment = np.array(tokenized[pos_offset])
        act_segment_kmers = kmers_list[pos_offset]
        act_segment_id = '{0}_i{1}-{2}_pos{3}'.format(basename, act_interval_start, act_interval_end, pos_offset)
        act_segment_id
        act_segment_kmers[0], act_segment_kmers[1]

        masked_vectors, descr, segment_seq = get_all_masked_vectors(act_segment,act_segment_kmers, tokenizer, N_masked_token, tokenizer_params)
        descr['masked_poss'] = N_masked_token
        descr['segment_id'] = act_segment_id
        descr['interval_start'] = act_interval_start
        descr['interval_end'] = act_interval_end
        descr['pos_offset'] = pos_offset
        descr['masked_seq_pos_start_abs']= descr['masked_seq_pos_start'] + act_interval_start + pos_offset
        descr['masked_seq_pos_end_abs']= descr['masked_seq_pos_end'] + act_interval_start + pos_offset
        descr['context_size'] = len(act_sequence)
        descr['model'] = model_name
        descr['cp'] = cp
        descr['kmer'] = tokenizer_params['kmer']
        descr['lca_shift'] = tokenizer_params['lca_shift']


        #masked_vectors[0]
        #masked_vectors
        final_logits = get_mask_predictions(model, torch.tensor(masked_vectors), batch_size=batch_size)
        metric_results = get_metric_for_masking_position(act_segment, masked_vectors, final_logits, id2token)
        pred_metric_results = metric_results[metric_results.mask_index_pos.isin(pred_masked_indeces_tokens)]
        pred_metric_results_descr = descr.merge(pred_metric_results, how='left', left_on= 'act_maskvector_id', right_on = 'act_maskvector_id')
        segment_pred_results_ls.append(pred_metric_results_descr)
    segment_pred_results = pd.concat(segment_pred_results_ls)
    
    return segment_pred_results




########      EVALUATING BINARY CLASSIFICATION  #######

def evaluate_binary_classification_bert_ds_depr(dataset,model, device, batch_size=100):
    eval_dataloader = DataLoader(dataset, batch_size)
    pred_results_ls = []
    for batch in eval_dataloader:

        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        pred_results = evaluate_binary_classification_bert_build_pred_results(outputs.logits,  batch['labels'])
        pred_results_ls.append(pred_results)
        print('{0} batch have been finished!'.format(len(pred_results_ls)))

    pred_results= np.concatenate(pred_results_ls)
    eval_results, eval_results_ls = evaluate_binary_classification_bert(pred_results)
    
    return eval_results, eval_results_ls

def evaluate_binary_classification_bert_ds(dataset, model, device, batch_size=100, num_gpus=1):
    # Make sure the model is on the specified device
    model.to(device)

    # If more than one GPU is available, use DataParallel for multi-GPU evaluation
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1 and torch.cuda.device_count() >= num_gpus:
        model = torch.nn.DataParallel(model, device_ids=list(range(num_gpus)))

    eval_dataloader = DataLoader(dataset, batch_size)
    pred_results_ls = []
    

    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        pred_results = evaluate_binary_classification_bert_build_pred_results(outputs.logits, batch['labels'])
        pred_results_ls.append(pred_results)
        print('{0} batch have been finished!'.format(len(pred_results_ls)))
        #del batch
        #torch.cuda.empty_cache()

    pred_results = np.concatenate(pred_results_ls)
    eval_results, eval_results_ls = evaluate_binary_classification_bert(pred_results)
    torch.cuda.empty_cache()


    return eval_results, eval_results_ls


def evaluate_binary_classification_bert_build_pred_results(logits, labels):
    
    predictions = torch.argmax(logits, dim=-1)
    p = predictions.detach().cpu()
    y = labels.detach().cpu()
    logits = logits.detach().cpu()
    pred = np.stack((y,p)).T
    pred_results = np.concatenate((pred, logits), axis=1)
    
    return pred_results
    

def evaluate_binary_classification_bert(pred_results):
    """ Calculating some metric based on the labels and predicted logit scores
    """
    
    y_true = pred_results[:,0]
    y_pred = pred_results[:,1]
    class_0_scores = pred_results[:,2]
    class_1_scores = pred_results[:,3]
    try:
        auc_class1 = roc_auc_score(y_true,class_0_scores)
    except ValueError:
        auc_class1=-1
    try:
        auc_class2 = roc_auc_score(y_true,class_1_scores)
    except ValueError:
        auc_class2=-1     

    
    acc = accuracy_score(y_true, y_pred, normalize=True)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    Np=tp+fn
    Nn=tn+fp
    eval_results = {
        'auc_class0' : auc_class1,
        'auc_class1' : auc_class2,
        'acc' : acc,
        'f1' : f1,
        'mcc': mcc,
        'recall': recall,
        'sensitivity': recall,
        'specificity': specificity,
        'tn' : tn,
        'fp' : fp,
        'fn' : fn,
        'tp' : tp,
        'Np' : Np,
        'Nn' : Nn
    }
    
    eval_results_ls = [auc_class1, auc_class2, f1, tn, fp, fn, tp, Np, Nn]
    
    return eval_results, eval_results_ls

def get_randomized_tensors_from_dataset_binary(act_dataset, setname, ds_type, device, max_sentence_lenght=79):
    
    try:
        xp = act_dataset[setname][1][ds_type]['X']
    except KeyError:
        xp = np.empty([0,max_sentence_lenght], dtype=np.int64)
    try:
        xn = act_dataset[setname][0][ds_type]['X']
    except KeyError:
        xn = np.empty([0, max_sentence_lenght], dtype=np.int64)

    x = np.concatenate((xp, xn))

    try:
        yp = act_dataset[setname][1][ds_type]['y']
    except KeyError:
        yp=np.empty([0], dtype=np.int64)
    try:
        yn = act_dataset[setname][0][ds_type]['y']
    except KeyError:
        yn=np.empty([0], dtype=np.int64)       

    y = np.concatenate((yp,yn))
    rs = np.random.permutation(x.shape[0])
    xgpu = torch.tensor(x[rs,:])
    ygpu= torch.tensor(y[rs])
    attention_mask = torch.full(x.shape, 1)
    token_type_ids = torch.full(x.shape, 0)

    return xgpu, ygpu, token_type_ids, attention_mask
    
def compute_metrics(eval_preds):
    ''' Binary classification evaluation
    '''

    logits, labels = eval_preds
    logits = torch.tensor(logits)
    labels = torch.tensor(labels)
    pred_results = evaluate_binary_classification_bert_build_pred_results(logits, labels)
    eval_results, eval_results_ls = evaluate_binary_classification_bert(pred_results)
    
    return eval_results



###### PROMOTER EVALUATION #######

def get_default_lca_tokenizer_get_default_tokenizer_params(actparams):
    print('Get default parameters for a tokenizer and its preprocessor')

    Ls = get_default_lca_tokenizer_get_default_value(actparams, 'Ls', 1024)
    kmer = get_default_lca_tokenizer_get_default_value(actparams, 'kmer', 6)
    prokbert_base_path = get_default_lca_tokenizer_get_default_value(actparams, 'prokbert_base_path', '.')
    lca_shift = get_default_lca_tokenizer_get_default_value(actparams, 'lca_shift', 1)
    minLs = get_default_lca_tokenizer_get_default_value(actparams, 'minLs', 2)
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
                    'minLs': minLs,
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





def get_default_lca_tokenizer_get_default_value(tokenizer_params, var_name, var_def_value = None):
    if var_name in tokenizer_params:
        var_value=tokenizer_params[var_name]
    else:
        var_value=var_def_value
    return var_value
    


def get_default_lca_tokenizer(tokenizer_params):
    print('Default LCA tokenizer')

    Ls = get_default_lca_tokenizer_get_default_value(tokenizer_params, 'Ls', 1024)
    kmer = get_default_lca_tokenizer_get_default_value(tokenizer_params, 'kmer', 6)
    prokbert_base_path = get_default_lca_tokenizer_get_default_value(tokenizer_params, 'prokbert_base_path', '.')
    lca_shift = get_default_lca_tokenizer_get_default_value(tokenizer_params, 'lca_shift', 1)
    token_vocab_file = get_default_lca_tokenizer_get_default_value(tokenizer_params, 'token_vocab_file', None)
    ds_type = 'lcas_k{0}_lcashift{1}'.format(kmer, lca_shift)
    
    tokenizer = dnatokenizer.BertTokenizer(do_upper_case = True, 
                                           vocab_file=token_vocab_file,  
                                           do_basic_tokenize=True)
    
    
    return tokenizer, ds_type




def promoter_get_tokenizer(prokbert_base_path, act_params):
    Ls = 512
    minLs = 0 #Minimum sentence length
    unkwon_tsh = 0.000005
    shifts = [0]
    kmer = int(act_params['kmer_size'])
    lca_shift = int(act_params['lca_shift'])
    
    nr_repetation = kmer*1
    
    token_vocab_file = join(prokbert_base_path, 'data/tokenizer/vocabs/bert-base-dna{0}/vocab.txt'.format(kmer))
    
    vocabmap = {line.strip(): i for i, line in enumerate(open(token_vocab_file))}
    params={}
    tokenization_params = {'kmer' : kmer,
                          'Ls' : Ls,
                          'minLs': minLs,
                          'unkwon_tsh': unkwon_tsh,
                          'vocabmap': vocabmap,
                          'shifts': shifts,
                          'nr_repetation': 1,
                          'coverage': 1,
                          'P_short': 0.1,
                          'tokenization_method': 'lcas',
                          'lca_shift': lca_shift,
                          'lca_left' : 0,
                          'lca_right': 0}
    params['tokenization']=tokenization_params
    ds_type = 'lcas_k{0}_lcashift{1}'.format(kmer, lca_shift)

    tokenizer = dnatokenizer.BertTokenizer(do_upper_case = True, 
                                           vocab_file=token_vocab_file,  
                                           do_basic_tokenize=True)
    
    
    return tokenizer, ds_type


def promoter_load_all_datasets(act_params, ds_type, device):
    
    if 'dataset_folder' in act_params:
        output_folder = act_params['dataset_folder']
        ppd_data = join(output_folder, act_params['input_dataset_path'])
    else:
        output_folder = '/scratch/fastscratch/NBL/training_datasets/ppd_seq_data/tokenized_ds'
        ppd_data = join(output_folder, act_params['input_dataset_path'])

    print('Loading dataset collections: {0}'.format(ppd_data))
    with open(ppd_data, 'rb') as f:
        prom_dataset = pickle.load(f)
    dataset_names = list(prom_dataset.keys())
    print(dataset_names)

    datasets={}
    for dataset_name in dataset_names:
        print(dataset_name)
        train_x, train_y, train_tt, train_am = get_randomized_tensors_from_dataset_binary(prom_dataset, dataset_name, ds_type, device)
        ds = dnadatasets.ProkDataset(train_x, train_y, train_tt, train_am)
        datasets[dataset_name]=ds

    


    return datasets




    
def promoter_get_training_data(act_params, ds_type, device):
    
    output_folder = '/scratch/fastscratch/NBL/training_datasets/ppd_seq_data/tokenized_ds'
    ppd_data = join(output_folder, 'tokenized_preprocessed_ds_eval_noecoli.pkl')
    ppd_data = join(output_folder, act_params['input_dataset_path'])

    print('Loading dataset collections: {0}'.format())

    with open(ppd_data, 'rb') as f:
        prom_dataset = pickle.load(f)
    prom_dataset.keys()

    from datasets.dataset_dict import DatasetDict
    from datasets import Dataset
    prom_dataset['train'][1].keys()
    
    train_x, train_y, train_tt, train_am = get_randomized_tensors_from_dataset_binary(prom_dataset, 'train', ds_type, device)
    test_x, test_y, test_tt, test_am  = get_randomized_tensors_from_dataset_binary(prom_dataset, 'test', ds_type, device)
    val_x, val_y, val_tt, val_am  = get_randomized_tensors_from_dataset_binary(prom_dataset, 'test', ds_type, device)
    ecoli_x, ecoli_y, ecoli_tt, ecoli_am  = get_randomized_tensors_from_dataset_binary(prom_dataset, 'ecolis70', ds_type, device)

    ds_train = dnadatasets.ProkDataset(train_x, train_y, train_tt, train_am)
    ds_test = dnadatasets.ProkDataset(test_x, test_y, test_tt, test_am)
    ds_eval = dnadatasets.ProkDataset(val_x, val_y, val_tt, val_am)
    ds_coli = dnadatasets.ProkDataset(ecoli_x, ecoli_y, ecoli_tt, ecoli_am)
    

    #ds_train=ds_eval
    #ds_test = ds_coli
    
    return ds_train, ds_test, ds_eval, ds_coli


def promoter_loading_pretrained_model(act_params):
    
    pretraining_model_dir = act_params['pretraining_model_dir']
    model_name = act_params['model_name']
    cp = act_params['checkpoint']
    
    output_cp = ''
    
    if cp=='latest':
        print('Loading the latest model!')
        [path_exists, largest_checkpoint_dir, largest_checkpoint, _] = prokbert_preproclib.check_model_existance_and_checkpoint(model_name,pretraining_model_dir )
        output_cp = str(largest_checkpoint)
        print('Loadin model: ' + largest_checkpoint_dir)
        model = MegatronBertForSequenceClassification.from_pretrained(largest_checkpoint_dir)
        print('No of parameters: ', model.num_parameters()/1000000)
    else:
        output_cp = str(cp)
        cp_model = join(pretraining_model_dir, model_name, 'checkpoint-' + str(cp))
        print('Loadin model: ' + cp_model)
        model = MegatronBertForSequenceClassification.from_pretrained(cp_model)
        print('No of parameters: ', model.num_parameters()/1000000)
    
    return model, output_cp


def promoter_get_act_model_name(act_params, model_cp):
    
    act_task = act_params['task_prefix']
    act_model = act_params['model_name']
    model_name = '{0}_{1}_cp{2}_LR{3}'.format(act_task,act_model, model_cp, act_params['learning_rate'])
    
    return model_name


def promoter_training_model(model,model_cp, act_params,ds_train,ds_test,tokenizer, ft_output_path = '', Nsample=10000, gradient_acc_step=1, logging_steps=100):
    
    #Nsample = 100000
    #Nsample = 50000
    #Nsample = 25000
    finetuned_model_name = promoter_get_act_model_name(act_params, model_cp)
    print('Creating a finetuned model: ', finetuned_model_name)
    
    train_output_path=join(ft_output_path, finetuned_model_name)
    train_batch_size = act_params['per_device_train_batch_size']
    eval_batch_size = act_params['per_device_eval_batch_size']
    lr = act_params['learning_rate']
    epoch_final = act_params['num_train_epochs']

    nr_gpus = int(torch.cuda.device_count())
    nr_gpus = 1
    #gradient_acc_step = 2
    samples_per_step = train_batch_size*gradient_acc_step*nr_gpus
    save_steps = int(np.round(Nsample/samples_per_step))
    print('save_steps: ', save_steps   )
    print('Cleaning output directory: ',train_output_path)
    
    if os.path.exists(train_output_path):
        # Remove the directory and all its contents
        shutil.rmtree(train_output_path)
    else:
        print("The directory does not exist")
    training_args = TrainingArguments(
        output_dir=train_output_path,
        overwrite_output_dir=True,
        num_train_epochs=epoch_final,
        save_steps=save_steps,
        save_total_limit=1000,
        logging_steps =logging_steps,
        learning_rate=lr,
        adam_epsilon=1e-6,
        #warmup_steps=10000,
        weight_decay=0.1,
        adam_beta1 = 0.9,
        adam_beta2 = 0.98,
        logging_first_step =True,
        gradient_accumulation_steps=gradient_acc_step,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size = eval_batch_size,
        evaluation_strategy="steps",
        do_eval = False
    #optim="adamw_torch"
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_test,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    #compute_metrics=compute_metrics
    )
    trainer.train()
    model.save_pretrained(train_output_path + '/checkpoint-0')
    torch.cuda.empty_cache()


def promoter_evaluate_finetuned_model_multiple_ds(ft_output_path, fine_tuned_model_name, eval_dss, batch_size, act_outoutfile, device, strategy='all', sampling_count=0):
    print('Evaluate multiple datasets!')

    path_exists, largest_checkpoint_dir, largest_checkpoint, chekcpoint_nr = prokbert_preproclib.check_model_existance_and_checkpoint(fine_tuned_model_name, ft_output_path)
    #print(chekcpoint_nr)
    print(f'Evaluation strategy:   {0}'.format(strategy))
    if strategy=='largest':
        print('Checking for zero ...')
        if 0 in chekcpoint_nr:
            print('Using final model only')
            chekcpoint_nr=[0]
        else:
            print('Using largest model only')
            chekcpoint_nr = [chekcpoint_nr[-1]]

    eval_results = []
    for cp in chekcpoint_nr:
        act_model_path = join(ft_output_path, fine_tuned_model_name, 'checkpoint-' + str(cp))
        print(act_model_path)
        model = MegatronBertForSequenceClassification.from_pretrained(act_model_path)
        print('No of parameters: ', model.num_parameters()/1000000)
        model.eval()
        model.to(device)
        for eval_ds_name in eval_dss:
            print('Evluating dataset: {0} on model: {1}'.format(eval_ds_name, act_model_path))
            eval_ds = eval_dss[eval_ds_name]
            eval_results_cp, eval_results_ls_cp = evaluate_binary_classification_bert_ds(eval_ds, model,  device, batch_size=batch_size)
            eval_results_cp['basemodel'] = fine_tuned_model_name
            eval_results_cp['cp'] = cp
            eval_results_cp['dataset'] = eval_ds_name
            #print(eval_results_cp)
            eval_results.append(eval_results_cp)
        del model
        torch.cuda.empty_cache()
    res = pd.DataFrame(eval_results)
    res.to_csv(act_outoutfile, sep='\t', index=False)
    
    return res


def get_train_test_validation_split_sentence_binary(sentences, binary_label, test_ratio=0.1, act_seed=23):
    
    N = len(sentences)
    val_ratio = (test_ratio/(1-test_ratio))
    train_ratio = (1-test_ratio)-(1-test_ratio)*val_ratio
    
    act_labels = [binary_label for i in range(N)]
    
    X_train, X_test, y_train, y_test = train_test_split(sentences, act_labels, test_size=test_ratio, random_state=act_seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, random_state=act_seed) # 0.25 x 0.8 = 0.2

    print('Splitting the dataset! Dataset size: {0}. Default label: {1}'.format(N, binary_label))
    print('Splitting params:\n  train_ratio:       {0}\n  validition ratio:  {1}\n  test ratio:        {2}'.format(train_ratio, val_ratio, test_ratio))
    
    dataset_sizes = """Dataset sizes:
  X_train:  {0}
  X_test:   {1}
  X_val:    {2}
""".format(len(X_train), len(X_test), len(X_val))
    print(dataset_sizes)
    
    return X_train, X_test, X_val, y_train, y_test, y_val

def get_non_overlapping_contigous_largest_cuts_lca(act_seq, Ls, kmer, lca_shift):
    
    Lc = len(act_seq)
    expected_length = (Ls-2)*lca_shift+(kmer-lca_shift)
    nr_cuts = math.ceil(Lc/expected_length)
    cut_intervals = [ (min(i*expected_length, Lc), min( (i+1)*expected_length,Lc)) for i in range(nr_cuts)]
    return cut_intervals

def promoter_add_tokenized_dataseet(prom_dataset, prokbert_base_path):
    import sys
    sys.path.insert(0,join(prokbert_base_path, 'bin'))
    from prokbert_preproclib import lca_tokenize_contig_contigous
    truncate_lenght = 79
    for lca_shift in [1, 2]:
        kmer = 6 # K-mer length
        Ls = 512 #sentence length, how many k-mers we are considering
        #lca_shift = 1
        minLs = 0 #Minimum sentence length
        unkwon_tsh = 0.000005
        shifts = [0]
        nr_repetation = kmer*1
        token_vocab_file = join(prokbert_base_path, 'data/tokenizer/vocabs/bert-base-dna{0}/vocab.txt'.format(kmer))
        vocabmap = {line.strip(): i for i, line in enumerate(open(token_vocab_file))}
        params={}
        tokenization_params = {'kmer' : kmer,
                            'Ls' : Ls,
                            'minLs': minLs,
                            'unkwon_tsh': unkwon_tsh,
                            'vocabmap': vocabmap,
                            'shifts': shifts,
                            'nr_repetation': 1,
                            'coverage': 1,
                            'P_short': 0.1,
                            'tokenization_method': 'lcas',
                            'lca_shift': lca_shift,
                            'lca_left' : 0,
                            'lca_right': 0}
        params['tokenization']=tokenization_params
        act_tokenized_ds_name = 'lcas_k{0}_lcashift{1}'.format(kmer, lca_shift)
        
        for act_set_id in prom_dataset:
            for act_label_id in prom_dataset[act_set_id]:
                print(act_set_id, act_label_id)
                act_seq_set = prom_dataset[act_set_id][act_label_id]['seq']
                act_tokenized_set = []
                for act_seq in act_seq_set:
                    sampling_intervals = get_non_overlapping_contigous_largest_cuts_lca(act_seq, Ls, kmer, lca_shift)
                    for segment_range in sampling_intervals:
                        act_sentence = act_seq[segment_range[0]:segment_range[1]]
                        tokenized_sentence,kmers_list = lca_tokenize_contig_contigous(act_sentence, params)
                        act_tokenized_set.extend(tokenized_sentence)

                tokenized_array = np.array(act_tokenized_set, dtype=np.int32)[:,0:truncate_lenght]
                N = tokenized_array.shape[0]
                label_array = np.full((N), act_label_id)
                prom_dataset[act_set_id][act_label_id][act_tokenized_ds_name] = {'X' : tokenized_array, 
                                                                                'y' : label_array}
    

def evaluation_get_defult_dataset(input_ids, model, device):
    x = torch.tensor(input_ids)
    y = torch.zeros(x.shape[0], dtype=torch.uint8)
    attention_mask = torch.full(x.shape, 1)
    token_type_ids = torch.full(x.shape, 0)

    eval_dataset = dnadatasets.ProkDataset(x, y, token_type_ids, attention_mask)
    eval_dataloader = DataLoader(eval_dataset, batch_size=x.shape[0])
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

    return eval_dataloader, eval_dataset, outputs


### Concatenating evaluation results

def generate_non_empty_tsv_files(start_path):
    for dirpath, _, filenames in os.walk(start_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if filename.endswith('.tsv') and os.path.getsize(filepath) > 0:
                yield filepath


def parse_model_string(model_string):
    # extract dataset name
    dataset_name = re.search(r'(.*?)_LCA_', model_string).group(1)
    
    # extract base model name
    base_model_name = 'LCA_' + re.search(r'_LCA_(.*?)_cp', model_string).group(1)
    
    # extract k and s values from base model name
    k, s = re.search(r'k(\d+)s(\d+)', base_model_name).groups()
    
    # extract cp_number, LR number, and other model parameters
    cp_number_str = re.search(r'_cp(\d+)', model_string).group(1)
    cp_number = int(cp_number_str)
    
    lr_match = re.search(r'_LR(\d+\.\d+)', model_string)
    lr_number = float(lr_match.group(1)) if lr_match else None
    
    return {
        'dataset_collection': dataset_name,
        'base_model_name': base_model_name,
        'token_k': int(k),
        'token_s': int(s),
        'pretrained_cp': cp_number,
        'learning_rate': lr_number
    }

def get_base_model_params(set_results):
    
    base_models = list(set_results[['basemodel']].drop_duplicates()['basemodel'])
    parsed_models = []
    for act_model in base_models:
        act_d = parse_model_string(act_model)
        act_d['basemodel']=act_model
        parsed_models.append(act_d)
    basemodel_params = pd.DataFrame(parsed_models)
    
    return basemodel_params

########## Evaluation batch size optimalization ##########


def get_optimal_evaluation_batch_size_megatron_bert_eval_load_model(model_path, device,act_model_class='MegatronBertForSequenceClassification', 
                                                                    isHiddenOutput=False, IsOutputAttentions=False):
    
    torch.cuda.empty_cache()
    from importlib import import_module
    ModelClass = getattr(import_module('transformers'), act_model_class)

    if IsOutputAttentions:
        output_attention_flag = True
    else:
        output_attention_flag = False

    if isHiddenOutput:
         model = ModelClass.from_pretrained(model_path, output_hidden_states=True,  output_attentions=output_attention_flag)
    else:
        model = ModelClass.from_pretrained(model_path, output_attentions=output_attention_flag)
    model.to(device)
    model.eval()
    num_gpus = torch.cuda.device_count()
    print('num_gpus: ', num_gpus)
    print('No of parameters: ', model.num_parameters()/1000000)
    if num_gpus > 1 and torch.cuda.device_count() >= num_gpus:
        model = torch.nn.DataParallel(model, device_ids=list(range(num_gpus)))
    
    
    return model
    

def get_optimal_evaluation_batch_size_megatron_bert_eval(model_path, dataset, device, min_batch_size=1, max_batch_size=16556, max_batch_increment_step=100):
    
    torch.cuda.empty_cache()

    gpu_memory_limit = torch.cuda.max_memory_allocated(device) * 0.9 #elérhető memória 90%
    print('gpu_memory_limit is:   {0}MB'.format(gpu_memory_limit/ (1024*1024)))
    print('Act batch step is:     {0}'.format(min_batch_size))
    act_batch_step = min_batch_size
    model = get_optimal_evaluation_batch_size_megatron_bert_eval_load_model(model_path, device)
    print(device)

    
    # exponential increase
    num_gpus = torch.cuda.device_count()
    i=0
    best_batch=True
    while best_batch and i<max_batch_increment_step:
        act_batch_step=act_batch_step*2
        print('Trying batch size:          {0}'.format(act_batch_step))
        eval_dataloader = DataLoader(dataset, act_batch_step)
        
        try:
            batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=act_batch_step, shuffle=True)))
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            gpu_memory_usage = torch.cuda.memory_allocated(device)
            print('gpu_memory_usage is:   {0} MB'.format(gpu_memory_usage/ (1024*1024)))

            if gpu_memory_usage <= gpu_memory_limit and gpu_memory_usage > max_memory_usage:
                max_memory_usage = gpu_memory_usage
            else:
                print('Too much memory usage???')
                #best_batch=False
                #act_batch_step = int(act_batch_step/2)


        except RuntimeError as e:
            torch.cuda.empty_cache()
            del model
            del batch
            torch.cuda.empty_cache()
            if "out of memory" in str(e):
                print(f"Batch size {act_batch_step} caused an out-of-memory error.")
                act_batch_step = int(act_batch_step/(num_gpus*2))
                best_batch=False
                model = get_optimal_evaluation_batch_size_megatron_bert_eval_load_model(model_path, device)
            else:
                raise e
        i+=1
    print('Second round. Adding batch size to maximazie memory usage!')
    torch.cuda.empty_cache()
    act_batch_step = int(act_batch_step)
    increment_step = max(int(act_batch_step/16), 8)
    optimal_batch_size = find_optimal_batch_size_memory(model, 
                                                    dataset,
                                                    device, 
                                                    min_batch_size=act_batch_step, 
                                                    max_batch_size=int(act_batch_step + act_batch_step/2), 
                                                    step=increment_step)
    print('New batch size: ', optimal_batch_size)
    act_batch_step = optimal_batch_size
    torch.cuda.empty_cache()
    del model
        
    print('\n\n_____________________________________')
    print('Best batch is:   {0}'.format(act_batch_step))
    print('Cleaning up!')
    print('\n_________________________________________')
    torch.cuda.empty_cache()
    return int(act_batch_step)
    
def find_optimal_batch_size_memory(model, dataset, device, min_batch_size=1, max_batch_size=512, step=8):
    
    num_gpus = torch.cuda.device_count()

    optimal_batch_size = None
    max_memory_usage = 0
    gpu_memory_limit = torch.cuda.max_memory_allocated(device) * 1.5  # Leave some room for other processes
    for batch_size in range(min_batch_size, max_batch_size + 1, step):
        torch.cuda.empty_cache()
        print(' Tryining new batch size:     {0}'.format(batch_size))
        print(' Act increment size:          {0}'.format(step))
        try:
            # Move the model to the target device
            model.to(device)

            # Create a random batch of input data with the current batch size
            sample_batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)))
            sample_batch = {k: v.to(device) for k, v in sample_batch.items()}

            # Perform a forward pass to measure the memory usage
            with torch.no_grad():
                outputs = model(**sample_batch)

            # Check GPU memory usage
            gpu_memory_usage = torch.cuda.memory_allocated(device)

            if gpu_memory_usage <= gpu_memory_limit and gpu_memory_usage > max_memory_usage:
                optimal_batch_size = batch_size
                max_memory_usage = gpu_memory_usage
            else:
                print('Too much memory usage???')
                break
                #return optimal_batch_size

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Batch size {batch_size} caused an out-of-memory error.")
                del sample_batch
                del model
                torch.cuda.empty_cache()
                break
            else:
                raise e
    optimal_batch_size = min_batch_size + int((optimal_batch_size - min_batch_size))/1

    print('skgjhjgkhfkThe optimal batch size: ', optimal_batch_size)
    return int(optimal_batch_size)




#### Promoter preprocessing ######

def get_true_labeled_prompred(input_data):
    
    input_data['y_true'] = input_data.apply(lambda x: get_true_labeled_prompred_from_id(x['ID']), axis=1)
    input_data['y_pred'] = input_data.apply(lambda x: 1 if x['Prediction']=='Promoter' else 0, axis=1)
    input_data['score_class1'] = input_data[' Probability Score']
    input_data['score_class0'] = 1- input_data[' Probability Score']
    
    return input_data
    

def get_true_labeled_prompred_from_id(id_string):
    
    label = None
    if id_string.startswith('>'):
        id_string=id_string[1:]

    if id_string.startswith('RAndom'):
        label = 0
    elif id_string.startswith('ECK12'):
        label = 1
    elif id_string[-1]=='0':
        label = 0
    elif id_string[-1]=='1':
        label = 1
    
    return label

