#parser utils

import pandas as pd
import numpy as np


fasta_id_colname="fasta_id" # fasta id 
y_pred_colname="y_pred" # value should be either 1 (if phage) or 0 (not phage) 
phage_score_colname="score_phage" # value should be a score otherwise nan
not_phage_score_colname="score_not_phage" # value should be a score otherwise nan
y_true_colname="y_true" # value should be either 1 or 0
method_name_colname="method_name" # name of the method


# SEEKER

def seeker_fragment_result_parser(input_df):
    
    #input_df = pd.read_csv(input_csv_path, sep= '\t')
    input_df.columns = [fasta_id_colname, y_pred_colname, phage_score_colname]
    input_df[y_true_colname] = input_df[fasta_id_colname].str.extract(r'___(\d)$')
    input_df[y_pred_colname] = input_df[y_pred_colname].replace({'Bacteria': 0, 'Phage': 1}).astype(int)
    input_df[not_phage_score_colname] =( 1 - input_df[phage_score_colname]).round(2)
    input_df[method_name_colname] = "Seeker"
    input_df = input_df[[fasta_id_colname, y_true_colname, y_pred_colname, phage_score_colname, not_phage_score_colname, method_name_colname]]

    return input_df
    
def seeker_microbiome_result_parser(input_df):
    
    #input_df = pd.read_csv(input_csv_path, sep= '\t')
    input_df.columns = [fasta_id_colname, y_pred_colname, phage_score_colname]
    input_df[y_pred_colname] = input_df[y_pred_colname].replace({'Bacteria': 0, 'Phage': 1}).astype(int)
    input_df[not_phage_score_colname] =( 1 - input_df[phage_score_colname]).round(2)
    input_df[method_name_colname] = "Seeker"
    input_df = input_df[[fasta_id_colname, y_pred_colname, phage_score_colname, not_phage_score_colname, method_name_colname]]

    return input_df


# METAPHINDER

def metaphinder_fragment_result_parser(input_df):
    
    #input_df = pd.read_csv(metaPhinder_fragments_path, sep="\t")
    #if the read is under 500bp it wont not be processed by metaPhinder, and enters "not processed"
    input_df.replace("not processed", np.nan, inplace=True)
    input_df = input_df.drop(columns=["merged coverage [%]", "number of hits", "size[bp]"])
    input_df.columns = [fasta_id_colname, y_pred_colname, phage_score_colname]
    input_df = input_df[input_df[y_pred_colname].notna()]
    input_df[y_true_colname] = input_df[fasta_id_colname].str.extract(r'___(\d)$').astype(int)
    input_df[y_pred_colname] = input_df[y_pred_colname].replace({'negative': 0, 'phage': 1})
    input_df[phage_score_colname] = pd.to_numeric(input_df[phage_score_colname])
    input_df[not_phage_score_colname] = ( 100 - input_df[phage_score_colname]).round(3)
    input_df[method_name_colname] = "metaPhinder"
    input_df = input_df[[fasta_id_colname, y_true_colname, y_pred_colname, phage_score_colname, not_phage_score_colname, method_name_colname]]

    return input_df

def metaphinder_microbiome_result_parser(input_df):
    
    #input_df = pd.read_csv(input_csv_path, sep= '\t')
    #if the read is under 500bp it wont not be processed by metaPhinder, and enters "not processed"
    input_df.replace("not processed", np.nan, inplace=True) #
    input_df = input_df.drop(columns=["merged coverage [%]", "number of hits", "size[bp]"])
    input_df.columns = [fasta_id_colname, y_pred_colname, phage_score_colname] 
    input_df = input_df[input_df[y_pred_colname].notna()] # 
    input_df[y_pred_colname] = input_df[y_pred_colname].replace({'negative': 0, 'phage': 1}).astype(int)
    input_df[not_phage_score_colname] =( 100 - input_df[phage_score_colname]).round(3)
    input_df[method_name_colname] = "metaPhinder"
    
    return input_df


# GENOMAD

def genomad_microbiome_result_parser(input_df, genomad_virus_results_df):
    input_df = pd.read_csv(genomad_tsv, sep="\t")
    input_df["not_virus_score"] = (input_df["chromosome_score"] + input_df["plasmid_score"]).round(3)
    input_df = input_df.drop(columns=["chromosome_score", "plasmid_score"])
    input_df['found_in_df2'] = input_df['seq_name'].isin(genomad_virus_results_df['seq_name']).astype(int)
    input_df[method_name_colname] = "genomad"
    input_df.columns = [fasta_id_colname, phage_score_colname,not_phage_score_colname, y_pred_colname,method_name_colname]
    input_df = input_df[[fasta_id_colname, y_pred_colname, phage_score_colname, not_phage_score_colname, method_name_colname]]

    return input_df

def genomad_fragment_result_parser(input_df, genomad_virus_results_df):
    input_df = pd.read_csv(fragments_tsv, sep="\t")
    input_df["not_virus_score"] = (input_df["chromosome_score"] + input_df["plasmid_score"]).round(3)
    input_df = input_df.drop(columns=["chromosome_score", "plasmid_score"])
    input_df['found_in_df2'] = input_df['seq_name'].isin(genomad_virus_results_df['seq_name']).astype(int)
    input_df[y_true_colname] = input_df["seq_name"].str.extract(r'___(\d)$').astype(int)
    input_df[method_name_colname] = "genomad"
    input_df.columns = [fasta_id_colname, phage_score_colname,not_phage_score_colname, y_pred_colname, y_true_colname, method_name_colname]
    input_df = input_df[[fasta_id_colname, y_true_colname, y_pred_colname, phage_score_colname, not_phage_score_colname, method_name_colname]]

    return input_df


# VIRSORTER2

def virsorter_microbiome_result_parser(input_df):
    
    input_df = input_df[["seqname", "dsDNAphage", "ssDNA", "max_score_group" ]]
    input_df[method_name_colname] = "virsorter2"
    input_df.columns = [fasta_id_colname, phage_score_colname,not_phage_score_colname, y_pred_colname, method_name_colname]
    input_df = input_df [input_df [y_pred_colname].notna()]
    input_df = input_df [input_df [phage_score_colname].notna()]
    input_df = input_df [input_df [not_phage_score_colname].notna()]
    input_df[y_pred_colname] = input_df [y_pred_colname].replace({'ssDNA': 0, 'dsDNAphage': 1})
    input_df = input_df[[fasta_id_colname, y_pred_colname, phage_score_colname, not_phage_score_colname, method_name_colname]]
    input_df[fasta_id_colname] = input_df[fasta_id_colname].str.split('|').str[0]

    return input_df

def virsorter_fragment_result_parser(input_df):
    
    input_df = input_df[["seqname", "dsDNAphage", "ssDNA", "max_score_group" ]]
    input_df[method_name_colname] = "virsorter2"
    input_df.columns = [fasta_id_colname, phage_score_colname,not_phage_score_colname, y_pred_colname, method_name_colname]
    input_df = input_df [input_df [y_pred_colname].notna()]
    input_df = input_df [input_df [phage_score_colname].notna()]
    input_df = input_df [input_df [not_phage_score_colname].notna()]
    input_df[y_pred_colname] = input_df [y_pred_colname].replace({'ssDNA': 0, 'dsDNAphage': 1})
    input_df = input_df[[fasta_id_colname, y_pred_colname, phage_score_colname, not_phage_score_colname, method_name_colname]]
    input_df[fasta_id_colname] = input_df[fasta_id_colname].str.split('|').str[0]
    input_df[y_true_colname] = input_df[fasta_id_colname].str.extract(r'___(\d)$').astype(int)
    input_df = input_df[[fasta_id_colname, y_true_colname, y_pred_colname, phage_score_colname, not_phage_score_colname, method_name_colname]]
    
    return input_df

# DEEPVIRFINDER

def deepvirfinder_microbiome_result_parser(input_df):
    
    input_df["y_pred"] = input_df["pvalue"].apply(lambda x: 1 if x < 0.05 else 0)
    input_df.rename(columns={"score": phage_score_colname, "name": fasta_id_colname }, inplace=True)
    input_df[not_phage_score_colname] = ( 1 - input_df[phage_score_colname]).round(3)
    input_df[phage_score_colname] = input_df[phage_score_colname].round(3)
    input_df = input_df.drop(columns=["len", "pvalue"])
    input_df[method_name_colname] = "deepVirFinder"
    input_df = input_df[[fasta_id_colname, y_pred_colname, phage_score_colname, not_phage_score_colname, method_name_colname]]

    return input_df

def deepvirfinder_fragment_result_parser(input_df):
    
    input_df["y_pred"] = input_df["pvalue"].apply(lambda x: 1 if x < 0.05 else 0)
    input_df.rename(columns={"score": phage_score_colname, "name": fasta_id_colname }, inplace=True)
    input_df[not_phage_score_colname] = ( 1 - input_df[phage_score_colname]).round(3)
    input_df[phage_score_colname] = input_df[phage_score_colname].round(3)
    input_df = input_df.drop(columns=["len", "pvalue"])
    input_df[method_name_colname] = "deepVirFinder"
    input_df[y_true_colname] = input_df[fasta_id_colname].str.extract(r'___(\d)$').astype(int)
    input_df = input_df[[fasta_id_colname,y_true_colname, y_pred_colname, phage_score_colname, not_phage_score_colname, method_name_colname]]
    
    return input_df