import numpy as np
import pandas as pd


FASTA_ID_COLUMN_NAME = "fasta_id"                   # fasta id
Y_PRED_COLUMN_NAME = "y_pred"                       # value should be either 1 (if phage) or 0 (not phage)
PHAGE_SCORE_COLUMN_NAME = "score_phage"             # value should be a score otherwise nan
NOT_PHAGE_SCORE_COLUMN_NAME = "score_not_phage"     # value should be a score otherwise nan
Y_TRUE_COLUMN_NAME = "y_true"                       # value should be either 1 or 0
METHOD_NAME_COLUMN_NAME = "method_name"             # name of the method


def seeker_fragment_result_parser(input_df):

    input_df.columns = [FASTA_ID_COLUMN_NAME, Y_PRED_COLUMN_NAME, PHAGE_SCORE_COLUMN_NAME]
    input_df[Y_TRUE_COLUMN_NAME] = input_df[FASTA_ID_COLUMN_NAME].str.extract(r'___(\d)$')
    input_df[Y_PRED_COLUMN_NAME] = input_df[Y_PRED_COLUMN_NAME].replace({'Bacteria': 0, 'Phage': 1}).astype(int)
    input_df[NOT_PHAGE_SCORE_COLUMN_NAME] =( 1 - input_df[PHAGE_SCORE_COLUMN_NAME]).round(2)
    input_df[METHOD_NAME_COLUMN_NAME] = "Seeker"
    input_df = input_df[[FASTA_ID_COLUMN_NAME, Y_TRUE_COLUMN_NAME, Y_PRED_COLUMN_NAME, PHAGE_SCORE_COLUMN_NAME, NOT_PHAGE_SCORE_COLUMN_NAME, METHOD_NAME_COLUMN_NAME]]

    return input_df

def seeker_microbiome_result_parser(input_df):

    input_df.columns = [FASTA_ID_COLUMN_NAME, Y_PRED_COLUMN_NAME, PHAGE_SCORE_COLUMN_NAME]
    input_df[Y_PRED_COLUMN_NAME] = input_df[Y_PRED_COLUMN_NAME].replace({'Bacteria': 0, 'Phage': 1}).astype(int)
    input_df[NOT_PHAGE_SCORE_COLUMN_NAME] =( 1 - input_df[PHAGE_SCORE_COLUMN_NAME]).round(2)
    input_df[METHOD_NAME_COLUMN_NAME] = "Seeker"
    input_df = input_df[[FASTA_ID_COLUMN_NAME, Y_PRED_COLUMN_NAME, PHAGE_SCORE_COLUMN_NAME, NOT_PHAGE_SCORE_COLUMN_NAME, METHOD_NAME_COLUMN_NAME]]

    return input_df


def metaphinder_fragment_result_parser(input_df):

    # if the read is under 500bp it wont not be processed by metaPhinder, and enters "not processed"
    input_df.replace("not processed", np.nan, inplace=True)
    input_df = input_df.drop(columns=["merged coverage [%]", "number of hits", "size[bp]"])
    input_df.columns = [FASTA_ID_COLUMN_NAME, Y_PRED_COLUMN_NAME, PHAGE_SCORE_COLUMN_NAME]
    input_df = input_df[input_df[Y_PRED_COLUMN_NAME].notna()]
    input_df[Y_TRUE_COLUMN_NAME] = input_df[FASTA_ID_COLUMN_NAME].str.extract(r'___(\d)$').astype(int)
    input_df[Y_PRED_COLUMN_NAME] = input_df[Y_PRED_COLUMN_NAME].replace({'negative': 0, 'phage': 1})
    input_df[PHAGE_SCORE_COLUMN_NAME] = pd.to_numeric(input_df[PHAGE_SCORE_COLUMN_NAME])
    input_df[NOT_PHAGE_SCORE_COLUMN_NAME] = ( 100 - input_df[PHAGE_SCORE_COLUMN_NAME]).round(3)
    input_df[METHOD_NAME_COLUMN_NAME] = "metaPhinder"
    input_df = input_df[[FASTA_ID_COLUMN_NAME, Y_TRUE_COLUMN_NAME, Y_PRED_COLUMN_NAME, PHAGE_SCORE_COLUMN_NAME, NOT_PHAGE_SCORE_COLUMN_NAME, METHOD_NAME_COLUMN_NAME]]

    return input_df

def metaphinder_microbiome_result_parser(input_df):

    # if the read is under 500bp it wont not be processed by metaPhinder, and enters "not processed"
    input_df.replace("not processed", np.nan, inplace=True) #
    input_df = input_df.drop(columns=["merged coverage [%]", "number of hits", "size[bp]"])
    input_df.columns = [FASTA_ID_COLUMN_NAME, Y_PRED_COLUMN_NAME, PHAGE_SCORE_COLUMN_NAME]
    input_df = input_df[input_df[Y_PRED_COLUMN_NAME].notna()] #
    input_df[Y_PRED_COLUMN_NAME] = input_df[Y_PRED_COLUMN_NAME].replace({'negative': 0, 'phage': 1}).astype(int)
    input_df[NOT_PHAGE_SCORE_COLUMN_NAME] =( 100 - input_df[PHAGE_SCORE_COLUMN_NAME]).round(3)
    input_df[METHOD_NAME_COLUMN_NAME] = "metaPhinder"

    return input_df


def genomad_microbiome_result_parser(input_df, genomad_virus_results_df):
    input_df = pd.read_csv(input_df, sep="\t")
    input_df["not_virus_score"] = (input_df["chromosome_score"] + input_df["plasmid_score"]).round(3)
    input_df = input_df.drop(columns=["chromosome_score", "plasmid_score"])
    input_df['found_in_df2'] = input_df['seq_name'].isin(genomad_virus_results_df['seq_name']).astype(int)
    input_df[METHOD_NAME_COLUMN_NAME] = "genomad"
    input_df.columns = [FASTA_ID_COLUMN_NAME, PHAGE_SCORE_COLUMN_NAME,NOT_PHAGE_SCORE_COLUMN_NAME, Y_PRED_COLUMN_NAME,METHOD_NAME_COLUMN_NAME]
    input_df = input_df[[FASTA_ID_COLUMN_NAME, Y_PRED_COLUMN_NAME, PHAGE_SCORE_COLUMN_NAME, NOT_PHAGE_SCORE_COLUMN_NAME, METHOD_NAME_COLUMN_NAME]]

    return input_df

def genomad_fragment_result_parser(input_df, genomad_virus_results_df):
    input_df = pd.read_csv(input_df, sep="\t")
    input_df["not_virus_score"] = (input_df["chromosome_score"] + input_df["plasmid_score"]).round(3)
    input_df = input_df.drop(columns=["chromosome_score", "plasmid_score"])
    input_df['found_in_df2'] = input_df['seq_name'].isin(genomad_virus_results_df['seq_name']).astype(int)
    input_df[Y_TRUE_COLUMN_NAME] = input_df["seq_name"].str.extract(r'___(\d)$').astype(int)
    input_df[METHOD_NAME_COLUMN_NAME] = "genomad"
    input_df.columns = [FASTA_ID_COLUMN_NAME, PHAGE_SCORE_COLUMN_NAME,NOT_PHAGE_SCORE_COLUMN_NAME, Y_PRED_COLUMN_NAME, Y_TRUE_COLUMN_NAME, METHOD_NAME_COLUMN_NAME]
    input_df = input_df[[FASTA_ID_COLUMN_NAME, Y_TRUE_COLUMN_NAME, Y_PRED_COLUMN_NAME, PHAGE_SCORE_COLUMN_NAME, NOT_PHAGE_SCORE_COLUMN_NAME, METHOD_NAME_COLUMN_NAME]]

    return input_df


def virsorter_microbiome_result_parser(input_df):

    input_df = input_df[["seqname", "dsDNAphage", "ssDNA", "max_score_group" ]]
    input_df[METHOD_NAME_COLUMN_NAME] = "virsorter2"
    input_df.columns = [FASTA_ID_COLUMN_NAME, PHAGE_SCORE_COLUMN_NAME,NOT_PHAGE_SCORE_COLUMN_NAME, Y_PRED_COLUMN_NAME, METHOD_NAME_COLUMN_NAME]
    input_df = input_df [input_df [Y_PRED_COLUMN_NAME].notna()]
    input_df = input_df [input_df [PHAGE_SCORE_COLUMN_NAME].notna()]
    input_df = input_df [input_df [NOT_PHAGE_SCORE_COLUMN_NAME].notna()]
    input_df[Y_PRED_COLUMN_NAME] = input_df [Y_PRED_COLUMN_NAME].replace({'ssDNA': 0, 'dsDNAphage': 1})
    input_df = input_df[[FASTA_ID_COLUMN_NAME, Y_PRED_COLUMN_NAME, PHAGE_SCORE_COLUMN_NAME, NOT_PHAGE_SCORE_COLUMN_NAME, METHOD_NAME_COLUMN_NAME]]
    input_df[FASTA_ID_COLUMN_NAME] = input_df[FASTA_ID_COLUMN_NAME].str.split('|').str[0]

    return input_df


def virsorter_fragment_result_parser(input_df):

    input_df = input_df[["seqname", "dsDNAphage", "ssDNA", "max_score_group" ]]
    input_df[METHOD_NAME_COLUMN_NAME] = "virsorter2"
    input_df.columns = [FASTA_ID_COLUMN_NAME, PHAGE_SCORE_COLUMN_NAME,NOT_PHAGE_SCORE_COLUMN_NAME, Y_PRED_COLUMN_NAME, METHOD_NAME_COLUMN_NAME]
    input_df = input_df [input_df [Y_PRED_COLUMN_NAME].notna()]
    input_df = input_df [input_df [PHAGE_SCORE_COLUMN_NAME].notna()]
    input_df = input_df [input_df [NOT_PHAGE_SCORE_COLUMN_NAME].notna()]
    input_df[Y_PRED_COLUMN_NAME] = input_df [Y_PRED_COLUMN_NAME].replace({'ssDNA': 0, 'dsDNAphage': 1})
    input_df = input_df[[FASTA_ID_COLUMN_NAME, Y_PRED_COLUMN_NAME, PHAGE_SCORE_COLUMN_NAME, NOT_PHAGE_SCORE_COLUMN_NAME, METHOD_NAME_COLUMN_NAME]]
    input_df[FASTA_ID_COLUMN_NAME] = input_df[FASTA_ID_COLUMN_NAME].str.split('|').str[0]
    input_df[Y_TRUE_COLUMN_NAME] = input_df[FASTA_ID_COLUMN_NAME].str.extract(r'___(\d)$').astype(int)
    input_df = input_df[[FASTA_ID_COLUMN_NAME, Y_TRUE_COLUMN_NAME, Y_PRED_COLUMN_NAME, PHAGE_SCORE_COLUMN_NAME, NOT_PHAGE_SCORE_COLUMN_NAME, METHOD_NAME_COLUMN_NAME]]

    return input_df


def deepvirfinder_microbiome_result_parser(input_df):

    input_df["y_pred"] = input_df["pvalue"].apply(lambda x: 1 if x < 0.05 else 0)
    input_df.rename(columns={"score": PHAGE_SCORE_COLUMN_NAME, "name": FASTA_ID_COLUMN_NAME }, inplace=True)
    input_df[NOT_PHAGE_SCORE_COLUMN_NAME] = ( 1 - input_df[PHAGE_SCORE_COLUMN_NAME]).round(3)
    input_df[PHAGE_SCORE_COLUMN_NAME] = input_df[PHAGE_SCORE_COLUMN_NAME].round(3)
    input_df = input_df.drop(columns=["len", "pvalue"])
    input_df[METHOD_NAME_COLUMN_NAME] = "deepVirFinder"
    input_df = input_df[[FASTA_ID_COLUMN_NAME, Y_PRED_COLUMN_NAME, PHAGE_SCORE_COLUMN_NAME, NOT_PHAGE_SCORE_COLUMN_NAME, METHOD_NAME_COLUMN_NAME]]

    return input_df

def deepvirfinder_fragment_result_parser(input_df):

    input_df["y_pred"] = input_df["pvalue"].apply(lambda x: 1 if x < 0.05 else 0)
    input_df.rename(columns={"score": PHAGE_SCORE_COLUMN_NAME, "name": FASTA_ID_COLUMN_NAME }, inplace=True)
    input_df[NOT_PHAGE_SCORE_COLUMN_NAME] = ( 1 - input_df[PHAGE_SCORE_COLUMN_NAME]).round(3)
    input_df[PHAGE_SCORE_COLUMN_NAME] = input_df[PHAGE_SCORE_COLUMN_NAME].round(3)
    input_df = input_df.drop(columns=["len", "pvalue"])
    input_df[METHOD_NAME_COLUMN_NAME] = "deepVirFinder"
    input_df[Y_TRUE_COLUMN_NAME] = input_df[FASTA_ID_COLUMN_NAME].str.extract(r'___(\d)$').astype(int)
    input_df = input_df[[FASTA_ID_COLUMN_NAME,Y_TRUE_COLUMN_NAME, Y_PRED_COLUMN_NAME, PHAGE_SCORE_COLUMN_NAME, NOT_PHAGE_SCORE_COLUMN_NAME, METHOD_NAME_COLUMN_NAME]]

    return input_df
