# coding=utf-8

# Lib for tokenization 

# Sequence preprocessing utils and functions

from mimetypes import guess_type
from functools import partial
import gzip

from Bio import SeqIO
import pandas as pd


def load_contigs(fasta_files_list, adding_reverse_complement=True, IsAddHeader=False, AsDataFrame=False):
    """ 
    Load contigs from a list of fasta files.

    :param:
    fasta_files_list (list): List of paths to fasta files.
    adding_reverse_complement (bool, optional): If True, add the reverse complement of each sequence. Defaults to True.
    IsAddHeader (bool, optional): If True, include the fasta ID and description in the output. Defaults to False.
    AsDataFrame (bool, optional): If True, return the sequences as a pandas DataFrame. Defaults to False.

    :return:
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
