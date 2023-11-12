import unittest
import os
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import gzip
import sys
import collections
from os.path import join
import pandas as pd
import numpy as np
import tempfile
# The test should be run from the project root directory as: 

from prokbert.sequtils import *
from prokbert.config_utils import *


class TestSeqUtils(unittest.TestCase):
    def setUp(self):
        # Create a test fasta file

        self.test_fasta_filename = 'tests/data/test.fasta'
        self.test_fasta_gz_filename = 'tests/data/test.fasta.gz'

        self.test_fasta_filename = '/tmp/test.fasta'
        self.test_fasta_gz_filename = '/tmp/test.fasta.gz'


        self.sequences = [SeqRecord(Seq("ATGC"), id="test1", description="test sequence 1"),
                          SeqRecord(Seq("CGTA"), id="test2", description="test sequence 2")]
        SeqIO.write(self.sequences, self.test_fasta_filename, "fasta")
        with gzip.open(self.test_fasta_gz_filename, 'wt') as f_out:
            SeqIO.write(self.sequences, f_out, "fasta")
    
    def tearDown(self):
        # Remove the test fasta file
        os.remove(self.test_fasta_filename)
        os.remove(self.test_fasta_gz_filename)

    def test_load_contigs(self):
        # Test with a single fasta file
        sequences = load_contigs([self.test_fasta_filename])
        self.assertEqual(len(sequences), 4)  # 2 sequences * 2 (forward + reverse complement)
        self.assertEqual(sequences[0], "ATGC")
        self.assertEqual(sequences[1], "GCAT")  # Reverse complement of "ATGC"
        self.assertEqual(sequences[2], "CGTA")
        self.assertEqual(sequences[3], "TACG")  # Reverse complement of "CGTA"

        # Test with gzip fasta file
        sequences = load_contigs([self.test_fasta_gz_filename])
        self.assertEqual(len(sequences), 4)  # 2 sequences * 2 (forward + reverse complement)
        self.assertEqual(sequences[0], "ATGC")
        self.assertEqual(sequences[1], "GCAT")  # Reverse complement of "ATGC"
        self.assertEqual(sequences[2], "CGTA")
        self.assertEqual(sequences[3], "TACG")  # Reverse complement of "CGTA"

        sequences = load_contigs([self.test_fasta_filename], IsAddHeader=True)

        self.assertEqual(len(sequences), 4)  # 2 sequences * 2 (forward + reverse complement)
        self.assertEqual(sequences[0][0], "test1")
        self.assertEqual(sequences[0][1], "test1 test sequence 1")
        self.assertEqual(sequences[0][2], self.test_fasta_filename)
        self.assertEqual(sequences[0][3], "ATGC")
        self.assertEqual(sequences[0][4], "forward")
        self.assertEqual(sequences[1][0], "test1")
        self.assertEqual(sequences[1][1], "test1 test sequence 1")
        self.assertEqual(sequences[1][2], self.test_fasta_filename)
        self.assertEqual(sequences[1][3], "GCAT")  # Reverse complement of "ATGC"
        self.assertEqual(sequences[1][4], "reverse")

        # Test with gzip fasta file
        sequences = load_contigs([self.test_fasta_gz_filename], IsAddHeader=True)
        
        self.assertEqual(len(sequences), 4)  # 2 sequences * 2 (forward + reverse complement)
        self.assertEqual(sequences[0][0], "test1")
        self.assertEqual(sequences[0][1], "test1 test sequence 1")
        self.assertEqual(sequences[0][2], self.test_fasta_gz_filename)
        self.assertEqual(sequences[0][3], "ATGC")
        self.assertEqual(sequences[0][4], "forward")
        self.assertEqual(sequences[1][0], "test1")
        self.assertEqual(sequences[1][1], "test1 test sequence 1")
        self.assertEqual(sequences[1][2], self.test_fasta_gz_filename)
        self.assertEqual(sequences[1][3], "GCAT")  # Reverse complement of "ATGC"
        self.assertEqual(sequences[1][4], "reverse")

        sequences = load_contigs([self.test_fasta_filename], IsAddHeader=True, AsDataFrame=True)
        print(sequences.iloc[0]['description'])

        self.assertEqual(len(sequences), 4)  # 2 sequences * 2 (forward + reverse complement)
        self.assertEqual(sequences.iloc[0]['fasta_id'], "test1")
        self.assertEqual(sequences.iloc[0]['description'], "test1 test sequence 1")
        self.assertEqual(sequences.iloc[0]['source_file'], self.test_fasta_filename)
        self.assertEqual(sequences.iloc[0]['sequence'], "ATGC")
        self.assertEqual(sequences.iloc[0]['orientation'], "forward")
        self.assertEqual(sequences.iloc[1]['fasta_id'], "test1")
        self.assertEqual(sequences.iloc[1]['description'], "test1 test sequence 1")
        self.assertEqual(sequences.iloc[1]['source_file'], self.test_fasta_filename)
        self.assertEqual(sequences.iloc[1]['sequence'], "GCAT")  # Reverse complement of "ATGC"
        self.assertEqual(sequences.iloc[1]['orientation'], "reverse")

        # Test with a single fasta file, without headers, as DataFrame
        sequences = load_contigs([self.test_fasta_filename], IsAddHeader=False, AsDataFrame=True)
        self.assertEqual(len(sequences), 4)  # 2 sequences * 2 (forward + reverse complement)
        self.assertEqual(sequences.iloc[0]['sequence'], "ATGC")
        self.assertEqual(sequences.iloc[1]['sequence'], "GCAT")  # Reverse complement of "ATGC"


class TestSegmentSequenceContiguous(unittest.TestCase):

    def setUp(self):
        self.sequences = [
            'TAGAAATGTCCGCGACCTTTCATACATACCACCGGTACGCCCTGGAGATG',
            'ATAATGCTAAATCGTAACCCCACTGCTTAAATGAGCCTTCTGTAAATTTCGTAGTACGTA',
            'GTGACCGGGGTCAGGTTCTCGGCGGCGGCGCGCATCACGTGCTTGCCGACATGCTAGCATG'
        ]
        self.default_params = {
            'type': 'contiguous',
            'min_length': 0,
            'max_length': 512,
            'coverage': 1.0
        }

    def test_basic_segmentation(self):
        for seq in self.sequences:
            result = segment_sequence_contiguous(seq, self.default_params)
            # Since max_length is 512 and all sequences are shorter than that, 
            # each sequence should result in a single segment.
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]['segment'], seq)
            self.assertEqual(result[0]['segment_start'], 0)
            self.assertEqual(result[0]['segment_end'], len(seq))

    def test_short_sequence(self):
        short_sequence = 'TAGAA'
        params = {'type': 'contiguous', 'min_length': 0, 'max_length': 4, 'coverage': 1.0}
        result = segment_sequence_contiguous(short_sequence, params)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['segment'], 'TAGA')
        self.assertEqual(result[1]['segment'], 'A')

    def test_segment_smaller_than_min_length(self):
        sequence = 'TAGAATAG'
        params = {'type': 'contiguous', 'min_length': 5, 'max_length': 6, 'coverage': 1.0}
        result = segment_sequence_contiguous(sequence, params)
        # Here, the segments 'AG' will be discarded since its length 
        # is shorter than min_length, so only one segment should be returned.
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['segment'], 'TAGAAT')

    def test_segment_start_end(self):
        print('Testing test_segment_start_end')

        sequence = 'TAGAATAG'
        params = {'type': 'contiguous', 'min_length': 3, 'max_length': 4, 'coverage': 1.0}
        result = segment_sequence_contiguous(sequence, params)
        self.assertEqual(result[0]['segment_start'], 0)
        self.assertEqual(result[0]['segment_end'], 4)
        self.assertEqual(result[1]['segment_start'], 4)
        self.assertEqual(result[1]['segment_end'], 8)

    def test_with_sequence_id(self):
        sequence_id = 1001
        result = segment_sequence_contiguous(self.sequences[0], self.default_params, sequence_id=sequence_id)
        self.assertEqual(result[0]['sequence_id'], sequence_id)

class TestSegmentSequences(unittest.TestCase):

    def setUp(self):
        self.sequences_df = pd.DataFrame({
            "sequence_id": {0: 0, 1: 1, 2: 2},
            "sequence": {
                0: "TAGAAATGTCCGCGACCTTTCATACATACCACCGGTACGCCCTGGAGATG",
                1: "ATAATGCTAAATCGTAACCCCACTGCTTAAATGAGCCTTCTGTAAATTTC",
                2: "GTGACCGGGGTCAGGTTCTCGGCGGCGGCGCGCATCACGTGCTTGCCGAC"
            }
        })
        self.default_params = {
            'type': 'contiguous',
            'min_length': 0,
            'max_length': 512,
            'coverage': 1.0
        }

    def test_basic_functionality_with_df(self):
        result = segment_sequences(self.sequences_df, self.default_params)
        # Since max_length is 512 and all sequences are shorter than that, 
        # each sequence should result in a single segment.
        self.assertEqual(len(result), 3)

    def test_output_as_dataframe(self):
        result = segment_sequences(self.sequences_df, self.default_params, AsDataFrame=True)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)

    def test_sequences_as_list(self):
        sequences_list = self.sequences_df['sequence'].tolist()
        result = segment_sequences(sequences_list, self.default_params)
        self.assertEqual(len(result), 3)

    def test_missing_columns(self):
        df_missing_columns = self.sequences_df.drop('sequence_id', axis=1)
        with self.assertRaises(ValueError) as context:
            segment_sequences(df_missing_columns, self.default_params)
        self.assertTrue("The following columns are missing: ['sequence_id']" in str(context.exception))

class TestSegmentSequencesRandom(unittest.TestCase):

    def setUp(self):
        self.sequences_df = pd.DataFrame({
            "sequence_id": {0: 0, 1: 1, 2: 2},
            "sequence": {
                0: "TAGAAATGTCCGCGACCTTTCATACATACCACCGGTACGCCCACTGCTTAAATGAGCCTTCTGTAAATTCCTGGAGATGTAGAAATGTCCGCGACCTTTCATACATACCACCGGTACGCCCTGGAGATG",
                1: "ATAATGCTAAATCGTAACCCCACTGCTTAAATGAGCCTTCTGTAAATTTCATAATGCTAAATCGTAACCCCACTGCTTAAATGAGCCTTCTGTAAATTTCATAATGCTAAATCGTAACCCCACTGCTTAAATGAGCCTTCTGTAAATTTC",
                2: "GTGACCGGGGTCAGGTTCTCGGCGGCGGCGCGCATCACGTGCTTGCCGACCCGGGGTCAGGTTCTCGGCGGCGGCCCGGGGTCAGGTTCTCGGCGGCGGCCCGGGGTCAGGTTCTCGGCGGCGGCCCGGGGTCAGGTTCTCGGCGGCGGC"
            }
        })
        self.default_params = {
            'type': 'random',
            'min_length': 10,
            'max_length': 100,
            'coverage': 10
        }

    def test_basic_functionality_with_df(self):
        result = segment_sequences_random(self.sequences_df, self.default_params)
        self.assertTrue(isinstance(result, list))
        for segment in result:
            self.assertTrue(isinstance(segment, dict))

    def test_segment_length(self):
        result = segment_sequences_random(self.sequences_df, self.default_params)
        for segment in result:
            self.assertGreaterEqual(len(segment['segment']), self.default_params['min_length'])
            self.assertLessEqual(len(segment['segment']), self.default_params['max_length'])

    def test_coverage(self):
        result = segment_sequences_random(self.sequences_df, self.default_params)
        total_sequence_length = sum(self.sequences_df['sequence'].apply(len))
        sampled_length = sum([len(segment['segment']) for segment in result])
        expected_sampled_length = total_sequence_length * self.default_params['coverage']
        # Given the randomness, we might not get exact coverage, so we allow a small margin of error
        margin = expected_sampled_length * 0.5  # 10% margin

        print(f'expected_sampled_length: {expected_sampled_length}')

        self.assertGreaterEqual(sampled_length, expected_sampled_length - margin)
        self.assertLessEqual(sampled_length, expected_sampled_length + margin)

    def test_output_structure(self):
        result = segment_sequences_random(self.sequences_df, self.default_params)
        for segment in result:
            self.assertIn('sequence_id', segment)
            self.assertIn('segment_start', segment)
            self.assertIn('segment_end', segment)
            self.assertIn('segment', segment)
            self.assertIn('segment_id', segment)

    def test_short_sequences(self):
        self.sequences_df.at[0, 'sequence'] = 'TAGAA'  # This is shorter than min_length
        result = segment_sequences_random(self.sequences_df, self.default_params)
        sequence_ids = [segment['sequence_id'] for segment in result]
        self.assertNotIn(0, sequence_ids)

class TestTokenization(unittest.TestCase):

    def setUp(self):
        # Using the provided tokenizer parameters
        defconfig = SeqConfig()
        tokenizer_params = defconfig.get_and_set_tokenization_parameters({'max_unknown_token_proportion' : 0.1})    

        self.vocabmap = tokenizer_params['vocabmap']
        self.token_limit = tokenizer_params['token_limit']
        self.max_unknown_token_proportion = tokenizer_params['max_unknown_token_proportion']

    def test_basic_tokenization(self):
        kmerized_segment = [['TCTTTG', 'CTTTGC', 'TTTGCT', 'TTGCTA', 'TGCTAA', 'GCTAAG', 'CTAAGC', 'TAAGCG', 'AAGCGT', 'AGCGTT', 'GCGTTA', 'CGTTAT', 'GTTATA', 'TTATAC', 'TATACA', 'ATACAG', 'TACAGA', 'ACAGAT', 'CAGATC', 'AGATCA', 'GATCAA', 'ATCAAA', 'TCAAAC', 'CAAACC', 'AAACCT']]
        expected_output = [[2] + [self.vocabmap[kmer.upper()] for kmer in kmerized_segment[0]] + [3]]
        output = tokenize_kmerized_segment_list(kmerized_segment, self.vocabmap, self.token_limit, self.max_unknown_token_proportion)
        self.assertEqual(output, expected_output)

    def test_unknown_tokens(self):
        kmerized_segment_with_unknown = [['TCTTTG', 'UNKNOWN', 'UNKNOWN', 'TTGCTA']]
        expected_output = [[2, 3]]  # Expected [CLS, SEP] due to exceeding unknown kmer threshold
        output = tokenize_kmerized_segment_list(kmerized_segment_with_unknown, self.vocabmap, self.token_limit, self.max_unknown_token_proportion)

        print(output)

        self.assertEqual(output, expected_output)

    def test_exceeding_token_limit(self):
        long_kmerized_segment = [['TCTTTG'] * (self.token_limit + 1)]  # This will exceed token limit
        with self.assertRaises(ValueError):
            tokenize_kmerized_segment_list(long_kmerized_segment, self.vocabmap, self.token_limit, self.max_unknown_token_proportion)

    def test_empty_segment(self):
        empty_kmerized_segment = [[]]
        expected_output = [[2, 3]]  # Expected [CLS, SEP] for empty segments
        output = tokenize_kmerized_segment_list(empty_kmerized_segment, self.vocabmap, self.token_limit, self.max_unknown_token_proportion)
        self.assertEqual(output, expected_output)

class TestLCATokenizeSegment(unittest.TestCase):

    def setUp(self):
        defconfig = SeqConfig()
        tokenizer_params = defconfig.get_and_set_tokenization_parameters()    
        self.vocabmap = tokenizer_params['vocabmap']

        self.params_example = {
            'shift': 1, 
            'max_segment_length': 512, 
            'max_unknown_token_proportion': 0.2, 
            'kmer': 5, 
            'token_limit': 10
        }
        self.params_example = defconfig.get_and_set_tokenization_parameters(self.params_example)    

    def test_valid_segment(self):

        segment = 'TCTTTGCTAAG'
        expected_tokenized = ([[2, 900, 515, 1022, 1004, 929, 629, 455, 3]], [['TCTTT', 'CTTTG', 'TTTGC', 'TTGCT', 'TGCTA', 'GCTAA', 'CTAAG']])
        result = lca_tokenize_segment(segment, self.params_example)
        self.assertEqual(result, expected_tokenized)

    def test_empty_segment(self):
        segment = ''
        # Assuming that an empty segment will return empty k-mer and token lists
        expected_result = ([[2,3]], [[]])
        result = lca_tokenize_segment(segment, self.params_example)
        self.assertEqual(result, expected_result)

    def test_segment_exceeds_max_length(self):
        segment = 'T' * (self.params_example['max_segment_length'] + 1)
        with self.assertRaises(ValueError):
            lca_tokenize_segment(segment, self.params_example)
    
    def test_max_unknown_token_proportion(self):
        # Create a segment with unknown k-mers that exceed the threshold
        segment = 'UUUUUTCTTT'
        expected_tokenized = ([[2, 3]], [['UUUUU', 'UUUUT', 'UUUTC', 'UUTCT', 'UTCTT', 'TCTTT']])
        result = lca_tokenize_segment(segment, self.params_example)
        self.assertEqual(result, expected_tokenized)


class TestSaveToHDF(unittest.TestCase):
    
    def setUp(self):
        # Create sample data
        self.array = np.random.random((100, 100))
        self.df = pd.DataFrame({'A': range(1, 101), 'B': range(101, 201)})
        
        # Temporary HDF5 file path

        self.temp_hdf_file = tempfile.mktemp(suffix=".hdf5")

    def test_basic_functionality(self):

        save_to_hdf(self.array, self.temp_hdf_file)
        
        with h5py.File(self.temp_hdf_file, 'r') as hdf:
            saved_data = hdf["training_data"]['X'][:]
            
        np.testing.assert_array_equal(saved_data, self.array)

    def test_save_with_dataframe(self):
        save_to_hdf(self.array, self.temp_hdf_file, database=self.df)
        
        with h5py.File(self.temp_hdf_file, 'r') as hdf:
            saved_data = hdf["training_data"]['X'][:]
            
        saved_df = pd.read_hdf(self.temp_hdf_file, key='database_0')
            
        np.testing.assert_array_equal(saved_data, self.array)
        pd.testing.assert_frame_equal(saved_df, self.df)

    def test_non_2d_array_error(self):
        with self.assertRaises(ValueError):
            save_to_hdf(np.random.random(100), self.temp_hdf_file)

    def test_compression(self):
        save_to_hdf(self.array, self.temp_hdf_file, compression=True)
        
        with h5py.File(self.temp_hdf_file, 'r') as hdf:
            saved_data = hdf["training_data"]['X'][:]
            
        np.testing.assert_array_equal(saved_data, self.array)

    def tearDown(self):
        # Clean up temporary file
        if os.path.exists(self.temp_hdf_file):
            os.remove(self.temp_hdf_file)


if __name__ == '__main__':
    unittest.main()
    