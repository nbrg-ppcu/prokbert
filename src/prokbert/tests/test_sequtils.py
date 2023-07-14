import unittest
import os
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import gzip
import sys

from os.path import join
prokbert_base_path = '/home/ligeti/gitrepos/prokbert'
sys.path.insert(0,join(prokbert_base_path))
from prokbert.sequtils import *

#from ..sequtils import *

class TestSeqUtils(unittest.TestCase):
    def setUp(self):
        # Create a test fasta file
        self.test_fasta_filename = 'test.fasta'
        self.test_fasta_gz_filename = 'test.fasta.gz'
        self.sequences = [SeqRecord(Seq("ATGC"), id="test1", description="test sequence 1"),
                          SeqRecord(Seq("CGTA"), id="test2", description="test sequence 2")]
        SeqIO.write(self.sequences, self.test_fasta_filename, "fasta")
        with gzip.open(self.test_fasta_gz_filename, 'wt') as f_out:
            SeqIO.write(self.sequences, f_out, "fasta")
        
        # Define test parameters
        self.params1 = {
            'segmentation': {
                'segmentation_type': 'contigous',
                'shifts': 2,
                'kmer': 5,
                'minLs': 10
            }
        }
        
        self.params2 = {
            'segmentation': {
                'segmentation_type': 'covering',
                'shifts': 2,
                'kmer': 5,
                'minLs': 10
            }
        }

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

    
    def test_segmentate_single_sequence(self):
        # Test with a single sequence without header, contigous
        sequence = "ATGCGATCGTAGCTAGCTAGC"
        segments = segmentate_single_sequence(sequence, self.params1, AddedHeader=False)
        expected_segments = ["ATGCG", "ATCGT", "AGCTA", "GCTAG"]
        self.assertEqual(segments, expected_segments)
        
        # Test with a single sequence without header, covering
        sequence = "ATGCGATCGTAGCTAGCTAGC"
        segments = segmentate_single_sequence(sequence, self.params2, AddedHeader=False)
        expected_segments = ['ATGCG', 'GCGAT', 'GATCG', 'TCGTA', 'GTAGC', 'AGCTA', 'CTAGC', 'AGCTA', 'CTAGC']
        self.assertEqual(segments, expected_segments)

        # Test with a single sequence with header
        sequence_with_header = ['test', 'test sequence', 'test.fasta', 'ATGCGATCGTAGCTAGCTAGC', 'forward']
        segments_with_header = segmentate_single_sequence(sequence_with_header, self.params1, AddedHeader=True)
        expected_segments_with_header = ["ATGCG", "ATCGT", "AGCTA", "GCTAG"]
        self.assertEqual(segments_with_header, expected_segments_with_header)

        # Test with a single sequence that doesn't meet the length constraint
        short_sequence = "ATGC"
        segments_short_sequence = segmentate_single_sequence(short_sequence, self.params1)
        self.assertEqual(segments_short_sequence, [])


if __name__ == '__main__':
    unittest.main()
    
