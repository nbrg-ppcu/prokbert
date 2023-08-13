import unittest
import os
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import gzip
import sys
import collections
from os.path import join

# The test should be run from the project root directory as: 

from sequtils import *
from config_utils import *
from prokbert_tokenizer import ProkBERTTokenizer

class TestProkBERTTokenizer(unittest.TestCase):

    def setUp(self):
        # This method will be called before every test. We can use it to set up our tokenizer instances
        self.tokenizer_k6_s1 = ProkBERTTokenizer(tokenization_params={'kmer': 6, 'shift': 1}, operation_space='sequence')
        self.tokenizer_k6_s2 = ProkBERTTokenizer(tokenization_params={'kmer': 6, 'shift': 2}, operation_space='sequence')

    def test_basic_tokenization_k6_s1(self):
        segment = 'AATCAAGGAATTATTATCGTT'
        tokens, _ = self.tokenizer_k6_s1.tokenize(segment, all=True)
        self.assertEqual(tokens[0], [2, 213, 3343, 165, 2580, 248, 3905, 978, 3296, 3])
        
    def test_id_conversion_k6_s1(self):
        tokens = ['AATCAA', 'TCAAGG', 'AAGGAA']
        ids = self.tokenizer_k6_s1.convert_tokens_to_ids(tokens)
        back_to_tokens = self.tokenizer_k6_s1.convert_ids_to_tokens(ids)
        self.assertEqual(tokens, back_to_tokens)
        
    def test_unknown_token_k6_s1(self):
        segment = 'AASTTAAGGAATTATTATCGT'
        tokens, _ = self.tokenizer_k6_s1.tokenize(segment, all=True)
        # Assuming 'N' is the unknown token representation
        self.assertIn('NNNNNN', tokens[0])
        
    def test_batch_decode_k6_s1(self):
        ids_list = [[2, 213, 3343, 165, 2580, 248, 3905, 978, 3296, 3]]
        sequences = self.tokenizer_k6_s1.batch_decode(ids_list)
        # Since it's a batch decode, it returns a list of sequences
        self.assertEqual(sequences[0], 'AATCAAGGAATTATTATCGTT')
        
    def test_basic_tokenization_k6_s2(self):
        segment = 'AATCAAGGAATTATTATCGTT'
        tokens, _ = self.tokenizer_k6_s2.tokenize(segment, all=True)
        # We're not asserting the exact tokens here since the values will change with shift=2
        # Just checking the length for simplicity
        self.assertEqual(len(tokens[0]), 10)

# TODO fixing unit test

if __name__ == "__main__":
    unittest.main()
