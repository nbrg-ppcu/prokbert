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

from prokbert.sequtils import *
from prokbert.config_utils import *
from prokbert.prokbert_tokenizer import ProkBERTTokenizer

class TestProkBERTTokenizer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = ProkBERTTokenizer()
        cls.kmer_tokenizer_k6_s1 = ProkBERTTokenizer(tokenization_params={'kmer': 6, 'shift': 1}, operation_space='kmer')
        cls.tokenizer_k6_s1 = ProkBERTTokenizer(tokenization_params={'kmer': 6, 'shift': 1}, operation_space='sequence')
        cls.tokenizer_k6_s2 = ProkBERTTokenizer(tokenization_params={'kmer': 6, 'shift': 2}, operation_space='sequence')

    def test_basic_encoding(self):
        segment = 'AATCAAGGAATTATTATCGTT'
        encoded_k6_s1 = self.tokenizer_k6_s1.encode(segment)
        encoded_k6_s1_all = self.tokenizer_k6_s1.encode(segment, all=True)
        encoded_k6_s1_nospecial = self.tokenizer_k6_s1.encode(segment, all=True, add_special_tokens=False)

        encoded_k6_s2_nospecial = self.tokenizer_k6_s2.encode(segment, all=True, add_special_tokens=False)
        encoded_k6_s2 = self.tokenizer_k6_s2.encode(segment)

        # Assert that the encoded output is correct (replace with expected output).
        self.assertEqual(encoded_k6_s1, [2, 213, 839, 3343, 1069, 165, 648, 2580, 2113, 248, 980, 3905, 3320, 978, 3899, 3296, 884, 3])
        self.assertEqual(encoded_k6_s1_all, [[2, 213, 839, 3343, 1069, 165, 648, 2580, 2113, 248, 980, 3905, 3320, 978, 3899, 3296, 884, 3]])
        self.assertEqual(encoded_k6_s1_nospecial, [[213, 839, 3343, 1069, 165, 648, 2580, 2113, 248, 980, 3905, 3320, 978, 3899, 3296, 884]])

        self.assertEqual(encoded_k6_s2_nospecial, [[213, 3343, 165, 2580, 248, 3905, 978, 3296], [839, 1069, 648, 2113, 980, 3320, 3899, 884]])
        self.assertEqual(encoded_k6_s2, [2, 213, 3343, 165, 2580, 248, 3905, 978, 3296, 3])


    def test_special_tokens(self):
        segment = 'AATCAAGGAATTATTATCGTT'
        encoded_with_special_tokens = self.tokenizer.encode(segment)
        encoded_without_special_tokens = self.tokenizer.encode(segment, add_special_tokens=False)
        # Assert the presence/absence of special tokens.
        self.assertTrue(encoded_with_special_tokens[0] == self.tokenizer.cls_token_id)
        self.assertTrue(encoded_without_special_tokens[0] != self.tokenizer.cls_token_id)

    def test_lca_shift(self):
        segment = 'AATCAAGGAATTATTATCGTT'
        encoded_shift_0 = self.tokenizer_k6_s2.encode(segment, lca_shift=0)
        encoded_shift_1 = self.tokenizer_k6_s2.encode(segment, lca_shift=1)
        # Assert that encoded outputs are different.
        self.assertNotEqual(encoded_shift_0, encoded_shift_1)
        with self.assertRaises(ValueError):
            self.tokenizer.encode(segment, lca_shift=1000)  # Assuming 1000 is an invalid lca_shift.

    def test_unknown_token_k6_s1(self):
        segment = 'AASTTAAGGAATTATTATCGT'
        tokens, _ = self.tokenizer_k6_s1.tokenize(segment, all=True)
        # Assuming 'N' is the unknown token representation
        self.assertIn(self.tokenizer_k6_s1.vocab['NNNNNN'], tokens[0])

    def test_id_conversion_k6_s1(self):
        #print('______________')
        tokens = ['AATCAA', 'TCAAGG', 'AAGGAA']
        ids = self.kmer_tokenizer_k6_s1.convert_tokens_to_ids(tokens)
        back_to_tokens = self.kmer_tokenizer_k6_s1.convert_ids_to_tokens(ids)
        #print('tokens', tokens)
        #print('ids:', ids)
        #print('back_to_tokens:', back_to_tokens)

        self.assertEqual(tokens, back_to_tokens)

    def test_restore_original_segment(self):
        segment = 'AATCAAGGAATTATTATCGTT'        
        # Tokenize the segment
        tokens = self.tokenizer_k6_s1.encode(segment)
        #print('segment: ', segment)
        #print('tokens: ', tokens)
        # Convert the tokens back to sequence
        restored_sequence = ''.join(self.tokenizer_k6_s1.convert_ids_to_tokens(tokens))        
        # Assert that the restored sequence is equal to original segment
        #print('restored_sequence: ', restored_sequence)
        self.assertEqual(restored_sequence, segment)

    def test_restore_original_segment(self):
        segment = 'AATCAAGGAATTATTATCGTT'        
        # Tokenize the segment
        tokenss = self.tokenizer_k6_s2.encode(segment, all=True)
        #print('segment: ', segment)
        #print('tokens: ', tokenss)
        # Convert the tokens back to sequence
        restored_sequence_0 = ''.join(self.tokenizer_k6_s2.convert_ids_to_tokens(tokenss[0]))
        restored_sequence_1 = ''.join(self.tokenizer_k6_s2.convert_ids_to_tokens(tokenss[1]))
        Nrestored_0 = len(restored_sequence_0)
        Nrestored_1 = len(restored_sequence_1)
        #print('restored_sequence shift 0: ', restored_sequence_0)
        #print('restored_sequence shift 1: ', restored_sequence_1)
        if Nrestored_0 < len(segment):
            control_seq = segment[0:len(segment)-1]
            self.assertEqual(restored_sequence_0, segment[0:len(segment)-1])
        else:
            self.assertEqual(Nrestored_0, segment)
        if Nrestored_1 < len(segment):
            control_seq = segment[1:len(segment)]
            self.assertEqual(restored_sequence_1, control_seq)            


        # Assert that the restored sequence is equal to original segment
 
    



if __name__ == "__main__":
    unittest.main()
