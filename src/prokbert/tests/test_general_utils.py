# test_general_utils.py

import unittest
import pandas as pd
from prokbert.general_utils import *


class TestCheckExpectedColumns(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    
    def test_columns_present(self):
        self.assertTrue(check_expected_columns(self.df, ['A', 'B']))
    
    def test_column_missing(self):
        with self.assertRaises(ValueError) as context:
            check_expected_columns(self.df, ['A', 'C'])
        
        self.assertTrue("The following columns are missing: ['C']" in str(context.exception))

    def test_multiple_columns_missing(self):
        with self.assertRaises(ValueError) as context:
            check_expected_columns(self.df, ['A', 'C', 'D'])
        
        self.assertTrue("The following columns are missing: ['C', 'D']" in str(context.exception))
    
    def test_empty_expected_columns(self):
        self.assertTrue(check_expected_columns(self.df, []))


class TestIsValidPrimaryKey(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            'A': [1, 2, 3, 4],
            'B': [5, 6, 6, 7],
            'C': [8, 9, 10, None]
        })
    
    def test_valid_primary_key(self):
        self.assertTrue(is_valid_primary_key(self.df, 'A'))
    
    def test_non_unique_values(self):
        self.assertFalse(is_valid_primary_key(self.df, 'B'))
    
    def test_missing_values(self):
        self.assertFalse(is_valid_primary_key(self.df, 'C'))
    
    def test_non_existent_column(self):
        with self.assertRaises(ValueError) as context:
            is_valid_primary_key(self.df, 'D')
        
        self.assertTrue("Column 'D' does not exist in the DataFrame." in str(context.exception))


if __name__ == "__main__":
    unittest.main()