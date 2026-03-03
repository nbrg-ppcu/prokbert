# The test should be run from the project root directory as: 


import unittest

from prokbert.config_utils import *


class TestSeqConfig(unittest.TestCase):

    def setUp(self):
        self.config = SeqConfig()

    def test_initialization(self):
        self.assertTrue(isinstance(self.config.parameters, dict))
        self.assertIn('segmentation', self.config.parameters)
        self.assertIn('tokenization', self.config.parameters)

    def test_get_parameter(self):
        segmentation_type = self.config.get_parameter('segmentation', 'type')
        self.assertEqual(segmentation_type, 'contiguous')

    def test_validate_type(self):
        self.assertTrue(self.config.validate_type('segmentation', 'type', 'contiguous'))
        self.assertFalse(self.config.validate_type('segmentation', 'type', 123))
        self.assertTrue(self.config.validate_type('segmentation', 'min_length', 5))
        self.assertFalse(self.config.validate_type('segmentation', 'min_length', 'five'))

    def test_validate_value(self):
        self.assertTrue(self.config.validate_value('segmentation', 'type', 'contiguous'))
        self.assertFalse(self.config.validate_value('segmentation', 'type', 'invalid_type'))
        self.assertTrue(self.config.validate_value('segmentation', 'min_length', 5))
        self.assertFalse(self.config.validate_value('segmentation', 'min_length', -5))

    def test_validate(self):
        with self.assertRaises(TypeError):
            self.config.validate('segmentation', 'type', 123)
        with self.assertRaises(ValueError):
            self.config.validate('segmentation', 'type', 'invalid_type')

    def test_get_set_segmentation_parameters(self):
        parameters = self.config.get_and_set_segmentation_parameters({'type': 'random'})
        self.assertEqual(parameters['type'], 'random')

        with self.assertRaises(ValueError):
            self.config.get_and_set_segmentation_parameters({'invalid_param': 'value'})

if __name__ == "__main__":
    unittest.main()