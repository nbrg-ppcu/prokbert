import unittest
import torch
from transformers import BertTokenizer

class TestProkBERTDataCollator(unittest.TestCase):

    def setUp(self):
        # Setup code: this runs before every individual test
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.collator = ProkBERTDataCollator(tokenizer=self.tokenizer, mlm_probability=0.15)

    def test_masking(self):
        # A simple test to check if the input tensor is masked properly
        inputs = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        masked_inputs, labels = self.collator.torch_mask_tokens(inputs)

        # Check if shape remains the same
        self.assertEqual(masked_inputs.shape, inputs.shape)
        self.assertEqual(labels.shape, inputs.shape)

        # Check if 80% tokens are masked
        total_tokens = torch.numel(inputs)
        masked_tokens = total_tokens - torch.sum(labels == -100)
        self.assertGreaterEqual(masked_tokens, int(0.8 * total_tokens))

    def test_mask_neighbourhood_params(self):
        # Test if setting mask neighborhood parameters works
        self.collator.set_mask_neighborhood_params(1, 1)
        self.assertEqual(self.collator.mask_to_left, 1)
        self.assertEqual(self.collator.mask_to_right, 1)

        # Test for invalid inputs
        with self.assertRaises(ValueError):
            self.collator.set_mask_neighborhood_params(-1, 1)
        with self.assertRaises(ValueError):
            self.collator.set_mask_neighborhood_params(1, -1)

    # ... Add more tests ...


if __name__ == '__main__':
    unittest.main()