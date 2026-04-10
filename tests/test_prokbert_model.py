import unittest

import torch
from transformers.testing_utils import (
    require_torch,
    slow,
    torch_device,
)

from src.prokbert.models.modernbert import (
    ProkBertConfig,
    ProkBertModel,
    ProkBertForMaskedLM,
    ProkBertForSequenceClassification
)

def get_config():
    """
    Returns a tiny configuration by default.
    """
    config = ProkBertConfig(
        # derived from tokenizer = LCATokenizer(kmer=1, shift=1)
        vocab_size = 9,
        pad_token_id = 0,
        eos_token_id = 3,
        bos_token_id = 2,
        cls_token_id = 2,
        sep_token_id = 3,

        hidden_size = 24,
        num_hidden_layers = 2,
        num_attention_heads = 2,
        intermediate_size = 48,
        hidden_activation = "gelu",
        mlp_dropout = 0.1,
        attention_dropout = 0.1,
        embedding_dropout = 0.1,
        classifier_dropout = 0.1,
        max_position_embeddings = 32,
        type_vocab_size = 2,
        is_decoder = False,
        initializer_range = 0.02,
    )
    config._attn_implementation = "eager"
    return config


@require_torch
class ProkBertModelTest(unittest.TestCase):

    def setUp(self):
        self.config = get_config()
        self.batch_size, self.seq_len = 2, 12
        self.input_ids = torch.randint(0, self.config.vocab_size, (self.batch_size, self.seq_len)).to(torch_device)
        self.attention_mask = torch.ones_like(self.input_ids).to(torch_device)
        self.labels = None

    def test_create_and_check_model(self):
        model = ProkBertModel(config=self.config)
        model.to(torch_device)
        model.eval()
        result = model(self.input_ids, attention_mask=self.attention_mask)
        result = model(self.input_ids)
        self.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_len, self.config.hidden_size))

    def test_for_masked_lm(self):
        model = ProkBertForMaskedLM(config=self.config)
        model.to(torch_device)
        model.eval()
        result = model(self.input_ids, attention_mask=None)
        result = model(self.input_ids, attention_mask=self.attention_mask)
        self.assertEqual(result.logits.shape, (self.batch_size, self.seq_len, self.config.vocab_size))
        self.assertIsNone(result.loss)

    def test_for_masked_lm_with_labels(self):
        model = ProkBertForMaskedLM(config=self.config)
        model.to(torch_device)
        model.eval()

        input_ids = torch.tensor( # 2 x 12
            [   # example masked sequence 1 (mask = 4)
                [2, 5, 7, 6, 8, 4, 7, 5, 7, 4, 7, 3],
                [2, 6, 4, 7, 8, 7, 6, 6, 4, 8, 7, 3],
            ],
            dtype=torch.long
        )
        attention_mask = torch.ones_like(input_ids).to(dtype=torch.bool)
        labels = torch.tensor(
            [
                [2, 5, 7, 6, 8, 5, 7, 5, 7, 6, 7, 3],
                [2, 6, 5, 7, 8, 7, 6, 6, 5, 8, 7, 3],
            ],
            dtype=torch.long
        )

        result = model(input_ids, attention_mask=None)
        result = model(input_ids, attention_mask=attention_mask)
        result = model(input_ids, attention_mask=None, labels=labels)
        result = model(input_ids, attention_mask=attention_mask, labels=labels)

        self.assertEqual(result.logits.shape, (self.batch_size, self.seq_len, self.config.vocab_size))
        self.assertIsNotNone(result.loss)

    def test_for_masked_lm_with_labels_and_sparse_prediction(self):
        model = ProkBertForMaskedLM(config=self.config)
        model.to(torch_device)
        model.eval()
        model.sparse_prediction = True

        input_ids = torch.tensor( # 2 x 12
            [
                [2, 5, 7, 6, 8, 7, 7, 5, 7, 7, 7, 3], # example sequence 1
                [2, 6, 5, 7, 8, 7, 6, 6, 5, 8, 7, 3], # example sequence 2
            ],
            dtype=torch.long
        )
        attention_mask = torch.ones_like(input_ids).to(dtype=torch.bool)
        labels = torch.tensor(
            [
                [-100, 5, -100, 6, -100, 7, -100, 5, -100, 7, -100, -100],
                [-100, 6, -100, 7, -100, 7, -100, 6, -100, 8, -100, -100],
            ], # -100 is model.sparse_pred_ignore_index
            dtype=torch.long
        )

        result = model(input_ids, attention_mask=None)
        result = model(input_ids, attention_mask=attention_mask)
        result = model(input_ids, attention_mask=None, labels=labels)
        result = model(input_ids, attention_mask=attention_mask, labels=labels)

        non_sparse_index_count = (labels != model.sparse_pred_ignore_index).sum().item()

        self.assertEqual(result.logits.shape, (non_sparse_index_count, self.config.vocab_size))
        self.assertIsNotNone(result.loss)

    def test_for_masked_lm_with_labels_dist(self):

        model = ProkBertForMaskedLM(config=self.config)
        model.to(torch_device)
        model.eval()

        input_ids = torch.tensor( # 1 x 6
            [
                [2, 5, 4, 6, 4, 3],
            ],
            dtype=torch.long
        )
        labels_dist = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # 2
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # 5
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.3, 0.4, 0.1], # 4
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # 6
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.2, 0.5], # 4
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # 3
                ]
            ],
            dtype=torch.float
        )
        result = model(input_ids, attention_mask=None, labels_dist=labels_dist)

        self.assertEqual(result.logits.shape, (input_ids.shape[0], input_ids.shape[1], self.config.vocab_size))
        self.assertIsNotNone(result.loss)

    def test_for_sequence_classification(self):
        self.config.num_labels = 3
        model = ProkBertForSequenceClassification(config=self.config)
        model.to(torch_device)
        model.eval()
        result = model(self.input_ids, attention_mask=None)
        self.assertEqual(result.logits.shape, (self.batch_size, self.config.num_labels))
        self.assertIsNone(result.loss)

    def test_for_sequence_classification_with_labels(self):
        self.config.num_labels = 3
        model = ProkBertForSequenceClassification(config=self.config)
        model.to(torch_device)
        model.eval()
        labels = torch.tensor([0, 2], dtype=torch.long).to(torch_device)
        result = model(self.input_ids, attention_mask=None, labels=labels)
        self.assertEqual(result.logits.shape, (self.batch_size, self.config.num_labels))
        self.assertIsNotNone(result.loss)

    @slow
    def test_model_from_pretrained(self):
        model_name = "neuralbioinfo/mini2-c"
        model = ProkBertForMaskedLM.from_pretrained(model_name)
        self.assertIsNotNone(model)