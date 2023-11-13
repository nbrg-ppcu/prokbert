=====================
ProkBERT Tokenizer
=====================

This module contains the tools required by the ProkBERT tokenizer. The `ProkBERTTokenizer` class is designed to handle specific tokenization processes required for ProkBERT, including LCA tokenization and sequence segmentation.

ProkBERTTokenizer Class and Methods
-----------------------------------

.. automodule:: prokbert.prokbert_tokenizer
    :members:
    :undoc-members:
    :show-inheritance:

The `ProkBERTTokenizer` class inherits from the standard tokenizer classes and includes additional methods specific to ProkBERT's requirements.

Additionally, below are more detailed listings and descriptions of the individual methods within the `ProkBERTTokenizer` class:

.. autosummary::
    :toctree: api
    :maxdepth: 1
    :titlesonly:

    prokbert.prokbert_tokenizer.load_vocab
    prokbert.prokbert_tokenizer.ProkBERTTokenizer
    prokbert.prokbert_tokenizer.ProkBERTTokenizer.tokenize
    prokbert.prokbert_tokenizer.ProkBERTTokenizer.convert_ids_to_tokens
    prokbert.prokbert_tokenizer.ProkBERTTokenizer.save_vocabulary
    prokbert.prokbert_tokenizer.ProkBERTTokenizer.from_pretrained
    prokbert.prokbert_tokenizer.ProkBERTTokenizer.encode_plus
    prokbert.prokbert_tokenizer.ProkBERTTokenizer.batch_encode_plus
    prokbert.prokbert_tokenizer.ProkBERTTokenizer.encode
    prokbert.prokbert_tokenizer.ProkBERTTokenizer.decode
    prokbert.prokbert_tokenizer.ProkBERTTokenizer.batch_decode
    