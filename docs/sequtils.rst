sequtils module
===============

This module contains utility functions for sequence processing and tokenization.

Functions
---------

.. automodule:: sequtils
   :members:
   :undoc-members:
   :show-inheritance:

Tokenizing a DNA Sequence Segment
---------------------------------

.. code-block:: python

   from sequtils import lca_tokenize_segment
   from config_utils import SeqConfig

   # Define a DNA sequence segment
   segment = "TCTTTGCTAAGCGTTATACAGATCAAACCT"

   # Retrieve tokenization parameters
   config = SeqConfig()
   tokenization_params = config.get_and_set_tokenization_params()

   # Tokenize the segment
   tokenized_segment, kmers_offset = lca_tokenize_segment(segment, tokenization_params)
   print(f"Tokenized Segment: {tokenized_segment}")
   print(f"K-mers Offset: {kmers_offset}")

... [Add other examples similarly] ...


