# The ProkBERT model family

ProkBERT is an advanced genomic language model specifically designed for microbiome analysis. This repository contains the ProkBERT package and utilities, as well as the LCA tokenizer and model definitions.

## Introduction
The ProkBERT model family is a transformer-based, encoder-only architecture based on [BERT](https://github.com/google-research/bert). Built on transfer learning and self-supervised methodologies, ProkBERT models capitalize on the abundant available data, demonstrating adaptability across diverse scenarios. The models’ learned representations align with established biological understanding, shedding light on phylogenetic relationships. With the novel Local Context-Aware (LCA) tokenization, the ProkBERT family overcomes the context size limitations of traditional transformer models without sacrificing performance or the information rich local context. In bioinformatics tasks like promoter prediction and phage identification, ProkBERT models excel. For promoter predictions, the best performing model achieved an MCC of 0.74 for E. coli and 0.62 in mixed-species contexts. In phage identification, they all consistently outperformed tools like VirSorter2 and DeepVirFinder, registering an MCC of 0.85. Compact yet powerful, the ProkBERT models are efficient, generalizable, and swift.

## Features
- Local Context-Aware (LCA) tokenization for better genomic sequence understanding.
- Pre-trained models available for immediate use and fine-tuning.
- High performance in various bioinformatics tasks.
- Facilitation of both supervised and unsupervised learning.

## Applications

ProkBERT has been validated in several key genomic tasks, including:
- Accurate bacterial promoter prediction.
- Detailed phage sequence analysis within complex microbiome datasets.

## Getting Started

To get started with ProkBERT, clone the repository and follow the setup instructions in the documentation.

```bash
pip install prokbert
cd prokbert
```

## Quick Start
Our model is easy to use with the [transformers](https://github.com/huggingface/transformers) package.


To load the model from Hugging Face:
```python
import torch
from transformers import AutoTokenizer, AutoModel
from prokbert.prokbert_tokenizer import ProkBERTTokenizer

tokenizer = ProkBERTTokenizer(tokenization_params={'kmer' : 6, 'shift' : 1})
model = AutoModel.from_pretrained("nerualbioinfo/prokbert-mini-k6s1", trust_remote_code=True)

segment = "TATGTAACATAATGCGACCAATAATCGTAATGAATATGAGAAGTGTGATATTATAACATTTCATGACTACTGCAAGACTAA"
inputs = tokenizer(segment)['input_ids']

tokenizer.batch_encode_plus([segment])

```


## Pre-Training

Codes for pre-training is coming soon.

## Finetune

Codes for pre-training is coming soon.


# Detailed pre-training and evaluation process

## The pre-training process

ProkBERT models were pre-trained using a modified Masked Language Modeling (MLM) approach, tailored to handle the unique characteristics of genomic sequences. This involved the innovative use of overlapping and shifted k-mers for tokenization (LCA tokenization), which allowed for a richer contextual understanding of sequences. The pretraining process was conducted on a diverse dataset from the NCBI RefSeq database, ensuring comprehensive coverage of various genomic entities. This rigorous approach has equipped ProkBERT models with the capability to accurately interpret and analyze complex microbiome data.

## ProkBERT trained for promoter prediction

In the first application of ProkBERT models, we focused on differentiating bacterial promoter from non-promoter sequences. Our database, mainly derived from the Prokaryotic Promoter Database (PPD), included experimentally validated promoter sequences from 75 organisms, complemented by an independent test set of _E. coli_ sigma70 promoters. To generate a comprehensive negative dataset, we combined non-promoter sequences (CDS), sequences generated via a 3rd-order Markov chain, and purely random sequences, ensuring a diverse and challenging training environment. For fine-tuning, we adapted the Megatron BERT architecture for binary classification, employing a novel weighting mechanism to integrate representations of all tokens, rather than relying on the conventional [CLS] token. This approach, combined with a tailored softmax operation, yielded a weighted sum of sequence representations for accurate classification. Our fine-tuning process involved the AdamW optimizer, a linear learning rate scheduler with warmup, and training on NVIDIA A100-40GB GPUs for two epochs. The comprehensive training and validation process, illustrated in Figure 1, ensured that ProkBERT models could efficiently and accurately predict bacterial promoters, setting a new standard in genomic sequence analysis.

![ProkBERT Promoters](assets/Figure4_promoter_db.png)
*Figure 1: Promoter dataset schematic*

Our models were evaluated in a binary classification setting, distinguishing between promoter and non-promoter sequences, and trained on a diverse set of data, including _E. coli_ sigma70 promoters. ProkBERT models demonstrated superior performance with an impressive accuracy and MCC (Matthews Correlation Coefficient) of 0.87 and 0.74 respectively, outperforming other established tools such as CNNProm and iPro70-FMWin (see Figure 2). This highlights ProkBERT's effectiveness in correctly identifying both promoters and non-promoters, with consistent results across various model variants. The evaluation also included a comparative analysis with newer tools like Promotech and iPromoter-BnCNN, where ProkBERT maintained its lead, especially in specificity and sensitivity. This robust performance is further emphasized in diverse testing scenarios, showcasing ProkBERT's generalizability and reliability in promoter prediction tasks across different bacterial species. The success of ProkBERT in this domain underscores its potential as a powerful tool in genomic sequence analysis, particularly in accurately predicting bacterial promoters.


![ProkBERT Promoters](assets/Figure6_prom_res.png)
*Figure 2: Comparative analysis of ProkBERT's promoter prediction performance*


# Documentation

[Read the Docs](https://prokbert.readthedocs.io/en/latest/)
