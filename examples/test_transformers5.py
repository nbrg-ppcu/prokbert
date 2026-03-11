
from dataclasses import dataclass
import os
from typing import Dict, Tuple

import numpy as np
import torch
from datasets import ClassLabel, load_dataset
from sklearn.metrics import accuracy_score
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)
import pandas as pd
import os
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
from datasets import load_dataset
import seaborn as sns
import matplotlib.pyplot as plt
import umap

from prokbert.models import ProkBertModel




@dataclass
class Config:
    seed: int = 42
    dataset_name: str = "neuralbioinfo/ESKAPE-genomic-features"
    dataset_split: str = "ESKAPE"
    model_name: str = "neuralbioinfo/prokbert-mini"
    output_dir: str = "./"
    max_length: int = 256
    eval_batch_size: int = 256


def main():
    print('Testing the EMBEDDING generation')

    sequences = ['ATAGACTATATGC', 
                 'GCTGCTATGGC']
    
    





if __name__ == "__main__":
    main()

