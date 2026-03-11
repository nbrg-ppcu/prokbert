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

from prokbert.models import *
from prokbert.tokenizer import LCATokenizer

# prokbert/register_autoclasses.py
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, AutoTokenizer
from prokbert.models import ProkBertConfig, ProkBertModel, ProkBertForMaskedLM
from prokbert.tokenizer import LCATokenizer




model_name = "neuralbioinfo/prokbert-mini"
model_output_folder = '/project/c_evolm/huggingface/prokbert-mini'

def register_local_autoclasses():
    AutoConfig.register("prokbert", ProkBertConfig, exist_ok=True)
    AutoModel.register(ProkBertConfig, ProkBertModel, exist_ok=True)
    AutoModelForMaskedLM.register(ProkBertConfig, ProkBertForMaskedLM, exist_ok=True)
    AutoTokenizer.register(ProkBertConfig, tokenizer_class=LCATokenizer, exist_ok=True)

def main():
    print('Testing the EMBEDDING generation')
    register_local_autoclasses()
    model = ProkBertForMaskedLM.from_pretrained(model_name)
    model.save_pretrained(model_output_folder)

    print(model.config)

    tokenizer = LCATokenizer(kmer=model.config.kmer, shift=model.config.shift)
    tokenizer.save_pretrained(model_output_folder)
    model.config.save_pretrained(model_output_folder)


    #



    




if __name__ == "__main__":
    main()
