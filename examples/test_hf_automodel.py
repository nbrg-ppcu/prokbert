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

#from prokbert.models import *
from prokbert.tokenizer import LCATokenizer

# prokbert/register_autoclasses.py
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, AutoTokenizer, AutoModelForSequenceClassification
from prokbert.models import ProkBertConfig, ProkBertModel, ProkBertForMaskedLM
#from prokbert.tokenizer import LCATokenizer

model_name = "neuralbioinfo/prokbert-mini-long"
model_revision = "update-configs"

# revision="update-configs"


def main():
    pass

    print('Loading the model')
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, revision=model_revision)
    tokenizer = LCATokenizer.from_pretrained(model_name, trust_remote_code=True, revision=model_revision)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, revision=model_revision)
    modellm = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True, revision=model_revision)

    modelseqclass = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True, revision=model_revision)


    print(modelseqclass)


if __name__ == "__main__":
    main()
