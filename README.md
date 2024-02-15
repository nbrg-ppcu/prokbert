# The ProkBERT model family

ProkBERT is an advanced genomic language model specifically designed for microbiome analysis. This repository contains the ProkBERT package and utilities, as well as the LCA tokenizer and model definitions.


## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
  - [Installing with pip](#installing-with-pip)
  - [Installing with conda](#installing-with-conda)
  - [Using Docker](#using-docker)
  - [Using Singularity (Apptainer)](#using-singularity-apptainer)


### Introduction
The ProkBERT model family is a transformer-based, encoder-only architecture based on [BERT](https://github.com/google-research/bert). Built on transfer learning and self-supervised methodologies, ProkBERT models capitalize on the abundant available data, demonstrating adaptability across diverse scenarios. The models’ learned representations align with established biological understanding, shedding light on phylogenetic relationships. With the novel Local Context-Aware (LCA) tokenization, the ProkBERT family overcomes the context size limitations of traditional transformer models without sacrificing performance or the information-rich local context. In bioinformatics tasks like promoter prediction and phage identification, ProkBERT models excel. For promoter predictions, the best-performing model achieved an MCC of 0.74 for E. coli and 0.62 in mixed-species contexts. In phage identification, they all consistently outperformed tools like VirSorter2 and DeepVirFinder, registering an MCC of 0.85. Compact yet powerful, the ProkBERT models are efficient, generalizable, and swift.

### Features
- Local Context-Aware (LCA) tokenization for better genomic sequence understanding.
- Pre-trained models available for immediate use and fine-tuning.
- High performance in various bioinformatics tasks.
- Facilitation of both supervised and unsupervised learning.


## Installation

### Installing with pip

The recommended way to install ProkBERT is through pip, which will handle most dependencies automatically:

```bash
pip install prokbert
```

### Installing with conda

ProkBERT is also available as a conda package from the Bioconda channel. To install it using conda, run:

```bash
conda install prokbert -c bioconda
```

### Using Docker

Before using the ProkBERT container with GPU support, make sure you have the following installed on your system:
- Python (3.10 or later)
- [Docker](https://docs.docker.com/get-docker/) (required if you plan to use the Docker image)
- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) (required if you intend to use Docker with GPU support)

To pull and run the ProkBERT Docker image, use:

```bash
docker pull obalasz/prokbert
```

To run the container with GPU support, use:

```bash
docker run --gpus all -it --rm -v $(pwd):/app obalasz/prokbert python /app/finetuning.py --help
```

This command runs the ProkBERT Docker container with GPU support, mounts the current directory to the container (allowing access to your local files), and displays the help message for the `finetuning.py` script.

### Using Singularity (Apptainer)

For users who prefer Singularity (now also known as Apptainer), a Singularity (Apptainer) container definition is provided. Please refer to the official documentation for instructions on building and running Singularity containers. A prebuilt container is available on Zenodo: [https://zenodo.org/records/10659030](https://zenodo.org/records/10659030).

Building the Singularity container:

```bash
apptainer build prokbert.sif prokbert.def
```
To pull directly from Docker Hub and convert to a Singularity image file:
```bash
singularity pull prokbert.sif docker://obalasz/prokbert
```

Once you have your `.sif` file, you can run ProkBERT with the following command:
```bash
singularity run --nv prokbert.sif python /opt/prokbert/examples/finetuning.py --help
```

The `--nv` flag enables NVIDIA GPU support, assuming your system has NVIDIA drivers and CUDA installed. Remove this flag if you're running on a CPU-only system.

You can also shell into the container or execute specific commands as follows:
Shell into the container:
```bash
singularity shell --nv prokbert.sif
```
**Note**: If you encounter any problems, please do not hesitate to contact us or open an issue. My email address is obalasz@gmail.com.

## Applications
ProkBERT has been validated in several key genomic tasks, including:
- Learning meaningful repreresentation for seqeuences (zero-shot capibility)
- Accurate bacterial promoter prediction.
- Detailed phage sequence analysis within complex microbiome datasets.


## Quick Start
Our models and datasets are avaialble on the [hugginface page](https://huggingface.co/neuralbioinfo). 
The models are easy to use with the [transformers](https://github.com/huggingface/transformers) package.
We provide examples and descriptions as notebooks in the next chapter and some example scsripts regarging how to preprocess your sequence data and how to finetune the available models. The examples are available in the [example](https://github.com/nbrg-ppcu/prokbert/tree/main/examples) folder of this repository. 

### TLDR example

To load the model from Hugging Face:
```python
import torch
from transformers import AutoTokenizer, AutoModel
from prokbert.prokbert_tokenizer import ProkBERTTokenizer

tokenizer = ProkBERTTokenizer(tokenization_params={'kmer' : 6, 'shift' : 1})
model = AutoModel.from_pretrained("nerualbioinfo/prokbert-mini", trust_remote_code=True)

segment = "TATGTAACATAATGCGACCAATAATCGTAATGAATATGAGAAGTGTGATATTATAACATTTCATGACTACTGCAAGACTAA"
inputs = tokenizer(segment)['input_ids']
tokenizer.batch_encode_plus([segment])
```


## Tutorials and examples:

### Tokenization and Segmentation (sequence preprocessing)
For examples for how to preprocess the raw seqeuence data, which are freqently stored in fasta format:
Examples:
- Segmentation (sequence preprocessing): [colab link](https://colab.research.google.com/github/nbrg-ppcu/prokbert/blob/main/examples/Segmentation.ipynb)
- Tokenization [colab link](https://colab.research.google.com/github/nbrg-ppcu/prokbert/blob/main/examples/Tokenization.ipynb)
- Preprocessing for pretraining
  

### Visualizing sequence representations (embeddings)
An example for how to visualize the genomic features of ESKAPE pathogens. More description about the dataset is available on huggingface
Example:
 - ESKAPE pathogen genomic features: [colab link](https://colab.research.google.com/github/nbrg-ppcu/prokbert/blob/main/examples/Embedding_visualization.ipynb) 

### Finetuning example for promoter sequences
Here we provide an example for a practical transfer learning task. It is formulated as binary classification. We provide a notebook for presenting the basic concepts and a command line script as template. 
Examples:
- Finetuning for promoter identification task: [colab link](https://colab.research.google.com/github/nbrg-ppcu/prokbert/blob/main/examples/Embedding_visualization.ipynb)
- Python script for the finetuning: [link](https://github.com/nbrg-ppcu/prokbert/blob/main/examples/finetuning.py)
  
Usage example:
```bash
git clone https://github.com/nbrg-ppcu/prokbert
cd examples
python finetuning.py \
  --model_name neuralbioinfo/prokbert-mini \
  --ftmodel mini_promoter \
  --model_outputpath finetuning_outputs \
  --num_train_epochs 1 \
  --per_device_train_batch_size 128 
```
For practical applications or for larger training tasks we recommend to use the [Distributed DataParallel](https://huggingface.co/docs/transformers/en/perf_train_gpu_many). 


### Pretraining Example

Here you can find an example for pretraining ProkBERT from scratch. Pretraining is an essential step, allowing the model to learn the underlying patterns before being fine-tuned for downstream tasks. All of the pretrained models are available on Hugging Face.

#### Pretrained Models:

| Model | k-mer | Shift | Hugging Face URL |
| ----- | ----- | ----- | ---------------- |
| ProkBERT-mini | 6 | 1 | [Link](https://huggingface.co/neuralbioinfo/prokbert-mini) |
| ProkBERT-mini-c | 1 | 1 | [Link](https://huggingface.co/neuralbioinfo/prokbert-mini-c) |
| ProkBERT-mini-long | 6 | 2 | [Link](https://huggingface.co/neuralbioinfo/prokbert-mini-long) |

#### Preprocessing the Example Data:

It is a good practice to preprocess larger sequence sets (>100MB) in advance. Below is an example script for achieving this by running on multiple fasta files:

Clone the ProkBERT repository and navigate to the examples directory. Then, preprocess the example data into a format suitable for training. This involves converting sequences into k-mer representations. Run the following commands:

```bash
git clone https://github.com/nbrg-ppcu/prokbert
cd prokbert/examples
python prokbert_seqpreprocess.py \
  --kmer 6 \
  --shift 1 \
  --fasta_file_dir ../src/prokbert/data/pretraining \
  --out ../src/prokbert/data/preprocessed/pretraining_k6s1.h5
```

Parameters:
- `--kmer`: The size of the k-mer (number of bases) to use for sequence encoding.
- `--shift`: The shift size for sliding the k-mer window across sequences.
- `--fasta_file_dir`: Directory containing your FASTA files for pretraining.
- `--out`: Output file path for the preprocessed data in HDF5 format.

#### Running the Pretraining from Scratch:

Use the preprocessed HDF file as input for pretraining. Execute the commands below:

```bash
python prokbert_pretrain.py \
  --kmer 6 \
  --shift 1 \
  --dataset_path ../src/prokbert/data/preprocessed/pretraining_k6s1.h5 \
  --model_name prokbert_k6s1 \
  --output_dir ./tmppretraining \
  --model_outputpath ./tmppretraining
```

Parameters:
- `--model_name`: Name for the model configuration to be used or saved.
- `--output_dir`: Directory where the training logs and temporary files will be saved.
- `--model_outputpath`: Path where the final trained model should be saved.


# About ProkBERT  

## Detailed pre-training and evaluation process

### The pre-training process

ProkBERT models were pre-trained using a modified Masked Language Modeling (MLM) approach, tailored to handle the unique characteristics of genomic sequences. This involved the innovative use of overlapping and shifted k-mers for tokenization (LCA tokenization), which allowed for a richer contextual understanding of sequences. The pretraining process was conducted on a diverse dataset from the NCBI RefSeq database, ensuring comprehensive coverage of various genomic entities. This rigorous approach has equipped ProkBERT models with the capability to accurately interpret and analyze complex microbiome data.

### ProkBERT trained for promoter prediction

In the first application of ProkBERT models, we focused on differentiating bacterial promoter from non-promoter sequences. Our database, mainly derived from the Prokaryotic Promoter Database (PPD), included experimentally validated promoter sequences from 75 organisms, complemented by an independent test set of _E. coli_ sigma70 promoters. To generate a comprehensive negative dataset, we combined non-promoter sequences (CDS), sequences generated via a 3rd-order Markov chain, and purely random sequences, ensuring a diverse and challenging training environment. For fine-tuning, we adapted the Megatron BERT architecture for binary classification, employing a novel weighting mechanism to integrate representations of all tokens, rather than relying on the conventional [CLS] token. This approach, combined with a tailored softmax operation, yielded a weighted sum of sequence representations for accurate classification. Our fine-tuning process involved the AdamW optimizer, a linear learning rate scheduler with warmup, and training on NVIDIA A100-40GB GPUs for two epochs. The comprehensive training and validation process, illustrated in Figure 1, ensured that ProkBERT models could efficiently and accurately predict bacterial promoters, setting a new standard in genomic sequence analysis.

![ProkBERT Promoters](assets/Figure4_promoter_db.png)
*Figure 1: Promoter dataset schematic*

Our models were evaluated in a binary classification setting, distinguishing between promoter and non-promoter sequences, and trained on a diverse set of data, including _E. coli_ sigma70 promoters. ProkBERT models demonstrated superior performance with an impressive accuracy and MCC (Matthews Correlation Coefficient) of 0.87 and 0.74 respectively, outperforming other established tools such as CNNProm and iPro70-FMWin (see Figure 2). This highlights ProkBERT's effectiveness in correctly identifying both promoters and non-promoters, with consistent results across various model variants. The evaluation also included a comparative analysis with newer tools like Promotech and iPromoter-BnCNN, where ProkBERT maintained its lead, especially in specificity and sensitivity. This robust performance is further emphasized in diverse testing scenarios, showcasing ProkBERT's generalizability and reliability in promoter prediction tasks across different bacterial species. The success of ProkBERT in this domain underscores its potential as a powerful tool in genomic sequence analysis, particularly in accurately predicting bacterial promoters.


![ProkBERT Promoters](assets/Figure6_prom_res.png)
*Figure 2: Comparative analysis of ProkBERT's promoter prediction performance*

### ProkBERT in phage identification

ProkBERT was secondly tested on phage sequence analysis, highlighting its critical role in understanding bacteriophages within microbiomes. Phages are instrumental in influencing host dynamics and driving horizontal gene transfer, pivotal in the spread of antibiotic resistance and virulence genes. Grasping phage diversity is essential for addressing global challenges such as climate change and various diseases. However, quantifying and characterizing phages accurately faces obstacles due to the limited viral sequences in databases and the complexities of viral taxonomy. To address these issues, we developed a unique phage sequence database with recent genomic data, employing a balanced benchmarking approach to reduce bias in viral group categorization.

In compiling this database, we assembled an extensive collection of phage sequences from various sources, including the RefSeq database. We utilized the CD-HIT algorithm to eliminate redundancy, resulting in a dataset of 40,512 unique phage sequences. This dataset was meticulously mapped to their bacterial hosts, shedding light on phage-bacteria interactions. Special emphasis was placed on the relationships between phages and key bacterial genera like Salmonella and Escherichia, due to their significant health impacts. This balanced dataset is crucial for exploring the intricate relationships between phages and bacteria, furthering our understanding of microbial ecology.

For the phage sequence analysis, we adopted a binary classification methodology akin to the approach used for promoter training.

ProkBERT demonstrated outstanding performance in phage sequence identification, excelling in both accuracy and speed. In our comprehensive evaluation against various established methods, ProkBERT models consistently outperformed others across different sequence lengths. _ProkBERT-mini_ showed exceptional accuracy with shorter sequences, maintaining high accuracy without increased false positives or negatives. Notably, for 2kb sequences, _ProkBERT-mini-long_ achieved an impressive accuracy of 92.90% and a Matthews Correlation Coefficient (MCC) of 0.859, indicating its high reliability and precision in prediction.

ProkBERT's superiority was further highlighted in its ability to maintain high sensitivity in identifying actual phage sequences and specificity in discerning non-phage sequences. Additionally, the ProkBERT family shone in terms of evaluation speed, consistently registering times under 10 seconds for all sequence lengths, making it an invaluable tool for applications that require rapid and accurate phage sequence predictions. These results underline ProkBERT's exceptional capabilities and mark it as a leading solution in the field of genomic sequence analysis, especially in the challenging area of phage sequence classification.

![ProkBERT Phages](assets/Figure7_phag_res.png)
*Figure 3: ProkBERT identifies phage sequences accurately and rapidly*

# Citing this work

If you use the code or data in this package, please cite:

```bibtex
@Article{ProkBERT2024,
  author  = {Ligeti, Balázs and Szepesi-Nagy, István and Bodnár, Babett and Ligeti-Nagy, Noémi and Juhász, János},
  journal = {Frontiers in Microbiology},
  title   = {{ProkBERT} family: genomic language models for microbiome applications},
  year    = {2024},
  volume  = {14},
  URL={https://www.frontiersin.org/articles/10.3389/fmicb.2023.1331233},       
	DOI={10.3389/fmicb.2023.1331233},      
	ISSN={1664-302X}
}
```
