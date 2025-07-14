# Use CUDA 12.8 with cuDNN on Ubuntu 24.04
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

LABEL maintainer="Balázs Ligeti <obbalasz@gmail.com>"

# Avoid interactive prompts during apt operations
ENV DEBIAN_FRONTEND=noninteractive

# 1. Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget \
        lbzip2 \
        mc \
        screen \
        curl \
        git \
        sudo \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 2. Create a non-root user "prokbert" and switch into its home directory
RUN useradd -m -s /bin/bash prokbert
USER prokbert
WORKDIR /home/prokbert

# 3. Set a USERDIR environment variable for clarity
ENV USERDIR=/home/prokbert

# 4. Download and install Miniforge3 under /home/prokbert/miniforge3
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    && chmod +x Miniforge3-Linux-x86_64.sh \
    && ./Miniforge3-Linux-x86_64.sh -b -p $USERDIR/miniforge3 \
    && rm Miniforge3-Linux-x86_64.sh

# 5. Ensure conda commands are on PATH
ENV PATH=$USERDIR/miniforge3/bin:$PATH

# 6. (Optional) Initialize conda for bash (useful if you later open an interactive shell)
RUN conda init bash

# 7. Install mamba (faster conda frontend) in the base environment
RUN conda install -y mamba -n base -c conda-forge

# 8. Use mamba to install additional utilities (lbzip2, mc, screen, git-lfs)
#    Note: git-lfs will be initialized below
RUN mamba install -y \
        lbzip2 \
        mc \
        screen \
        git-lfs \
    && mamba clean -afy

# 9. Initialize Git LFS so that repositories using large files will work correctly
RUN git lfs install --skip-repo

# 10. Install PyTorch (CUDA 12.8) and other Python packages via pip
RUN pip install --no-cache-dir \
        torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 \
    && pip install --no-cache-dir protobuf \
    && pip install --no-cache-dir git+https://github.com/huggingface/datasets \
    && pip install --no-cache-dir git+https://github.com/huggingface/accelerate \
    && pip install --no-cache-dir git+https://github.com/huggingface/transformers.git \
    && pip install --no-cache-dir peft einops \
    && pip install --no-cache-dir packaging ninja \
    && pip install --no-cache-dir tensorboard \
    && pip install --no-cache-dir triton \
    && pip install --no-cache-dir scikit-learn pandas matplotlib h5py jupyterlab seaborn jupyter scipy biopython openpyxl xlsxwriter statsmodels intervaltree

# 11. Install the latest ProkBERT package directly from GitHub
RUN pip install --no-cache-dir git+https://github.com/nbrg-ppcu/prokbert.git

# 12. Default to bash
CMD ["/bin/bash"]