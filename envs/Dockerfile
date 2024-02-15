FROM nvidia/cuda:12.1.1-base-ubuntu22.04
LABEL maintainer="Balázs Ligeti <obbalasz@gmail.com>"

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    unzip \
    lbzip2 \
    mc \
    libgtextutils-dev \
    sudo \
    curl \
    gpg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create and switch to the 'prokbert' user before installing Miniconda
RUN useradd -m -s /bin/bash prokbert
USER prokbert
WORKDIR /home/prokbert

# Install Miniconda as 'prokbert'
RUN mkdir -p /home/prokbert/bin && \
    cd /home/prokbert/bin && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-py311_23.10.0-1-Linux-x86_64.sh && \
    chmod +x Miniconda3-py311_23.10.0-1-Linux-x86_64.sh && \
    ./Miniconda3-py311_23.10.0-1-Linux-x86_64.sh -b -p /home/prokbert/miniconda3 && \
    /home/prokbert/miniconda3/bin/conda init bash && \
    echo "source /home/prokbert/miniconda3/etc/profile.d/conda.sh" >> /home/prokbert/.bashrc

# Set PATH to include the user's conda and potentially other user-specific binaries
ENV PATH /home/prokbert/miniconda3/bin:$PATH


# Use pip to install prokbert and other dependencies
RUN pip install prokbert seaborn umap-learn jupyterlab

# Set up a work directory for prokbert stuff
RUN git clone https://github.com/nbrg-ppcu/prokbert.git
USER prokbert

CMD ["/bin/bash"]
