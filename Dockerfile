FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    curl \
    git \
    nano \
    python3-venv \
    python3-wheel \
    && apt-get clean

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

# Set path to conda
ENV PATH=/opt/conda/bin:$PATH

# Create a new conda environment
RUN /opt/conda/bin/conda create -n spooffrontiers_env python=3.8

# Activate the environment and install packages
RUN /opt/conda/bin/conda run -n spooffrontiers_env \
    pip install torch torchvision torchaudio && \
    pip install notebook ipykernel && \
    python -m ipykernel install --user

# Set the working directory
WORKDIR /root

# Configure git
RUN git config --global --add safe.directory /workspace/spooffrontiers

# Set the entrypoint to use bash
ENTRYPOINT ["/bin/bash"]
