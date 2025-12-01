# Use official TensorFlow GPU image as base
FROM tensorflow/tensorflow:2.2.0-gpu

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    miniconda \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/theguyinthe-box/Checkers-MCTS.git

WORKDIR /Checkers-MCTS

# Install Python dependencies
RUN conda create -n checkers --file requirements.txt

# Set environment variables for TensorFlow
ENV TF_CPP_MIN_LOG_LEVEL=2

# Default command
ENTRYPOINT ['entrypoint.sh']
