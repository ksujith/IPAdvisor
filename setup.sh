#!/bin/bash

# Install system dependencies
brew install cmake
brew install faiss
brew install sentencepiece

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch first
pip install torch==2.7.0 torchvision==0.22.0

# Install other dependencies
pip install -r requirements.txt

# Verify sentence-transformers installation
python3 -c "import sentence_transformers; print('sentence-transformers version:', sentence_transformers.__version__)" 