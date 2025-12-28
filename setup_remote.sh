#!/bin/bash

# Update and Install system dependencies
apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 git zip unzip htop screen

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
mkdir -p data/train data/test data/outputs

echo "Setup complete. Ready to run training."
echo "Usage: python train_net.py"
