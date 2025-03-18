#!/bin/bash

# Initialize conda
__conda_setup="$('/home/antonio/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/antonio/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/antonio/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/antonio/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup

# Create a new Conda environment with Python 3.12
conda create -y -n gaussiansplatting python=3.12

# Activate the environment
conda activate gaussiansplatting

# Install PyTorch with CUDA 12.1
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install CUDA toolkit (matching PyTorch version)
conda install -y nvidia/label/cuda-12.1.1::cuda-toolkit 
conda install -y nvidia/label/cuda-12.1.1::cuda

# Install submodules
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
pip install submodules/fused-ssim

# Install other dependencies
pip install plyfile tqdm opencv-python joblib tensorboard ipykernel matplotlib

echo "Setup completed successfully!"

# to run write in the terminal ". setup_env.sh" This will execute this .sh directly in the terminal and wont open a subshell