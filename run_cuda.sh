#!/bin/bash
set -e  # Exit on error

echo "Setting up CUDA environment..."

# Function to check if a directory exists
check_dir() {
    if [ ! -d "$1" ]; then
        echo "Error: Directory $1 does not exist"
        return 1
    fi
    return 0
}

# Check if CUDA is installed
if ! command -v nvcc &> /dev/null; then
    echo "Error: CUDA compiler (nvcc) not found"
    echo "Please install CUDA toolkit:"
    echo "sudo apt-get update"
    echo "sudo apt-get install nvidia-cuda-toolkit"
    exit 1
fi

# Print CUDA version
echo "CUDA Version:"
if ! nvcc --version; then
    echo "Error: Failed to get CUDA version"
    exit 1
fi

# Check NVIDIA driver
echo -e "\nNVIDIA Driver Status:"
if ! nvidia-smi; then
    echo "Error: Failed to get NVIDIA driver status"
    exit 1
fi

# Set CUDA paths
CUDA_PATH="/usr/local/cuda-11.8"
CUDA_LIB_PATH="$CUDA_PATH/lib64"

# Verify CUDA installation paths
if ! check_dir "$CUDA_PATH"; then
    echo "Error: CUDA installation not found at $CUDA_PATH"
    echo "Please verify your CUDA installation"
    exit 1
fi

if ! check_dir "$CUDA_LIB_PATH"; then
    echo "Error: CUDA libraries not found at $CUDA_LIB_PATH"
    echo "Please verify your CUDA installation"
    exit 1
fi

# Set environment variables
export CUDA_PATH
export PATH="$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_LIB_PATH:$LD_LIBRARY_PATH"

# Print environment
echo -e "\nCUDA Environment:"
echo "CUDA_PATH: $CUDA_PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Verify cargo is installed
if ! command -v cargo &> /dev/null; then
    echo "Error: Cargo not found. Please install Rust and Cargo"
    exit 1
fi

# Run the program
echo -e "\nRunning program with CUDA environment..."
cargo run "$@" 