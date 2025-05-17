#!/bin/bash
set -e  # Exit on error

echo "Setting up CUDA environment..."

# Function to check if a directory exists
check_dir() {
    if [ ! -d "$1" ]; then
        return 1
    fi
    return 0
}

# Function to find CUDA installation
find_cuda_path() {
    # Common CUDA installation paths on Debian/Ubuntu
    local possible_paths=(
        "/usr/local/cuda-11.8"
        "/usr/local/cuda"
        "/usr/lib/cuda"
        "/usr/lib/nvidia-cuda-toolkit"
        "/usr/lib/nvidia-cuda"
    )

    for path in "${possible_paths[@]}"; do
        if check_dir "$path"; then
            echo "$path"
            return 0
        fi
    done

    # If not found in common paths, try to find using nvcc
    if command -v nvcc &> /dev/null; then
        local nvcc_path=$(which nvcc)
        local cuda_path=$(dirname $(dirname "$nvcc_path"))
        if check_dir "$cuda_path"; then
            echo "$cuda_path"
            return 0
        fi
    fi

    return 1
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

# Find CUDA installation
echo -e "\nSearching for CUDA installation..."
CUDA_PATH=$(find_cuda_path)
if [ -z "$CUDA_PATH" ]; then
    echo "Error: Could not find CUDA installation"
    echo "Please verify your CUDA installation"
    exit 1
fi

echo "Found CUDA installation at: $CUDA_PATH"

# Set CUDA paths
CUDA_LIB_PATH="$CUDA_PATH/lib64"
if [ ! -d "$CUDA_LIB_PATH" ]; then
    # Try alternative lib path
    CUDA_LIB_PATH="$CUDA_PATH/lib"
    if [ ! -d "$CUDA_LIB_PATH" ]; then
        echo "Error: Could not find CUDA libraries directory"
        exit 1
    fi
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