#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    int count;
    cudaError_t error = cudaGetDeviceCount(&count);
    if (error != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(error));
        return 1;
    }
    printf("Found %d CUDA devices\n", count);
    return 0;
} 
