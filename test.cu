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
    
    // Get device properties
    for (int i = 0; i < count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total Memory: %lu MB\n", prop.totalGlobalMem / (1024*1024));
    }
    return 0;
} 
