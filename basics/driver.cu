#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>
#include "basic_hashmap.cu"

namespace cg = cooperative_groups;

__global__ void testIntInsert(int *keys, int *values, size_t numElements, Hashmap<int, int> *hashmap)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numElements)
    {
        auto group = cg::tiled_partition<4>(cg::this_thread_block());
        hashmap->insert(group, keys[idx], values[idx]);
    }
}

int main()
{
    // Initialize data
    const size_t numElements = 10000; // Adjust as needed
    int *keys = new int[numElements];
    int *values = new int[numElements];

    // Fill keys and values with test data
    for (size_t i = 0; i < numElements; i++)
    {
        keys[i] = i;
        values[i] = i;
    }

    // Allocate memory on GPU and copy data
    int *d_keys;
    int *d_values;
    cudaMalloc(&d_keys, numElements * sizeof(int));
    cudaMalloc(&d_values, numElements * sizeof(int));
    cudaMemcpy(d_keys, keys, numElements * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values, numElements * sizeof(int), cudaMemcpyHostToDevice);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error malloc memcpy: " << cudaGetErrorString(err) << std::endl;
        // handle error
    }

    // Define grid and block sizes
    int blockSize = 256;
    int gridSize = (numElements + blockSize - 1) / blockSize;

    // Create and initialize hashmap
    size_t capacity = 10000; // Or any other size you prefer

    Hashmap<int, int> *hashmap = new Hashmap<int, int>(capacity); // Assuming constructor initializes the GPU memory
    // ...

    // Start benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch kernel
    testIntInsert<<<gridSize, blockSize>>>(d_keys, d_values, numElements, hashmap);
    cudaDeviceSynchronize(); // Wait for the kernel to finish
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error kernel: " << cudaGetErrorString(err) << std::endl;
        // handle error
    }

    // End benchmark
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Insertion time: " << milliseconds << " ms\n";

    // Cleanup
    cudaFree(d_keys);
    cudaFree(d_values);
    delete hashmap;
    delete[] keys;
    delete[] values;

    return 0;
}
