#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>

#include "basic_hashmap.cu"

namespace cg = cooperative_groups;

__global__ void testIntInsertCG(int *keys, int *values, size_t numElements, Hashmap<int, int> *hashmap)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numElements)
    {
        auto group = cg::tiled_partition<4>(cg::this_thread_block());
        hashmap->insert(group, keys[idx], values[idx]);
    }
}

__global__ void testIntInsert(int *keys, int *values, size_t numElements, Hashmap<int, int> *hashmap)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numElements)
    {
        hashmap->insert(keys[idx], values[idx]);
    }
}

int main()
{
    // Initialize data
    const size_t numElements = 200; // Adjust as needed
    thrust::host_vector<int> keys(numElements);
    thrust::host_vector<int> values(numElements);

    // Fill keys and values with test data
    thrust::sequence(keys.begin(), keys.end());
    thrust::sequence(values.begin(), values.end());
    thrust::default_random_engine gen;
    thrust::shuffle(keys.begin(), keys.end(), gen);

    // Copy data from host to device
    thrust::device_vector<int> d_keys = keys;
    thrust::device_vector<int> d_values = values;

    // Define grid and block sizes
    int blockSize = 256;
    int gridSize = (numElements + blockSize - 1) / blockSize;

    // Create and initialize hashmap
    int capacity = 100000; // Or any other size you prefer

    Hashmap<int, int> *hashmap; // Assuming constructor initializes the GPU memory
    cudaMallocManaged(&hashmap, sizeof(Hashmap<int, int>));
    new (hashmap) Hashmap<int, int>(capacity);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error malloc memcpy: " << cudaGetErrorString(err) << std::endl;
        // handle error
    }
    // ...

    // Start benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch kernel
    testIntInsert<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(d_keys.data()), thrust::raw_pointer_cast(d_values.data()), numElements, hashmap);
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

    // Check if all elements were inserted
    thrust::device_vector<int> d_results(d_keys.size());
    hashmap->getValues(d_keys, d_results);
    thrust::host_vector<int> h_results = d_results;

    bool areEqual = thrust::equal(h_results.begin(), h_results.end(), values.begin());
    if (areEqual)
    {
        std::cout << "Success: d_results and h_values are the same." << std::endl;
    }
    else
    {
        std::cout << "Error: d_results and h_values differ." << std::endl;
    }

    // Cleanup
    hashmap->~Hashmap();
    cudaFree(hashmap);

    return 0;
}
