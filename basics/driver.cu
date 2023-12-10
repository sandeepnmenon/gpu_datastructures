#include <iostream>
#include <map>
#include <unistd.h>
#include <functional>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>
#include <thrust/sequence.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "basic_hashmap.cu"
#include "utils.cuh"

namespace cg = cooperative_groups;

__global__ void testIntInsertCG(const int *keys, const int *values, const size_t numElements, Hashmap<int, int> *hashmap)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numElements)
    {
        auto group = cg::tiled_partition<4>(cg::this_thread_block());
        hashmap->insert(group, keys[idx], values[idx]);
    }
}

__global__ void testIntInsert(const int *keys, const int *values, const size_t numElements, Hashmap<int, int> *hashmap)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numElements)
    {
        hashmap->insert(keys[idx], values[idx]);
    }
}

void insertionBenchmarkFunc(Hashmap<int, int> *hashmap, const thrust::device_vector<int> &d_keys, const thrust::device_vector<int> &d_values)
{
    // Define grid and block sizes
    int numElements = d_keys.size();
    int blockSize = 256;
    int gridSize = (numElements + blockSize - 1) / blockSize;

    testIntInsert<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(d_keys.data()), thrust::raw_pointer_cast(d_values.data()), numElements, hashmap);
    cudaDeviceSynchronize();
}

void insertionBenchmarkCGFunc(Hashmap<int, int> *hashmap, const thrust::device_vector<int> &d_keys, const thrust::device_vector<int> &d_values)
{
    // Define grid and block sizes
    int numElements = d_keys.size();
    int blockSize = 256;
    int gridSize = (numElements + blockSize - 1) / blockSize;

    testIntInsertCG<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(d_keys.data()), thrust::raw_pointer_cast(d_values.data()), numElements, hashmap);
    cudaDeviceSynchronize();
}

void searchBenchMarkFunc(Hashmap<int, int> *hashmap, const thrust::device_vector<int> &d_keys, thrust::device_vector<int> &d_results)
{
    hashmap->getValues(d_keys, d_results);
    cudaDeviceSynchronize();
}

bool defaultInsert = false;
bool cooperativeGroupsInsert = false;
bool defaultSearch = false;
std::map<char, std::function<void()>> actions;

void setupActions()
{
    actions['d'] = [&]()
    { defaultInsert = true; std::cout << "Default insert\n"; };
    actions['c'] = [&]()
    { cooperativeGroupsInsert = true; std::cout << "Cooperative groups insert\n"; };
    actions['s'] = [&]()
    { defaultSearch = true; std::cout << "Default search\n"; };
}

int main(int argc, char **argv)
{   
    cudaSetDevice(2);

    setupActions();

    int opt;
    while ((opt = getopt(argc, argv, "dcs")) != -1)
        if (actions.find(opt) != actions.end())
            actions[opt]();
        else
            std::cerr << "Unknown option: " << char(opt) << std::endl;

    // Initialize data
    const size_t numElements = 1000000; // Adjust as needed
    thrust::host_vector<int> h_keys(numElements), h_values(numElements);

    // Fill keys and values with test data
    initializeData(h_keys, numElements);
    initializeData(h_values, numElements);

    // Copy data from host to device
    thrust::device_vector<int> d_keys = h_keys;
    thrust::device_vector<int> d_values = h_values;

    // Create and initialize hashmap
    int capacity = 2 * numElements; // Or any other size you prefer

    Hashmap<int, int> *hashmap; // Assuming constructor initializes the GPU memory
    cudaMallocManaged(&hashmap, sizeof(Hashmap<int, int>));
    new (hashmap) Hashmap<int, int>(capacity);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error malloc memcpy: " << cudaGetErrorString(err) << std::endl;
        // handle error
    }

    if (defaultInsert)
        benchmarkKernel([&]()
                        { insertionBenchmarkFunc(hashmap, d_keys, d_values); },
                        "Insertion");

    if (cooperativeGroupsInsert)
        benchmarkKernel([&]()
                        { insertionBenchmarkCGFunc(hashmap, d_keys, d_values); },
                        "Insertion CG");

    if (defaultSearch)
    {
        thrust::device_vector<int> d_results(d_keys.size());
        benchmarkKernel([&]()
                        { searchBenchMarkFunc(hashmap, d_keys, d_results); },
                        "Search");

        checkResults(d_results, h_values);
    }

    // Cleanup
    hashmap->~Hashmap();
    cudaFree(hashmap);

    return 0;
}
