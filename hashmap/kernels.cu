#include "hashmap_gpu.cu"
#include "kernels.cuh"
#include "config.cuh"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__global__ void testIntInsertCG(const int *keys, const int *values, const size_t numElements, Hashmap<int, int> *hashmap)
{
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) / CG_SIZE;
    if (idx < numElements)
    {
        auto group = cg::tiled_partition<CG_SIZE>(cg::this_thread_block());
        hashmap->insert(group, keys[idx], values[idx]);
    }
}

__global__ void testIntInsertCG_2(const int *keys, const int *values, const size_t numElements, Hashmap<int, int> *hashmap)
{
    int threadId = (threadIdx.x + blockIdx.x * blockDim.x) / CG_SIZE;
    int totalThreads = (gridDim.x * blockDim.x) / CG_SIZE; // Total number of active threads

    for (int idx = threadId; idx < numElements; idx += totalThreads)
    {
        auto group = cg::tiled_partition<CG_SIZE>(cg::this_thread_block());
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

__global__ void testIntInsert_2(const int *keys, const int *values, const size_t numElements, Hashmap<int, int> *hashmap)
{
    int threadId = threadIdx.x + blockIdx.x * blockDim.x; // unique thread id
    int totalThreads = gridDim.x * blockDim.x;            // Total number of active threads

    // Loop over elements. Each thread handles multiple insertions.
    for (int idx = threadId; idx < numElements; idx += totalThreads)
    {
        hashmap->insert(keys[idx], values[idx]);
    }
}

template <typename Key, typename Value>
__global__ void findKernel(const Hashmap<Key, Value> *hashmap, const Key *keys, Value *results, int numValues)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numValues)
    {
        results[idx] = hashmap->find(keys[idx]);
    }
}

template <typename Key, typename Value>
__global__ void findKernel_2(const Hashmap<Key, Value> *hashmap, const Key *keys, Value *results, int numValues)
{
    int threadId = threadIdx.x + blockIdx.x * blockDim.x; // unique thread id
    int totalThreads = gridDim.x * blockDim.x;            // Total number of active threads

    // Loop over elements. Each thread handles multiple insertions.
    for (int idx = threadId; idx < numValues; idx += totalThreads)
    {
        results[idx] = hashmap->find(keys[idx]);
    }
}

template <typename Key, typename Value>
__global__ void findKernelCG(const Hashmap<Key, Value> *hashmap, const Key *keys, Value *results, int numValues)
{
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) / CG_SIZE;
    if (idx < numValues)
    {
        auto group = cg::tiled_partition<CG_SIZE>(cg::this_thread_block());
        hashmap->find(group, keys[idx], &results[idx]);
    }
}

template <typename Key, typename Value>
__global__ void findKernelCG_2(const Hashmap<Key, Value> *hashmap, const Key *keys, Value *results, int numValues)
{
    int threadId = (threadIdx.x + blockIdx.x * blockDim.x) / CG_SIZE;
    int totalThreads = (gridDim.x * blockDim.x) / CG_SIZE; // Total number of active threads

    for (int idx = threadId; idx < numValues; idx += totalThreads)
    {
        auto group = cg::tiled_partition<CG_SIZE>(cg::this_thread_block());
        hashmap->find(group, keys[idx], &results[idx]);
    }
}

// Explicit instantiation because these are templated functions
template __global__ void findKernel<int, int>(const Hashmap<int, int> *hashmap, const int *keys, int *results, int numValues);
template __global__ void findKernel_2<int, int>(const Hashmap<int, int> *hashmap, const int *keys, int *results, int numValues);
template __global__ void findKernelCG<int, int>(const Hashmap<int, int> *hashmap, const int *keys, int *results, int numValues);
template __global__ void findKernelCG_2<int, int>(const Hashmap<int, int> *hashmap, const int *keys, int *results, int numValues);
