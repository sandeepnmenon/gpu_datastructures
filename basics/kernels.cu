#include "basic_hashmap.cu"
#include "kernels.cuh"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__global__ void testIntInsertCG(const int *keys, const int *values, const size_t numElements, Hashmap<int, int> *hashmap, size_t cg_size)
{
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) / cg_size;
    if (idx < numElements)
    {
        auto group = cg::tiled_partition<4>(cg::this_thread_block());
        if (!hashmap->insert(group, keys[idx], values[idx]))
        {
            printf("Insertion failed for key[%d] %d\n", idx, keys[idx]);
        }
    }
}

__global__ void testIntInsertCG_2(const int *keys, const int *values, const size_t numElements, Hashmap<int, int> *hashmap, size_t cg_size)
{
    int threadId = (threadIdx.x + blockIdx.x * blockDim.x) / cg_size;
    int totalThreads = (gridDim.x * blockDim.x) / cg_size; // Total number of active threads

    for (int idx = threadId; idx < numElements; idx += totalThreads)
    {
        auto group = cg::tiled_partition<4>(cg::this_thread_block());
        if (!hashmap->insert(group, keys[idx], values[idx]))
        {
            printf("Insertion failed for key[%d] %d\n", idx, keys[idx]);
        }
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
__global__ void findKernelCG(const Hashmap<Key, Value> *hashmap, const Key *keys, Value *results, int numValues, size_t cg_size)
{
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) / cg_size;
    if (idx < numValues)
    {
        auto group = cg::tiled_partition<4>(cg::this_thread_block());
        results[idx] = hashmap->find(group, keys[idx]);
    }
}

template <typename Key, typename Value>
__global__ void findKernelCG_2(const Hashmap<Key, Value> *hashmap, const Key *keys, Value *results, int numValues, size_t cg_size)
{
    int threadId = (threadIdx.x + blockIdx.x * blockDim.x) / cg_size;
    int totalThreads = (gridDim.x * blockDim.x) / cg_size; // Total number of active threads

    for (int idx = threadId; idx < numValues; idx += totalThreads)
    {
        auto group = cg::tiled_partition<4>(cg::this_thread_block());
        results[idx] = hashmap->find(group, keys[idx]);
    }
}

template __global__ void findKernel<int, int>(const Hashmap<int, int> *hashmap, const int *keys, int *results, int numValues);
