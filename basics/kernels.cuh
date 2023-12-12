#ifndef KERNELS_CUH_
#define KERNELS_CUH_

#include "basic_hashmap.cu"

// Declare the kernel functions
__global__ void testIntInsertCG(const int *keys, const int *values, const size_t numElements, Hashmap<int, int> *hashmap, size_t cg_size);
__global__ void testIntInsertCG_2(const int *keys, const int *values, const size_t numElements, Hashmap<int, int> *hashmap, size_t cg_size);
__global__ void testIntInsert(const int *keys, const int *values, const size_t numElements, Hashmap<int, int> *hashmap);
__global__ void testIntInsert_2(const int *keys, const int *values, const size_t numElements, Hashmap<int, int> *hashmap);

template <typename Key, typename Value>
__global__ void findKernel(const Hashmap<Key, Value> *hashmap, const Key *keys, Value *results, int numValues);

#endif // KERNELS_CUH_
