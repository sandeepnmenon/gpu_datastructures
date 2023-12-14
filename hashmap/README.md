# GPU-Accelerated Hashmaps

## Description
This project is a custom implemmentation of a GPU Hashmap Data structure with the insert and find functionalities. The project is written in C++ and CUDA. The project is a part of the course "GPU Programming" at New York University.

## Usage
The usage of the code is demonstrated in `driver.cu` and `kernels.cu`. 
`basic_hashmap.cu` contains the hashmap implementation. The definition of the hashmap is as follows:
```cpp
template <typename Key, typename Value>
class Hashmap
{
public:
    Hashmap(size_t capacity);
    ~Hashmap();

    __device__ bool insert(const Key k, const Value v);

    __device__ bool insert(cg::thread_block_tile<CG_SIZE> group, const Key k, const Value v);

    __device__ Value find(const Key k) const;

    __device__ void find(cg::thread_block_tile<CG_SIZE> group, const Key k, Value *out) const;

    Bucket<Key, Value> *buckets;
    size_t capacity{};
```
Example usage
```cpp
#include "hashmap_gpu.cu"   // Include the hashmap header file

Hashmap<int, int> *hashmap; // Integer to Integer hashmap
cudaMallocManaged(&hashmap, sizeof(Hashmap<int, int>));
new (hashmap) Hashmap<int, int>(capacity);

// Initialize the hashmap. This is done by default in the constructor. You can call it again if you want to reinitialize the hashmap.    
hashmap->initialize();

// Using this hashmap in a kernel
__global__ void testIntInsert(const int *keys, const int *values, const size_t numElements, Hashmap<int, int> *hashmap)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numElements)
    {
        hashmap->insert(keys[idx], values[idx]);
    }
}

// Using this hashmap in a kernel with cooperative groups
__global__ void testIntInsertCG(const int *keys, const int *values, const size_t numElements, Hashmap<int, int> *hashmap)
{
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) / CG_SIZE;
    if (idx < numElements)
    {
        auto group = cg::tiled_partition<CG_SIZE>(cg::this_thread_block());
        hashmap->insert(group, keys[idx], values[idx]);
    }
}
```
The `find` function can be used in a similar way by calling `hashmap->find(key)` or `hashmap->find(group, key, &out)`.

## Running Experiments
**Note:The code runs best on the cuda2 cluster. Other clusters might give errors**
1. Build using the `make` command.
2. Run the code using this template
```bash
./driver -n <num_elements> -l <load_factor> -t <threads_per_block> -b <num_blocks> -iIsS
```
* `-n` is the number of elements to be inserted into the hashmap.
* `-l` is the load factor of the hashmap.
* `-t` is the number of threads per block.
* `-b` is the number of blocks.
* `-i` runs default insert
* `-I` runs insert with cooperative groups
* `-s` runs default search
* `-S` runs search with cooperative groups

For benchmarking run `./benchmark.sh` and you will get the results in `benchmark_results.csv`..
