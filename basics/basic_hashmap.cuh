#ifndef BASIC_HASHMAP_H
#define BASIC_HASHMAP_H

#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

enum class probing_state
{
    SUCCESS,
    DUPLICATE,
    CONTINUE
};

template <typename Key, typename Value>
struct Bucket
{
    Key key;
    Value value;
    // Other fields or methods for atomic operations
};

template <typename Key, typename Value>
class Hashmap
{
public:
    Hashmap(size_t capacity);
    ~Hashmap();

    __device__ bool insert(cg::thread_block_tile<4> group, Key k, Value v);

    __device__ Value *find(cg::thread_block_tile<4> group, Key k);

    __device__ bool erase(cg::thread_block_tile<4> group, Key k);

    // Additional functions like find, erase, etc.

private:
    Bucket<Key, Value> *buckets;
    size_t capacity;
    // Other private members and methods
};

#endif // BASIC_HASHMAP_H
