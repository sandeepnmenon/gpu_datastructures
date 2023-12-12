#ifndef HASHMAP_H
#define HASHMAP_H
#include <memory>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <cooperative_groups.h>
#include "config.cuh"
namespace cg = cooperative_groups;

enum class probing_state
{
    SUCCESS,
    DUPLICATE,
    CONTINUE
};

template <typename T1, typename T2>
struct Pair
{
    T1 first;
    T2 second;

    __device__ Pair() : first(T1()), second(T2()) {}

    __device__ Pair(const T1 &a, const T2 &b) : first(a), second(b) {}
};

template <typename Key, typename Value>
struct Bucket
{
    Key key;
    Value value;
    // Other fields or methods for atomic operations

    __device__ Pair<Key, Value> load(std::memory_order order) const
    {
        return Pair<Key, Value>(key, value);
    }

    __device__ bool compare_exchange_strong(Pair<Key, Value> expected,
                                            Pair<Key, Value> desired,
                                            std::memory_order order)
    {
        return atomicCAS(reinterpret_cast<unsigned long long int *>(this),
                         *reinterpret_cast<unsigned long long int *>(&expected),
                         *reinterpret_cast<unsigned long long int *>(&desired)) ==
               *reinterpret_cast<unsigned long long int *>(&expected);
    }
};

template <typename T>
__host__ __device__ unsigned int hash_custom(T key)
{
    return static_cast<unsigned int>(key) * 2654435761u;
}

__device__ static constexpr int empty_sentinel = -1; // Or any other appropriate value

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

    void printAllBucketValues()
    {
        printf("Printing all %lu: bucket values\n", capacity);
        Bucket<Key, Value> *host_buckets;
        cudaMallocHost(&host_buckets, capacity * sizeof(Bucket<Key, Value>)); // Allocate pinned host memory
        cudaMemcpy(host_buckets, this->buckets, capacity * sizeof(Bucket<Key, Value>), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        for (int i = 0; i < capacity; i++)
        {
            auto old_kv = host_buckets[i];
            printf("key:%d value:%d i: %d\n", old_kv.key, old_kv.value, i);
        }
        printf("Over...\n");
        cudaFreeHost(host_buckets); // Free the allocated host memory
    }

    void initialize()
    {
        cudaMemset(buckets, empty_sentinel, capacity * sizeof(Bucket<Key, Value>)); // Initialize to default values
    }
};

template <typename Key, typename Value>
Hashmap<Key, Value>::Hashmap(size_t cap) : capacity{cap}, buckets(nullptr)
{
    cudaMalloc(&buckets, capacity * sizeof(Bucket<Key, Value>));
    initialize();
}

template <typename Key, typename Value>
Hashmap<Key, Value>::~Hashmap()
{
    cudaFree(buckets);
}

template <typename Key, typename Value>
__device__ bool Hashmap<Key, Value>::insert(Key k, Value v)
{
    // get initial probing position from the hash value of the key
    auto i = hash_custom(k) % capacity;
    while (true)
    {
        // load the content of the bucket at the current probe position
        auto old_kv = buckets[i].load(std::memory_order_relaxed);
        // if the bucket is empty we can attempt to insert the pair
        if (old_kv.first == empty_sentinel)
        {
            // try to atomically replace the current content of the bucket with the input pair
            Pair<Key, Value> desired(k, v);
            bool const success = buckets[i].compare_exchange_strong(
                old_kv, desired, std::memory_order_relaxed);
            if (success)
            {
                // store was successful
                return true;
            }
        }
        else if (old_kv.first == k)
        {
            // input key is already present in the map
            return false;
        }
        // if the bucket was already occupied move to the next (linear) probing position
        // using the modulo operator to wrap back around to the beginning if we
        // go beyond the capacity
        i = ++i % capacity;
    }
}

template <typename Key, typename Value>
__device__ bool Hashmap<Key, Value>::insert(cg::thread_block_tile<CG_SIZE> group, Key k, Value v)
{
    // get initial probing position from the hash value of the key
    auto i = (hash_custom(k) + group.thread_rank()) % capacity;
    auto state = probing_state::CONTINUE;
    // printf("inserting key:%d value:%d i: %d\n", k, v, i);

    while (true)
    {
        // load the contents of the bucket at the current probe position of each rank in a coalesced manner
        auto old_kv = buckets[i].load(std::memory_order_relaxed);
        // input key is already present in the map
        if (group.any(old_kv.first == k))
            return false;
        // each rank checks if its current bucket is empty, i.e., a candidate bucket for insertion
        auto const empty_mask = group.ballot(old_kv.first == empty_sentinel);
        // it there is an empty buckets in the group's current probing window
        if (empty_mask)
        {
            // elect a candidate rank (here: thread with lowest rank in mask)
            auto const candidate = __ffs(empty_mask) - 1;
            // printf("candidate: %d, rank: %d key: %d value: %d i: %d\n", candidate, group.thread_rank(), k, v, i);
            if (group.thread_rank() == candidate)
            {
                // attempt atomically swapping the input Pair into the bucket
                Pair<Key, Value> desired(k, v);
                bool const success = buckets[i].compare_exchange_strong(
                    old_kv, desired, std::memory_order_relaxed);
                if (success)
                {
                    // printf("inserted key:%d value:%d i: %d\n", k, v, i);
                    //  insertion went successful
                    state = probing_state::SUCCESS;
                }
                else if (old_kv.first == k)
                {
                    // printf("duplicate key:%d value:%d i: %d\n", k, v, i);
                    //  else, re-check if a duplicate key has been inserted at the current probing position
                    state = probing_state::DUPLICATE;
                }
            }
            // broadcast the insertion result from the candidate rank to all other ranks
            auto const candidate_state = group.shfl(state, candidate);
            // printf("candidate_state: %d key: %d value: %d i: %d\n", candidate_state, k, v, i);
            if (candidate_state == probing_state::SUCCESS)
                return true;
            if (candidate_state == probing_state::DUPLICATE)
                return false;
        }
        else
        {
            // printf("continuing key:%d value:%d i: %d\n", k, v, i);
            //  else, move to the next (linear) probing window
            i = (i + group.size()) % capacity;
        }
    }
}

template <typename Key, typename Value>
__device__ Value Hashmap<Key, Value>::find(const Key k) const
{
    auto i = hash_custom(k) % capacity;
    while (true)
    {
        auto old_kv = buckets[i].load(std::memory_order_relaxed);
        if (old_kv.first == k)
        {
            // Found the key, return the value
            return old_kv.second;
        }
        else if (old_kv.first == empty_sentinel)
        {
            // Key not found
            return empty_sentinel;
        }
        i = ++i % capacity;
    }
}

template <typename Key, typename Value>
__device__ void Hashmap<Key, Value>::find(cg::thread_block_tile<CG_SIZE> group, const Key k, Value *out) const
{
    auto i = (hash_custom(k) + group.thread_rank()) % capacity;
    auto thread_rank = group.thread_rank();
    while (true)
    {
        auto old_kv = buckets[i].load(std::memory_order_relaxed);
        auto found_mask = group.ballot(old_kv.first == k);
        if (found_mask)
        {
            if (old_kv.first == k)
            {
                *out = old_kv.second;
                return;
            }
            return;
        }
        else if (group.all(old_kv.first == empty_sentinel))
        {
            // If all threads in the group find an empty sentinel, the key is not present
            *out = empty_sentinel;
            return;
        }

        i = (i + group.size()) % capacity;
    }
}

#endif // HASHMAP_H