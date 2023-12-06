
#include <memory>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
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

// Simple hash function for integers
__host__ __device__ unsigned int hash_custom(int key)
{
    // Simple example hash function (you can replace it with a better one)
    return key * 2654435761u;
}
__device__ static constexpr int empty_sentinel = -1; // Or any other appropriate value

template <typename Key, typename Value>
class Hashmap
{
public:
    Hashmap(size_t capacity);
    ~Hashmap();

    __device__ bool insert(Key k, Value v);

    __device__ bool insert(cg::thread_block_tile<4> group, Key k, Value v);

    __device__ Value find(Key k);

    __device__ bool erase(cg::thread_block_tile<4> group, Key k);

    void getValues(const thrust::device_vector<Key> &keys, thrust::device_vector<Value> &results);

    Bucket<Key, Value> *buckets;
    size_t capacity{};
    // Other private members and methods
};

template <typename Key, typename Value>
Hashmap<Key, Value>::Hashmap(size_t cap) : capacity{cap}, buckets(nullptr)
{
    cudaMalloc(&buckets, capacity * sizeof(Bucket<Key, Value>));
    cudaMemset(buckets, empty_sentinel, capacity * sizeof(Bucket<Key, Value>)); // Initialize to default values
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
__device__ bool Hashmap<Key, Value>::insert(cg::thread_block_tile<4> group, Key k, Value v)
{
    // get initial probing position from the hash value of the key
    auto i = (hash_custom(k) + group.thread_rank()) % capacity;
    auto state = probing_state::CONTINUE;
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
            if (group.thread_rank() == candidate)
            {
                // attempt atomically swapping the input Pair into the bucket
                Pair<Key, Value> desired(k, v);
                bool const success = buckets[i].compare_exchange_strong(
                    old_kv, desired, std::memory_order_relaxed);
                if (success)
                {
                    // insertion went successful
                    state = probing_state::SUCCESS;
                }
                else if (old_kv.first == k)
                {
                    // else, re-check if a duplicate key has been inserted at the current probing position
                    state = probing_state::DUPLICATE;
                }
            }
            // broadcast the insertion result from the candidate rank to all other ranks
            auto const candidate_state = group.shfl(state, candidate);
            if (candidate_state == probing_state::SUCCESS)
                return true;
            if (candidate_state == probing_state::DUPLICATE)
                return false;
        }
        else
        {
            // else, move to the next (linear) probing window
            i = (i + group.size()) % capacity;
        }
    }
}

template <typename Key, typename Value>
__device__ Value Hashmap<Key, Value>::find(Key k)
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
            return Value{};
        }
    }
}

template <typename Key, typename Value>
__device__ bool Hashmap<Key, Value>::erase(cg::thread_block_tile<4> group, Key k)
{
    auto i = (hash_custom(k) + group.thread_rank()) % capacity;
    while (true)
    {
        auto old_kv = buckets[i].load(std::memory_order_relaxed);
        if (group.any(old_kv.first == k))
        {
            // Found the key, attempt to remove it
            Pair<Key, Value> empty(empty_sentinel, Value{});
            bool const success = buckets[i].compare_exchange_strong(
                old_kv, empty, std::memory_order_relaxed);
            return success;
        }
        else if (old_kv.first == empty_sentinel)
        {
            // Key not found
            return false;
        }
        i = (i + group.size()) % capacity;
    }
}

template <typename Key, typename Value>
__global__ void findKernel(Hashmap<Key, Value> *hashmap, const Key *keys, Value *results, int numValues)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numValues)
    {
        results[idx] = hashmap->find(keys[idx]);
    }
}

template <typename Key, typename Value>
void Hashmap<Key, Value>::getValues(const thrust::device_vector<Key> &keys, thrust::device_vector<Value> &results)
{
    int blockSize = 256;
    int gridSize = (keys.size() + blockSize - 1) / blockSize;
    findKernel<<<gridSize, blockSize>>>(this, thrust::raw_pointer_cast(keys.data()), thrust::raw_pointer_cast(results.data()), keys.size());
}

// Explicit template instantiation
template class Hashmap<int, int>; // Example instantiation, adjust types as needed
