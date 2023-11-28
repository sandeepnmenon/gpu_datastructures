#include "basic_hashmap.cuh"

template <typename Key, typename Value>
Hashmap<Key, Value>::Hashmap(size_t cap) : capacity(cap), buckets(nullptr)
{
    cudaMalloc(&buckets, capacity * sizeof(Bucket<Key, Value>));
    cudaMemset(buckets, 0, capacity * sizeof(Bucket<Key, Value>)); // Initialize to default values
}

template <typename Key, typename Value>
Hashmap<Key, Value>::~Hashmap()
{
    cudaFree(buckets);
}

template <typename Key, typename Value>
__device__ bool Hashmap<Key, Value>::insert(cg::thread_block_tile<4> group, Key k, Value v)
{
    // get initial probing position from the hash value of the key
    auto i = (hash(k) + group.thread_rank()) % capacity;
    auto state = probing_state::CONTINUE;
    while (true)
    {
        // load the contents of the bucket at the current probe position of each rank in a coalesced manner
        auto [old_k, old_v] = buckets[i].load(memory_order_relaxed);
        // input key is already present in the map
        if (group.any(old_k == k))
            return false;
        // each rank checks if its current bucket is empty, i.e., a candidate bucket for insertion
        auto const empty_mask = group.ballot(old_k == empty_sentinel);
        // it there is an empty buckets in the group's current probing window
        if (empty_mask)
        {
            // elect a candidate rank (here: thread with lowest rank in mask)
            auto const candidate = __ffs(empty_mask) - 1;
            if (group.thread_rank() == candidate)
            {
                // attempt atomically swapping the input pair into the bucket
                bool const success = buckets[i].compare_exchange_strong(
                    {old_k, old_v}, {k, v}, memory_order_relaxed);
                if (success)
                {
                    // insertion went successful
                    state = probing_state::SUCCESS;
                }
                else if (old_k == k)
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
__device__ Value *Hashmap<Key, Value>::find(cg::thread_block_tile<4> group, Key k)
{
    auto i = (hash(k) + group.thread_rank()) % capacity;
    while (true)
    {
        auto [old_k, old_v] = buckets[i].load(memory_order_relaxed);
        if (group.any(old_k == k))
        {
            // Found the key
            return &old_v;
        }
        else if (old_k == empty_sentinel)
        {
            // Key not found
            return nullptr;
        }
        i = (i + group.size()) % capacity;
    }
}

template <typename Key, typename Value>
__device__ bool Hashmap<Key, Value>::erase(cg::thread_block_tile<4> group, Key k)
{
    auto i = (hash(k) + group.thread_rank()) % capacity;
    while (true)
    {
        auto [old_k, old_v] = buckets[i].load(memory_order_relaxed);
        if (group.any(old_k == k))
        {
            // Found the key, attempt to remove it
            bool const success = buckets[i].compare_exchange_strong(
                {old_k, old_v}, {empty_sentinel, Value{}}, memory_order_relaxed);
            return success;
        }
        else if (old_k == empty_sentinel)
        {
            // Key not found
            return false;
        }
        i = (i + group.size()) % capacity;
    }
}

// Explicit template instantiation
template class Hashmap<int, int>; // Example instantiation, adjust types as needed
