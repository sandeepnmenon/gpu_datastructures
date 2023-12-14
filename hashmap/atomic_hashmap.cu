#include <memory>
#include <cuda/atomic>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
using namespace std;

enum class probing_state
{
    SUCCESS,
    DUPLICATE,
    CONTINUE
};

template <typename Key, typename Value>
struct Pair
{
    Key first;
    Value second;

    __host__ __device__ Pair() : first(), second() {}
    __host__ __device__ Pair(Key k, Value v) : first(k), second(v) {}

    static Pair load(cuda::std::atomic<Pair> &atomicPair, cuda::std::memory_order order = cuda::std::memory_order_relaxed)
    {
        Pair temp = atomicPair.load(order);
        return temp;
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

    __device__ bool insert(Key k, Value v);

    __device__ bool insert(cg::thread_block_tile<4> group, Key k, Value v);

    __device__ bool erase(cg::thread_block_tile<4> group, Key k);

    void getValues(const thrust::device_vector<Key> &keys, thrust::device_vector<Value> &results);

    cuda::std::atomic<Pair<Key, Value>> *buckets;
    size_t capacity{};
    // Other private members and methods
};

template <typename Key, typename Value>
Hashmap<Key, Value>::Hashmap(size_t cap) : capacity{cap}, buckets(nullptr)
{
    cudaMalloc(&buckets, capacity * sizeof(cuda::std::atomic<Pair<Key, Value>>));
    cudaMemset(buckets, empty_sentinel, capacity * sizeof(cuda::std::atomic<Pair<Key, Value>>)); // Initialize to default values
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
        auto temp = buckets[i].load();
        auto [old_k, old_v] = temp;

        // if the bucket is empty we can attempt to insert the pair
        if (old_k == empty_sentinel)
        {
            // try to atomically replace the current content of the bucket with the input pair
            bool success = buckets[i].compare_exchange_strong(
                {old_k, old_v}, {k, v});
            if (success)
            {
                // store was successful
                return true;
            }
        }
        else if (old_k == k)
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
        auto [old_k, old_v] = buckets[i].load();
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
