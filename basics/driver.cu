#include <iostream>
#include <iomanip>
#include <map>
#include <unistd.h>
#include <functional>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/sequence.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cassert>

#include "basic_hashmap.cu"
#include "kernels.cuh"
#include "utils.cuh"

struct Config
{

    bool defaultInsert = false;
    bool cooperativeGroupsInsert = false;
    bool defaultSearch = false;
    size_t device = 0;
    size_t threads = 1;
    size_t blocks = 1;
    size_t numElements = 1;
    static constexpr size_t cg_size = 4;
    float load = 1.0f;

} config;

std::map<char, std::function<void(const char *)>> actions;

void insertionBenchmarkFunc(Hashmap<int, int> *hashmap, const thrust::device_vector<int> &d_keys, const thrust::device_vector<int> &d_values)
{
    // Define grid and block sizes
    int numElements = config.numElements;
    int blockSize = config.threads;
    int gridSize = (numElements + blockSize - 1) / blockSize;

    std::cout << std::setw(25) << "Threads per block:" << blockSize << "\n";
    std::cout << std::setw(25) << "Number of blocks:" << gridSize << "\n";
    std::cout << std::setw(25) << "Total number of threads:" << blockSize * gridSize << "\n";

    testIntInsert<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(d_keys.data()), thrust::raw_pointer_cast(d_values.data()), numElements, hashmap);
    cudaDeviceSynchronize();
}

void insertionBenchmarkFunc_2(Hashmap<int, int> *hashmap, const thrust::device_vector<int> &d_keys, const thrust::device_vector<int> &d_values)
{
    // Define grid and block sizes
    int numElements = config.numElements;
    int blockSize = config.threads;
    int gridSize = config.blocks;

    std::cout << std::setw(25) << "Threads per block:" << blockSize << "\n";
    std::cout << std::setw(25) << "Number of blocks:" << gridSize << "\n";
    std::cout << std::setw(25) << "Total number of threads:" << blockSize * gridSize << "\n";

    testIntInsert_2<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(d_keys.data()), thrust::raw_pointer_cast(d_values.data()), numElements, hashmap);
    cudaDeviceSynchronize();
}

void insertionBenchmarkCGFunc(Hashmap<int, int> *hashmap, const thrust::device_vector<int> &d_keys, const thrust::device_vector<int> &d_values)
{
    // Define default grid and block sizes
    int numElements = config.numElements;
    int blockSize = config.threads;
    int gridSize = (numElements * config.cg_size + blockSize - 1) / blockSize;

    assert(blockSize >= 4); // make sure there are at least 4 threads for the cooperative insert

    std::cout << std::setw(25) << "Threads per block:" << blockSize << "\n";
    std::cout << std::setw(25) << "Number of blocks:" << gridSize << "\n";
    std::cout << std::setw(25) << "Total number of threads:" << blockSize * gridSize << "\n";

    testIntInsertCG<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(d_keys.data()), thrust::raw_pointer_cast(d_values.data()), numElements, hashmap, config.cg_size);
    cudaDeviceSynchronize();
}

template <typename Key, typename Value>
void getValues(const Hashmap<int, int> *hashmap, const thrust::device_vector<Key> &keys, thrust::device_vector<Value> &results)
{
    int blockSize = 256;
    int gridSize = (keys.size() + blockSize - 1) / blockSize;
    findKernel<<<gridSize, blockSize>>>(hashmap, thrust::raw_pointer_cast(keys.data()), thrust::raw_pointer_cast(results.data()), keys.size());
}

void insertionBenchmarkCGFunc_2(Hashmap<int, int> *hashmap, const thrust::device_vector<int> &d_keys, const thrust::device_vector<int> &d_values)
{
    // Define default grid and block sizes
    int numElements = config.numElements;
    int blockSize = config.threads;
    int gridSize = config.blocks;

    assert(blockSize >= 4); // make sure there are at least 4 threads for the cooperative insert

    std::cout << std::setw(25) << "Threads per block:" << blockSize << "\n";
    std::cout << std::setw(25) << "Number of blocks:" << gridSize << "\n";
    std::cout << std::setw(25) << "Total number of threads:" << blockSize * gridSize << "\n";

    testIntInsertCG_2<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(d_keys.data()), thrust::raw_pointer_cast(d_values.data()), numElements, hashmap, config.cg_size);
    cudaDeviceSynchronize();
}

void searchBenchMarkFunc(const Hashmap<int, int> *hashmap, const thrust::device_vector<int> &d_keys, thrust::device_vector<int> &d_results)
{
    getValues<int, int>(hashmap, d_keys, d_results);
    cudaDeviceSynchronize();
}

void setupActions()
{
    actions['d'] = [](const char *)
    { config.defaultInsert = true; std::cout << "Default insert\n"; };
    actions['c'] = [](const char *)
    { config.cooperativeGroupsInsert = true; std::cout << "Cooperative groups insert\n"; };
    actions['s'] = [](const char *)
    { config.defaultSearch = true; std::cout << "Default search\n"; };
    actions['n'] = [](const char *arg)
    { config.numElements = std::stoul(arg); };
    actions['l'] = [](const char *arg)
    { config.load = std::stof(arg); };
    actions['t'] = [](const char *arg)
    { config.threads = std::stoul(arg); };
    actions['b'] = [](const char *arg)
    { config.blocks = std::stoul(arg); };
    actions['g'] = [](const char *arg)
    { config.device = std::stoul(arg); };
}

int main(int argc, char **argv)
{
    // Set CUDA Device - Ensure this is valid for your system
    cudaError_t cudaStatus = cudaSetDevice(config.device);
    if (cudaStatus != cudaSuccess)
    {
        std::cerr << "cudaSetDevice failed!" << std::endl;
        return 1;
    }

    setupActions();

    int opt;
    while ((opt = getopt(argc, argv, "dcsn:l:t:b:g:")) != -1)
    {
        auto action = actions.find(opt);
        if (action != actions.end())
            action->second(optarg);
        else
            std::cerr << "Unknown option: " << static_cast<char>(opt) << '\n';
    }

    // Define hashmap capacity
    std::size_t const capacity = std::ceil(config.numElements / config.load);

    std::cout << std::left; // Align text to the left
    std::cout << std::setw(25) << "Using device:" << config.device << "\n";
    std::cout << std::setw(25) << "Elements to insert:" << config.numElements << "\n";
    std::cout << std::setw(25) << "Load factor:" << config.load << "\n";
    std::cout << std::setw(25) << "Capacity:" << capacity << "\n";

    // Initialize data
    thrust::host_vector<int> h_keys(config.numElements), h_values(config.numElements);

    // Fill keys and values with test data
    initializeData(h_keys, config.numElements);
    initializeData(h_values, config.numElements);

    // Copy data from host to device
    thrust::device_vector<int> d_keys = h_keys;
    thrust::device_vector<int> d_values = h_values;

    // Create and initialize hashmap
    Hashmap<int, int> *hashmap; // Assuming constructor initializes the GPU memory
    cudaMallocManaged(&hashmap, sizeof(Hashmap<int, int>));
    new (hashmap) Hashmap<int, int>(capacity);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error malloc memcpy: " << cudaGetErrorString(err) << std::endl;
        // handle error
    }

    if (config.defaultInsert)
    {
        hashmap->initialize();
        benchmarkKernel([&]()
                        { insertionBenchmarkFunc_2(hashmap, d_keys, d_values); },
                        "non-CG Insertion");
    }

    if (config.cooperativeGroupsInsert)
    {
        hashmap->initialize();
        benchmarkKernel([&]()
                        { std::cout << std::setw(25) << "Cooperative group size:" << config.cg_size << "\n";
                          insertionBenchmarkCGFunc_2(hashmap, d_keys, d_values); },
                        "Insertion CG");
    }

    if (config.defaultSearch)
    {
        thrust::device_vector<int> d_results(d_keys.size());
        benchmarkKernel([&]()
                        { searchBenchMarkFunc(hashmap, d_keys, d_results); },
                        "Search");

        if (!checkResults(d_results, h_values))
        {
            // hashmap->printAllBucketValues();
        }
    }

    // Cleanup
    hashmap->~Hashmap();
    cudaFree(hashmap);

    return 0;
}
