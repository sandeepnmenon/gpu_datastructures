#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>

void initializeData(thrust::host_vector<int> &array, size_t numElements)
{
    thrust::sequence(array.begin(), array.end());
    thrust::default_random_engine gen;
    thrust::shuffle(array.begin(), array.end(), gen);
}

void benchmarkKernel(std::function<void()> kernelFunc, const std::string &kernelName)
{
    // Start benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch kernel
    kernelFunc();

    // End benchmark
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        // handle error
        return;
    }

    std::cout << kernelName << " time: " << milliseconds << " ms\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void checkResults(const thrust::device_vector<int> &d_results, const thrust::host_vector<int> &h_values)
{
    thrust::host_vector<int> h_results = d_results;

    bool areEqual = thrust::equal(h_results.begin(), h_results.end(), h_values.begin());
    if (areEqual)
    {
        std::cout << "Success: d_results and h_values are the same." << std::endl;
    }
    else
    {
        std::cout << "Error: d_results and h_values differ." << std::endl;
    }
}