#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>

void initializeData(thrust::host_vector<int> &array, size_t numElements);
void initializeRandomData(thrust::host_vector<int> &array, size_t numElements);
void benchmarkKernel(std::function<void()> kernelFunc, const std::string &kernelName);
bool checkResults(const thrust::device_vector<int> &d_results, const thrust::host_vector<int> &h_values);