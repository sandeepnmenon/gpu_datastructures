#include <cuco/static_map.cuh>
#include <thrust/device_vector.h>
#include <thrust/equal.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <iostream>
#include <cmath>
#include <chrono>

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <num_keys> <load_factor>" << std::endl;
        return 1;
    }

    std::size_t num_keys = std::stoull(argv[1]);
    double load_factor = std::stod(argv[2]);

    using Key = int;
    using Value = int;

    Key constexpr empty_key_sentinel = -1;
    Value constexpr empty_value_sentinel = -1;

    std::size_t const capacity = std::ceil(num_keys / load_factor);

    cuco::static_map<Key, Value> map{
        capacity, cuco::empty_key{empty_key_sentinel}, cuco::empty_value{empty_value_sentinel}};

    thrust::device_vector<Key> insert_keys(num_keys);
    thrust::sequence(insert_keys.begin(), insert_keys.end(), 0);
    thrust::device_vector<Value> insert_values(num_keys);
    thrust::sequence(insert_values.begin(), insert_values.end(), 0);
    auto zipped = thrust::make_zip_iterator(thrust::make_tuple(insert_keys.begin(), insert_values.begin()));

    auto start = std::chrono::high_resolution_clock::now();
    map.insert(zipped, zipped + insert_keys.size());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> insert_duration = end - start;

    thrust::device_vector<Value> found_values(num_keys);
    start = std::chrono::high_resolution_clock::now();
    map.find(insert_keys.begin(), insert_keys.end(), found_values.begin());
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> find_duration = end - start;

    bool const all_values_match = thrust::equal(found_values.begin(), found_values.end(), insert_values.begin());

    double insert_time = insert_duration.count();
    double find_time = find_duration.count();
    double insert_throughput = num_keys / insert_time;
    double find_throughput = num_keys / find_time;

    std::cout << "Insert Time: " << insert_time << " seconds\n";
    std::cout << "Find Time: " << find_time << " seconds\n";
    std::cout << "Insert Throughput: " << insert_throughput << " keys/second\n";
    std::cout << "Find Throughput: " << find_throughput << " keys/second\n";

    if (all_values_match)
    {
        std::cout << "Success! Found all values.\n";
    }
    else
    {
        std::cout << "Error! Some values were not found correctly.\n";
    }

    return 0;
}
