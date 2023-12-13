#!/bin/bash

# Compile the program
make clean
make

# Define arrays for benchmarks
num_keys_arr=(16384 131072 1048576 16777216 67108864)
load_factors_arr=(0.2 0.4 0.6 0.8 1.0)

# Output file
output_file="cucollections_benchmark_results.csv"

# Write CSV header
echo "NumKeys,LoadFactor,InsertTime,FindTime,InsertThroughput,FindThroughput" > "$output_file"

# Function to run the program and extract timings
run_and_extract_time() {
    local num_keys=$1
    local load_factor=$2
    local output

    output=$(./static_map_example "$num_keys" "$load_factor" 2>&1)

    # Use grep and awk to extract timing values
    local insert_time=$(echo "$output" | grep "Insert Time:" | awk '{print $3}')
    local find_time=$(echo "$output" | grep "Find Time:" | awk '{print $3}')
    local insert_throughput=$(echo "$output" | grep "Insert Throughput:" | awk '{print $3}')
    local find_throughput=$(echo "$output" | grep "Find Throughput:" | awk '{print $3}')

    # Write to CSV
    echo "$num_keys,$load_factor,$insert_time,$find_time,$insert_throughput,$find_throughput" >> "$output_file"
}

# Iterate over all combinations of num_keys and load_factor
for num_keys in "${num_keys_arr[@]}"; do
    for load_factor in "${load_factors_arr[@]}"; do
        echo "Running with NumKeys=$num_keys, LoadFactor=$load_factor"
        run_and_extract_time "$num_keys" "$load_factor"
    done
done

echo "Benchmark results written to $output_file"
