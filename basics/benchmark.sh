#!/bin/bash

make clean
make

# Define arrays for benchmarks
num_elements=(10000 20000 30000)
load_factors=(0.2 0.4 0.6 0.8 1.0)
threads_per_block=(32 64 128)
num_blocks=(500 1000 1500)

# Test run. Comment out for real benchmark
num_elements=(10000 20000 )
load_factors=(0.2 0.4)
threads_per_block=(32 64 )
num_blocks=(500 1000 )


# Output file
output_file="benchmark_results.csv"

# Write CSV header
echo "NumElements,LoadFactor,ThreadsPerBlock,NumBlocks,DefaultInsertTime,CGInsertTime,DefaultSearchTime,CGSearchTime,DefaultSearchSuccess,CGSearchSuccess" > "$output_file"

# Function to run the program and extract timings
run_and_extract_time() {
    local n=$1
    local l=$2
    local t=$3
    local b=$4
    local output

    output=$(./driver -iIsS -n "$n" -l "$l" -t "$t" -b "$b" 2>&1)

    # Use grep and awk to extract timing values
    local default_insert_time=$(echo "$output" | grep "^non-CG Insertion time:" | awk '{print $4}')
    local cg_insert_time=$(echo "$output" | grep "^CG Insertion time:" | awk '{print $4}')
    local default_search_time=$(echo "$output" | grep "^non-CG Search time:" | awk '{print $4}')
    local cg_search_time=$(echo "$output" | grep "^CG Search time:" | awk '{print $4}')

    local default_search_success=$(echo "$output" | grep -A 1 "^non-CG Search time:" | grep "Success" | wc -l)
    local cg_search_success=$(echo "$output" | grep -A 1 "^CG Search time:" | grep "Success" | wc -l)

    # Write to CSV
    echo "$n,$l,$t,$b,$default_insert_time,$cg_insert_time,$default_search_time,$cg_search_time,$default_search_success,$cg_search_success" >> "$output_file"
}

# Iterate over all combinations of parameters
for n in "${num_elements[@]}"; do
    for l in "${load_factors[@]}"; do
        for t in "${threads_per_block[@]}"; do
            for b in "${num_blocks[@]}"; do
                echo "Running with NumElements=$n, LoadFactor=$l, ThreadsPerBlock=$t, NumBlocks=$b"
                run_and_extract_time "$n" "$l" "$t" "$b"
            done
        done
    done
done

echo "Benchmark results written to $output_file"
