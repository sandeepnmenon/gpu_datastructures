#!/bin/bash

make clean
make

# Define arrays for benchmarks
num_elements=(16384 131072 1048576 16777216 67108864)
load_factors=(0.2 0.4 0.6 0.8 1.0)
threads_per_block=(32 128 256 512 1024)
num_blocks=(8 256 2048 8192 16384 65536)

# Test run. Comment out for real benchmark
# num_elements=(10000 20000 )
# load_factors=(0.2 0.4)
# threads_per_block=(32 64 )
# num_blocks=(500 1000 )


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
                total_threads=$((t * b))
                work_per_thread=$((n / total_threads))
                if [ "$total_threads" -le "$n" ] && [ "$(($n / $total_threads))" -le 2048 ]; then
                    echo "Running with NumElements=$n, LoadFactor=$l, ThreadsPerBlock=$t, NumBlocks=$b"
                    run_and_extract_time "$n" "$l" "$t" "$b"
                else
                    echo "Skipping configuration where total_threads > num_elements: NumElements=$n, ThreadsPerBlock=$t, NumBlocks=$b"
            done
        done
    done
done

echo "Benchmark results written to $output_file"
