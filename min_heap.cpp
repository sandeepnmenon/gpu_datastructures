#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include <random>
#include <algorithm>

using namespace std;

class MinHeap {
private:
    vector<int> heap;

    // Helper functions
    void heapifyUp(int index) {
        int parentIndex = (index - 1) / 2;
        while (index > 0 && heap[index] < heap[parentIndex]) {
            swap(heap[index], heap[parentIndex]);
            index = parentIndex;
            parentIndex = (index - 1) / 2;
        }
    }

    void heapifyDown(int index) {
        int leftChild = 2 * index + 1;
        int rightChild = 2 * index + 2;
        int smallest = index;

        if (leftChild < heap.size() && heap[leftChild] < heap[smallest]) {
            smallest = leftChild;
        }

        if (rightChild < heap.size() && heap[rightChild] < heap[smallest]) {
            smallest = rightChild;
        }

        if (smallest != index) {
            swap(heap[index], heap[smallest]);
            heapifyDown(smallest);
        }
    }

public:
    // Constructor
    MinHeap() {}

    // Insert a value into the min heap
    void insert(int value) {
        heap.push_back(value);
        heapifyUp(heap.size() - 1);
    }

    // Extract the minimum value from the min heap
    int extractMin() {
        if (heap.empty()) {
            throw runtime_error("Heap is empty");
        }

        int min = heap[0];
        heap[0] = heap.back();
        heap.pop_back();
        heapifyDown(0);

        return min;
    }

    // Get the size of the heap
    size_t size() {
        return heap.size();
    }

    // Search for a specific value in the heap
    bool search(int value) {
        for (int i = 0; i < heap.size(); i++) {
            if (heap[i] == value) {
                return true;
            }
        }
        return false;
    }

};

int main() {

    MinHeap minHeap;

    const int N = 1e5;   // Number of nodes
    vector<int> data(N); // Keys to be inserted
    iota(data.begin(), data.end(), 0);
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    shuffle(data.begin(), data.end(), default_random_engine(seed));

    for (int key : data) {
        minHeap.insert(key);
    }

    // Search benchmark
    int M = 1e5; // Number of searches
    vector<int> search_keys(M);
    iota(search_keys.begin(), search_keys.end(), 0);
    shuffle(search_keys.begin(), search_keys.end(), default_random_engine(seed));

    std::chrono::duration<double> duration;

    auto start = chrono::high_resolution_clock::now();
    for (int key : search_keys)
        minHeap.search(key);

    auto end = chrono::high_resolution_clock::now();
    duration = end - start;
    printf("Time taken to search %d keys: %f milliseconds\n", M, duration.count() * 1000);


    // Extract benchmark
    start = chrono::high_resolution_clock::now();
    while (minHeap.size() > 0) {
        minHeap.extractMin();
        //cout << minHeap.extractMin() << " \n";
    }
    end = chrono::high_resolution_clock::now();
    duration = end - start;
    printf("Time taken to extract %d keys by their priority: %f milliseconds\n", N, duration.count() * 1000);

    return 0;
}