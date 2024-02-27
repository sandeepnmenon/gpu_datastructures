#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include <random>
#include <algorithm>
#include <unordered_map>

using namespace std;

class MinHeap {
private:
    vector<int> heap;
    unordered_map<int, int> element_to_index;  // Hash table to store element indices
    size_t heapSize;  // Variable to store the size of the heap

    // Helper functions
    void heapifyUp(int index) {
        int parentIndex = (index - 1) / 2;
        while (index > 0 && heap[index] < heap[parentIndex]) {
            swap(heap[index], heap[parentIndex]);
            element_to_index[heap[index]] = index;  // Update the index in the hash table
            element_to_index[heap[parentIndex]] = parentIndex;
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
            element_to_index[heap[index]] = index;  // Update the index in the hash table
            element_to_index[heap[smallest]] = smallest;
            heapifyDown(smallest);
        }
    }

public:
    // Constructor
    MinHeap() : heapSize(0) {}

    // Insert a value into the min heap
    void insert(int value) {
        heap.push_back(value);
        element_to_index[value] = heapSize;  // Store the index in the hash table
        heapifyUp(heapSize);
        heapSize++;
    }

    // Extract the minimum value from the min heap
    int extractMin() {
        if (heapSize == 0) {
            throw runtime_error("Heap is empty");
        }

        int min = heap[0];
        int lastElement = heap.back();
        heap[0] = lastElement;
        element_to_index[lastElement] = 0;  // Update the index in the hash table
        heap.pop_back();
        heapifyDown(0);
        element_to_index.erase(min);  // Remove the index from the hash table
        heapSize--;

        return min;
    }

    // Search for a specific value in the heap
    bool search(int value) {
        return element_to_index.find(value) != element_to_index.end();
    }

    // Delete a specific value from the heap
    bool deleteElement(int value) {
        if (element_to_index.find(value) == element_to_index.end()) {
            return false;  // Value not found in the heap
        }

        int index = element_to_index[value];
        int lastElement = heap.back();

        // Swap the element to be deleted with the last element
        heap[index] = lastElement;
        element_to_index[lastElement] = index;

        // Update the size of the heap and remove the index from the hash table
        heap.pop_back();
        heapSize--;
        element_to_index.erase(value);

        // Perform heapifyUp and heapifyDown to maintain the heap property
        heapifyUp(index);
        heapifyDown(index);

        return true;
    }

    // Get the size of the heap
    size_t size() const {
        return heapSize;
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

    // Delete benchmark
    start = chrono::high_resolution_clock::now();
    for (int key : search_keys)
        minHeap.deleteElement(key);

    end = chrono::high_resolution_clock::now();
    duration = end - start;
    printf("Time taken to delete %d keys: %f milliseconds\n", M, duration.count() * 1000);

    for (int key : data) {
        minHeap.insert(key);
    }

    // Extract benchmark
    start = chrono::high_resolution_clock::now();
    while (minHeap.size() > 0) {
        minHeap.extractMin();
    }
    end = chrono::high_resolution_clock::now();
    duration = end - start;
    printf("Time taken to extract %d keys by their priority: %f milliseconds\n", N, duration.count() * 1000);

    return 0;
}
