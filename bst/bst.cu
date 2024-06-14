#include <iostream>
#include <memory>
#include <chrono>
#include <random>
#include <vector>
#include <queue>
#include <climits>
#include <numeric>
#include <algorithm>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#define NULL_VAL INT_MAX
using namespace std;

__global__ void searchKernel(int *tree, int tree_size, int *keys, bool *results, int num_keys)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < num_keys)
    {
        int key = keys[index];
        bool found = false;
        int node_index = 0;
        while (node_index < tree_size && tree[node_index] != INT_MAX)
        {
            if (tree[node_index] == key)
            {
                found = true;
                break;
            }
            if (key < tree[node_index])
            {
                node_index = 2 * node_index + 1;
            }
            else
            {
                node_index = 2 * node_index + 2;
            }
        }
        results[index] = found;
    }
}

class BSTNode;
using BSTNodePtr = BSTNode *;

class BSTNode
{
public:
    int key;
    int height;
    BSTNodePtr left;
    BSTNodePtr right;

    BSTNode(int value) : key(value), height(1), left(nullptr), right(nullptr) {}
};

class BST
{
private:
    BSTNodePtr root;

    int height(BSTNodePtr node)
    {
        return (node) ? node->height : 0;
    }

    int getBalance(BSTNodePtr &node)
    {
        return (node) ? height(node->left) - height(node->right) : 0;
    }

    BSTNodePtr rotateRight(BSTNodePtr y)
    {
        BSTNodePtr x = y->left;
        BSTNodePtr T2 = x->right;

        // Perform rotation
        x->right = y;
        y->left = T2;

        // Update heights
        y->height = max(height(y->left), height(y->right)) + 1;
        x->height = max(height(x->left), height(x->right)) + 1;

        return x;
    }

    BSTNodePtr rotateLeft(BSTNodePtr x)
    {
        BSTNodePtr y = x->right;
        BSTNodePtr T2 = y->left;

        // Perform rotation
        y->left = x;
        x->right = T2;

        // Update heights
        x->height = max(height(x->left), height(x->right)) + 1;
        y->height = max(height(y->left), height(y->right)) + 1;

        return y;
    }

    BSTNodePtr insert(BSTNodePtr root, int key)
    {
        if (!root)
            return new BSTNode(key);

        if (key < root->key)
            root->left = insert(root->left, key);
        else if (key > root->key)
            root->right = insert(root->right, key);
        else
            return root; // Duplicate keys not allowed

        root->height = max(height(root->left), height(root->right)) + 1;
        int balance = getBalance(root);

        // Left Left Case
        if (balance > 1 && root->left && key < root->left->key)
            return rotateRight(root);
        // Right Right Case
        if (balance < -1 && root->right && key > root->right->key)
            return rotateLeft(root);
        // Left Right Case
        if (balance > 1 && root->left && key > root->left->key)
        {
            root->left = rotateLeft(root->left);
            return rotateRight(root);
        }
        // Right Left Case
        if (balance < -1 && root->right && key < root->right->key)
        {
            root->right = rotateRight(root->right);
            return rotateLeft(root);
        }

        return root;
    }

    bool search(BSTNodePtr &root, int key)
    {
        if (root == nullptr)
            return false;

        if (key == root->key)
            return true;
        else if (key < root->key)
            return search(root->left, key);
        else
            return search(root->right, key);
    }

public:
    BST() : root(nullptr) {}

    void insert(int key)
    {
        root = insert(root, key);
    }
    bool search(int key)
    {
        return search(root, key);
    }

    vector<int> covertToArray()
    {
        vector<int> treeArray;
        if (!root)
            return treeArray;

        queue<pair<BSTNodePtr, size_t>> nodeQueue;
        nodeQueue.push({root, 0});
        while (!nodeQueue.empty())
        {
            auto [node, index] = nodeQueue.front();
            nodeQueue.pop();

            size_t newSize = max(treeArray.size(), index + 1);
            if (node->left)
                newSize = max(newSize, 2 * index + 2);
            if (node->right)
                newSize = max(newSize, 2 * index + 3);

            if (newSize > treeArray.size())
                treeArray.resize(newSize, NULL_VAL);

            // printf("New size: %lu ; Index: %lu ; Tree size: %lu\n", newSize, index, treeArray.size());
            treeArray[index] = node->key;

            if (node->left)
                nodeQueue.push({node->left, 2 * index + 1});
            if (node->right)
                nodeQueue.push({node->right, 2 * index + 2});
        }
        return treeArray;
    }
};

void batchSearch_thrust(const std::vector<int> &tree_arr, const std::vector<int> &search_keys, std::vector<bool> &results)
{
    int num_keys = search_keys.size();

    // Transfer data to device
    thrust::device_vector<int> d_tree = tree_arr;
    thrust::device_vector<int> d_keys = search_keys;
    thrust::device_vector<bool> d_results(num_keys);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_keys + threadsPerBlock - 1) / threadsPerBlock;
    searchKernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(d_tree.data()),
        d_tree.size(),
        thrust::raw_pointer_cast(d_keys.data()),
        thrust::raw_pointer_cast(d_results.data()),
        num_keys);

    // Transfer results back to host
    thrust::copy(d_results.begin(), d_results.end(), results.begin());
}

void batchSearch(const std::vector<int> &tree_arr, const std::vector<int> &search_keys, std::vector<char> &results)
{
    int num_keys = search_keys.size();

    int *d_tree, *d_keys;
    bool *d_results;
    cudaMalloc(&d_tree, tree_arr.size() * sizeof(int));
    cudaMalloc(&d_keys, num_keys * sizeof(int));
    cudaMalloc(&d_results, num_keys * sizeof(bool));

    cudaMemcpy(d_tree, tree_arr.data(), tree_arr.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_keys, search_keys.data(), num_keys * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_keys + threadsPerBlock - 1) / threadsPerBlock;
    searchKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_tree,
        tree_arr.size(),
        d_keys,
        d_results,
        num_keys);

    cudaDeviceSynchronize();

    cudaMemcpy(results.data(), d_results, num_keys * sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(d_tree);
    cudaFree(d_keys);
    cudaFree(d_results);
}
int main()
{
    BST bst;
    const int N = 1e4;   // Number of nodes
    vector<int> data(N); // Keys to be inserted
    iota(data.begin(), data.end(), 0);
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    shuffle(data.begin(), data.end(), default_random_engine(seed));

    for (int key : data)
    {
        bst.insert(key);
    }

    // Search benchmark
    int M = 1e5; // Number of searches
    vector<int> search_keys(M);
    iota(search_keys.begin(), search_keys.end(), 0);
    shuffle(search_keys.begin(), search_keys.end(), default_random_engine(seed));

    auto start = chrono::high_resolution_clock::now();
    for (int key : search_keys)
        bst.search(key);

    auto end = chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    printf("Time taken to search %d keys: %f milliseconds\n", M, duration.count() * 1000);

    vector<int> tree_arr = bst.covertToArray();
    vector<char> results(search_keys.size(), false);

    // Batch search benchmark
    start = chrono::high_resolution_clock::now();
    batchSearch(tree_arr, search_keys, results);
    end = chrono::high_resolution_clock::now();
    duration = end - start;
    printf("Time taken to batch search %d keys: %f milliseconds\n", M, duration.count() * 1000);

    // Thrust batch search benchmark
    vector<bool> thrust_results(search_keys.size(), false);
    start = chrono::high_resolution_clock::now();
    batchSearch_thrust(tree_arr, search_keys, thrust_results);
    end = chrono::high_resolution_clock::now();
    duration = end - start;
    printf("Time taken to thrust batch search %d keys: %f milliseconds\n", M, duration.count() * 1000);

    // Check results
    for (int i = 0; i < M; i++)
    {
        if (thrust_results[i] != bst.search(search_keys[i]))
        {
            printf("Error at index %d\n", i);
            break;
        }
    }
}