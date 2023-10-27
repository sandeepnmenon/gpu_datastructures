#include <iostream>
#include <memory>
#include <chrono>
#include <random>
#include <vector>
#include <numeric>
#include <algorithm>
using namespace std;

class BSTNode;
using BSTNodePtr = unique_ptr<BSTNode>;

class BSTNode
{
public:
    int key;
    unique_ptr<BSTNode> left;
    unique_ptr<BSTNode> right;

    BSTNode(int value) : key(value), left(nullptr), right(nullptr) {}
};

class BST
{
private:
    BSTNodePtr root;

    BSTNodePtr insert(BSTNodePtr root, int key)
    {
        if (root == nullptr)
            return make_unique<BSTNode>(key);

        if (key < root->key)
            root->left = insert(move(root->left), key);
        else if (key > root->key)
            root->right = insert(move(root->right), key);

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
        root = insert(move(root), key);
    }
    bool search(int key)
    {
        return search(root, key);
    }
};

int main()
{
    BST bst;
    const int N = 1e5;   // Number of nodes
    vector<int> data(N); // Keys to be inserted
    iota(data.begin(), data.end(), 0);
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    shuffle(data.begin(), data.end(), default_random_engine(seed));

    for (int key : data)
        bst.insert(key);

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
}