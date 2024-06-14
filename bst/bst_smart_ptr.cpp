#include <iostream>
#include <memory>
#include <chrono>
#include <random>
#include <vector>
#include <queue>
#include <climits>
#include <numeric>
#include <algorithm>
#define NULL_VAL INT_MAX
using namespace std;

class BSTNode;
using BSTNodePtr = unique_ptr<BSTNode>;

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

    int height(BSTNodePtr &node)
    {
        return (node) ? node->height : 0;
    }

    int getBalance(BSTNodePtr &node)
    {
        return (node) ? height(node->left) - height(node->right) : 0;
    }

    BSTNodePtr rotateRight(BSTNodePtr y)
    {
        BSTNodePtr x = std::move(y->left);
        y->left = std::move(x->right);
        x->right = std::move(y);
        x->right->height = std::max(height(x->right->left), height(x->right->right)) + 1;
        x->height = std::max(height(x->left), height(x->right)) + 1;
        return x;
    }

    BSTNodePtr rotateLeft(BSTNodePtr x)
    {
        BSTNodePtr y = std::move(x->right);
        x->right = std::move(y->left);
        y->left = std::move(x);
        y->left->height = std::max(height(y->left->left), height(y->left->right)) + 1;
        y->height = std::max(height(y->left), height(y->right)) + 1;
        return y;
    }

    BSTNodePtr insert(BSTNodePtr root, int key)
    {
        if (!root)
            return std::make_unique<BSTNode>(key);

        if (key < root->key)
            root->left = insert(std::move(root->left), key);
        else if (key > root->key)
            root->right = insert(std::move(root->right), key);
        else
            return root; // Duplicate keys not allowed

        root->height = std::max(height(root->left), height(root->right)) + 1;
        int balance = getBalance(root);

        // Left Left Case
        if (balance > 1 && root->left && key < root->left->key)
            return rotateRight(std::move(root));
        // Right Right Case
        if (balance < -1 && root->right && key > root->right->key)
            return rotateLeft(std::move(root));
        // Left Right Case
        if (balance > 1 && root->left && key > root->left->key)
        {
            root->left = rotateLeft(std::move(root->left));
            return rotateRight(std::move(root));
        }
        // Right Left Case
        if (balance < -1 && root->right && key < root->right->key)
        {
            root->right = rotateRight(std::move(root->right));
            return rotateLeft(std::move(root));
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
        root = insert(std::move(root), key);
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

        std::queue<std::pair<BSTNode *, std::size_t>> nodeQueue;
        nodeQueue.push({root.get(), 0});
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
                nodeQueue.push({node->left.get(), 2 * index + 1});
            if (node->right)
                nodeQueue.push({node->right.get(), 2 * index + 2});
        }
        return treeArray;
    }
};

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

    // Convert to array
    vector<int> treeArray = bst.covertToArray();
}