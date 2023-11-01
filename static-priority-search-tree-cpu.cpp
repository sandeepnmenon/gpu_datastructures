#include <algorithm>	// To use sort()
#include <cmath>		// To use ceil() and log2()
#include <iostream>
#include <type_traits>	// To filter out non-numeric types of T
#include "data-node.h"
#include "static-priority-search-tree-cpu.h"

// Allows for insertion of any numeric type T; only compiles if T is numeric (see SFINAE for details)
template
<
        typename T,
        // std::is_arithmetic, std::enable_if: C++11 feature; defined in <type_traits>
        // std::enable_if<condition, T>::type returns T if condition is true, else no such member
        // std::is_arithmetic<T>::value returns true if T is an arithmetic type, false otherwise
        typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type
>
StaticPrioritySearchTreeCPU<T>::StaticPrioritySearchTreeCPU(DataNode<T> *data_arr, int num_elems)
{
	if (num_elems == 0)
	{
		root = nullptr;
		return;
	}

	DataNode<T>** search_key_ptr_arr = new DataNode<T>*[num_elems]();
	DataNode<T>** priority_ptr_arr = new DataNode<T>*[num_elems]();

	for (int i = 0; i < num_elems; i++)
		search_key_ptr_arr[i] = priority_ptr_arr[i] = data_arr + i;

	// Sort search key pointer array in ascending order; in-place sort
	std::sort(search_key_ptr_arr, search_key_ptr_arr + num_elems,
				[](DataNode<T> *&node_ptr_1, DataNode<T> *&node_ptr_2)
				{
					return node_ptr_1->search_key < node_ptr_2->search_key;
				});

	// Sort priority pointer array in descending order; in-place sort
	std::sort(priority_ptr_arr, priority_ptr_arr + num_elems,
				[](DataNode<T> *&node_ptr_1, DataNode<T> *&node_ptr_2)
				{
					return node_ptr_1->priority > node_ptr_2->priority;
				});


	// Minimum number of array slots necessary to construct tree given it is fully balanced by construction and given the unknown placement of nodes in the partially empty last row
	// Number of slots in container array is 2^ceil(lg(num_elem + 1)) - 1
	// Use of () after new and new[] causes value-initialisation (to 0) starting in C++03; needed for any nodes that technically contain no data
	root = new TreeNode[(1 << std::ceil(std::log2(num_elems + 1))) - 1]();

	populateTreeRecur(root, 0, search_key_ptr_arr, 1, num_elems, priority_ptr_arr);

	delete[] search_key_ptr_arr;
}

template
<
        typename T,
        typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type
>
StaticPrioritySearchTreeCPU<T>::~StaticPrioritySearchTreeCPU()
{
	delete[] root;
}

// Value-initialisation is more efficient with member initialiser lists, as they are not default-initialised before being overriden
template
<
        typename T,
        typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type
>
StaticPrioritySearchTreeCPU<T>::TreeNode::TreeNode()
	// When no members are explicitly initialised, default-initialisation of non-class variables with automatic or dynamic storage duration produces objects with indeterminate values
	: search_key(0),
	priority(0),
	median_search_key(0),
	code(0)
{}

template
<
        typename T,
        typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type
>
StaticPrioritySearchTreeCPU<T>::TreeNode::TreeNode(DataNode<T> &source_data, T median_search_key)
	// When other subobjects are explicitly initialised, those that are not are implicit initialised in the same way as objects with static storage duration, i.e. with 0 or nullptr (stated in 6.7.8 (19) of the C++ standard)
	: search_key(source_data.search_key),
	priority(source_data.priority),
	median_search_key(median_search_key)
{}

template
<
        typename T,
        typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type
>
TreeNode& StaticPrioritySearchTreeCPU<T>::TreeNode::operator=(TreeNode &source)
{
	if (this == &source)
		return *this;	// If the two addresses match, it's the same object

	search_key = source.search_key;
	priority = source.priority;
	median_search_key = source.median_search_key;
	code = source.code;

	return *this;
}

template
<
        typename T,
        typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type
>
void StaticPrioritySearchTreeCPU<T>::TreeNode::setTreeNode(DataNode<T> &source_data, T median_search_key)
{
	search_key = source_data.search_key;
	priority = source_data.priority;
	this->median_search_key = median_search_key;
}
