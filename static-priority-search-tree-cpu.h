#ifndef PRIORITY_SEARCH_TREE_CPU_H
#define PRIORITY_SEARCH_TREE_CPU_H

class StaticPrioritySearchTreeCPU
{
	public:
		StaticPrioritySearchTreeCPU(int *search_keys, int num_search_keys, int *priorities, int num_priorities);
		~StaticPrioritySearchTreeCPU();
		// Returns a list of satisfying nodes; maybe the array index of each satisfying node?
		ThreeSidedSearch();
		TwoSidedLeftSearch();
		TwoSidedRightSearch();

	private:
		// Want unique copies of each tree, so no assignment or copying allowed
		StaticPrioritySearchTreeCPU& operator=(StaticPrioritySearchTreeCPU &tree);	// assignment operator
		StaticPrioritySearchTreeCPU(PrioritySearchTreeCPU &tree);	// copy constructor

		class Node
		{
			int search_key;
			int priority;
			int median_search_key;
			bool is_leaf;
		};
}

#endif
