#ifndef PRIORITY_SEARCH_TREE_CPU_H
#define PRIORITY_SEARCH_TREE_CPU_H

// Allows for insertion of any numeric type T; only compiles if T is numeric (see SFINAE for details)
template
<
	typename T,
	// std::is_arithmetic, std::enable_if: C++11 feature; defined in <type_traits>
	// std::enable_if<condition, T>::type returns T if condition is true, else no such member
	// std::is_arithmetic<T>::value returns true if T is an arithmetic type, false otherwise
	typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type
>
class StaticPrioritySearchTreeCPU
{
	public:
		StaticPrioritySearchTreeCPU(T *search_keys, int num_search_keys, T *priorities, int num_priorities);
		virtual ~StaticPrioritySearchTreeCPU();
		// Returns a list of satisfying nodes; maybe the array index of each satisfying node?
		ThreeSidedSearch();
		TwoSidedLeftSearch();
		TwoSidedRightSearch();

	private:
		// Want unique copies of each tree, so no assignment or copying allowed
		StaticPrioritySearchTreeCPU& operator=(StaticPrioritySearchTreeCPU &tree);	// assignment operator
		StaticPrioritySearchTreeCPU(PrioritySearchTreeCPU &tree);	// copy constructor

		struct Node
		{
			T search_key;
			T priority;
			T median_search_key;
			// also 1 byte, like char, but doesn't auto-convert non-zero values other than 1 to 1
			char codes;
		};

		// Bitcodes used to indicate presence of left/right children (and potentially other values as necessary) to save space, as bool actually takes up 1 byte, same as a char
		static enum Bitcodes
		{
			HAS_LEFT_CHILD = 0x2,
			HAS_RIGHT_CHILD = 0x1
		};
}

#endif
