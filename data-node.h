#ifndef DATA_NODE_H
#define DATA_NODE_H
// Allows for insertion of any numeric type T; only compiles if T is numeric (see SFINAE for details)
template
<
        typename T,
        // std::is_arithmetic, std::enable_if: C++11 feature; defined in <type_traits>
        // std::enable_if<condition, T>::type returns T if condition is true, else no such member
        // std::is_arithmetic<T>::value returns true if T is an arithmetic type, false otherwise
        typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type
>
struct DataNode
{
	T search_key;
	T priority;
};

#endif
