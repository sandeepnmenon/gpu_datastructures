#include <cuco/static_map.cuh>
#include <cuda_runtime_api.h>
#include <cassert>

int main()
{
    cuco::static_map<int, int> map{1000, -1, -1};
    map.insert(42, 42);

    int value = -1;
    bool found = map.find(42, value);
    assert(found && value == 42);

    return 0;
}
