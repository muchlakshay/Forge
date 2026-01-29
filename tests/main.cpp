#include <iostream>
#include "memory_pool_cpu.h"

int main() {
    try {
        Forge::MemoryPoolCPU pool;
        pool.free(pool.allocate(70*1024*1024));
    }catch(std::exception& e) {std::cout<<e.what();}
    return 0;
}
