#include <iostream>
#include "memory_pool_cpu.h"

int main() {
    try {
        MemoryPoolCPU memory_pool{};
        int size {10};
        for (int i = 0; i < 3; i++) {
            auto* mem {static_cast<int *>(memory_pool.getMem(10*1024))};
            memory_pool.returnMem(mem);
        }
    }catch(std::exception& e) {std::cout<<e.what();}



    return 0;
}
