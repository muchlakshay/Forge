#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cstdlib>
#include <source_location>

#include "../core/tensor/Storage/memory_pool_cpu.h"
#include "Dispatcher/kernel_base.h"
// class A {
// public:
//     constexpr auto get_str() const {
//         return std::source_location::current().function_name();
//     }
// };

class Add : public Kernel{
    public:
    Add() : Kernel{ctti::type_id<Add>()} {}
};

int main() {

    return 0;
}
