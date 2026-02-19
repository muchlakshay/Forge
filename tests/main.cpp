#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cstdlib>
#include <tensor/tensor.h>



int main() {
    Forge::Tensor tensor {{3, 4}};
    tensor = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
    std::cout << tensor<<"\n";
    const auto reshaped {tensor.reshape(1, 1, 1, 2, 6)};
    std::cout<<reshaped<<"\n";

    return 0;
}
