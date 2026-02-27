#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cstdlib>
#include <tensor/tensor.h>

int main() {
    Forge::Tensor tensor {{3, 4}};
    tensor = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
    // std::cout << tensor<<"\n";
    const auto reshaped {tensor.reshape(12)};
    // std::cout<<reshaped<<"\n";

    auto tensor2 {tensor.clone()};
    tensor2[1] = {6,6,6,6};
    std::cout<<tensor<<"\n\n"<<tensor2<<"\n";

    return 0;
}
