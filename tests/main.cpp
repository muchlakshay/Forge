#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cstdlib>
#include <tensor/tensor.h>



int main() {
    Forge::Tensor tensor {{3, 4}};
    tensor = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
    std::cout << tensor;
    std::cout<<tensor.reshape(6, 2);
    return 0;
}
