#include <iostream>
#include <chrono>
// #include <tensor/tensor.h>

#include "../Forge.h"

int main() {
    Forge::Tensor input {{3, 4}, Forge::Dtype::float32, false};
    input = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};

    const Forge::Linear linear_1 {4, 4};
    const Forge::Linear linear_2 {4, 2};
    const Forge::Linear linear_3 {2, 1};

    auto opt_1 {linear_1(input)};
    auto opt_2 {linear_2(opt_1)};
    auto opt_3 {linear_3(opt_2)};

    std::cout<<"Input:\n\n"<<input<<"\n\n";
    std::cout<<"Layer 1:\n\nWeights:\n\n"<<linear_1.weights()<<"\n\nOutput\n\n"<<opt_1<<"\n\n";
    std::cout<<"Layer 2:\n\nWeights:\n\n"<<linear_2.weights()<<"\n\nOutput\n\n"<<opt_2<<"\n\n";
    std::cout<<"Layer 3:\n\nWeights:\n\n"<<linear_3.weights()<<"\n\nOutput\n\n"<<opt_3<<"\n\n";

    opt_3.backward();

    std::cout<<"Output Grads:\n\n";
    std::cout<<"Layer 1 Output Grads:\n"<<opt_1.gradients()<<"\n\n";
    std::cout<<"Layer 2 Output Grads:\n"<<opt_2.gradients()<<"\n\n";
    std::cout<<"Layer 3 Output Grads:\n"<<opt_3.gradients()<<"\n\n";


    std::cout<<"Layer 1 Grads:\n\nWeights Grads:\n"<<linear_1.weights().gradients()<<"\n\nBias Grads:\n"<<linear_1.bias().gradients()<<"\n\n";
    std::cout<<"Layer 2 Grads:\n\nWeights Grads:\n"<<linear_2.weights().gradients()<<"\n\nBias Grads:\n"<<linear_2.bias().gradients()<<"\n\n";
    std::cout<<"Layer 3 Grads:\n\nWeights Grads:\n"<<linear_3.weights().gradients()<<"\n\nBias Grads:\n"<<linear_3.bias().gradients()<<"\n\n";

    return 0;

}
