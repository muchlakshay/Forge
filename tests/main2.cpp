#include <iostream>
#include "../Forge.h"
#include "../primitives/Activations/activations.h"

struct MLP {
    Forge::Linear linear_1 {2, 4}, linear_2 {4, 4};
    Forge::Relu sigmoid{};
    Forge::Linear opt {4, 1};
    auto operator()(const Forge::Tensor& input) const {return opt(sigmoid(linear_2(sigmoid(linear_1(input)))));}

    [[nodiscard]] auto parameters() const {
        std::vector params {const_cast<Forge::Tensor*>(&linear_1.weights()), const_cast<Forge::Tensor*>(&linear_1.bias()),
            const_cast<Forge::Tensor*>(&opt.weights()), const_cast<Forge::Tensor*>(&opt.bias()), const_cast<Forge::Tensor*>(&linear_2.weights()),
            const_cast<Forge::Tensor*>(&linear_2.bias())};
        return params;
    }
};

int main() {

    Forge::Tensor XOR {{4, 2}};
    Forge::Tensor OPT {{4, 1}};
    XOR = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
    OPT = {{0}, {1}, {1}, {0}};

    MLP mlp;
    Forge::SGD optimizer {mlp.parameters(), 0.8f};
    Forge::BinaryCrossEntropy bce;

    int epochs {100};
    for (int i {1}; i<=epochs; ++i) {
        auto loss {bce(mlp(XOR), OPT)};
        bce.backward();
        std::cout<<"Epoch: "<<i<<" Loss: "<<*static_cast<float*>(loss.data())<<"\n";
        optimizer.update();
        optimizer.clear_grads();
    }
    Forge::Sigmoid sigmoid;
    std::cout<<sigmoid(mlp(XOR));
    return 0;

}
