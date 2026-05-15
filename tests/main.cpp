#include "../Forge.h"

int main() {

    auto tensor {Forge::Tensor::Ones({10, 6}, true, Forge::Dtype::float32)};
    auto* init {Forge::primitive_dispatcher().lookup<Forge::WeightsInitAbstract>(Forge::primitive_ops::weightsInit, tensor.dispatch_key())};
    init->initialize(tensor, tensor.shape(), Forge::Initializers::he_normal);

    // std::cout<<tensor;

    Forge::SelfAttention attention {6, 3, 10, 1, true, Forge::Device::CPU, Forge::Dtype::float32};
    attention.createMask(10);
    auto ctx_embd {attention(tensor)};

    std::cout<<ctx_embd;
    ctx_embd.backward();

    std::cout<<"\nInput Grads:\n"<<tensor.gradients()<<"\n\n"
             <<"Q_W grads:\n"<<attention.query().gradients()<<"\n\n"
             <<"K_W grads:\n"<<attention.key().gradients()<<"\n\n"
             <<"V_W grads:\n"<<attention.value().gradients()<<"\n\n";

    std::cout<<"\nL_W grads: \n"<<attention.linear().weights().gradients();
    // Forge::Softmax softmax;
    // std::cout<<softmax(tensor);

    return 0;
}

/*
 * inp = (10, 6) -> (1, 1, 10, 6)
 *  Q and K = (1, 10, 3) -> (1, 1, 10, 3)
 *
 */