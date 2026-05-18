#include "../Forge.h"

int main() {

    auto tensor {Forge::Tensor::Ones({10, 6}, true, Forge::Dtype::float32)};
    auto tensor_2 {Forge::Tensor::Ones({1, 10, 6}, true)};

    auto* init {Forge::primitive_dispatcher().lookup<Forge::WeightsInitAbstract>(Forge::primitive_ops::weightsInit, tensor.dispatch_key())};
    init->initialize(tensor, tensor.shape(), Forge::Initializers::he_normal);
    init->initialize(tensor_2, tensor.shape(), Forge::Initializers::he_normal);

    Forge::SelfAttention attention {6, 3, 10, 1, true, Forge::Device::CPU, Forge::Dtype::float32};
    attention.createMask(10);
    auto ctx_embd {attention(tensor)};

    std::cout<<"Ctx Embd: \n\n"<<ctx_embd<<"\n"<<"Dims: "<<ctx_embd.gradients().shape().size()<<" "<<ctx_embd.shape().size()<<"\n";

    Forge::LayerNorm layer_norm {6, tensor.dtype(), tensor.device()};
    Forge::MSE mse {};
    auto layer_norm_opt {layer_norm(ctx_embd)};
    auto loss {mse(layer_norm_opt, tensor_2)};

    loss.backward();

    std::cout<<"Norm:\n"<<layer_norm_opt<<"\n\n"<<"Gamma:\n"<<layer_norm.gamma()<<"\nBeta"<<layer_norm.beta()<<"\n";

    std::cout<<"Q_W grads:\n"<<attention.query().gradients()<<"\n\n"
             <<"K_W grads:\n"<<attention.key().gradients()<<"\n\n"
             <<"V_W grads:\n"<<attention.value().gradients()<<"\n\n"
             <<"\nInput Grads:\n"<<tensor.gradients()<<"\n\n"
             <<"\nCTX EMBD Grads:\n"<<ctx_embd.gradients()<<"\n\n"
             <<"gamma grads:\n"<<layer_norm.gamma().gradients()<<"\n\n"
             <<"beta grads:\n"<<layer_norm.beta().gradients()<<"\n\n";


    std::cout<<"\nL_W grads: \n"<<attention.linear().weights().gradients();

    return 0;
}

/*
 * inp = (10, 6) -> (1, 1, 10, 6)
 *  Q and K = (1, 10, 3) -> (1, 1, 10, 3)
 *  mean = -0.166748483
 * var = 0.36880879578
 *
 */