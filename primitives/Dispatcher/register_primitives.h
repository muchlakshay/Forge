#pragma once

#include "primitives.h"
#include "primitive_dispatcher.h"

namespace  Forge {
    struct RegisterPrimitives;
}

struct Forge::RegisterPrimitives {
    LinearCPU linear_layer_cpu{};
    LinearGradsCPU linear_grads_cpu {};

    WeightsInitCPU weights_init_cpu{};

    ReluCPU relu_cpu{};
    ReluGradsCPU relu_grads_cpu {};

    SoftmaxCPU softmax_cpu{};
    SoftmaxGradsCPU softmax_grads_cpu {};

    TanhCPU tanh_cpu{};
    TanhGradsCPU tanh_grads_cpu {};

    GeluCPU gelu_cpu{};
    GeluGradsCPU gelu_grads_cpu {};

    SigmoidCPU sigmoid_cpu{};
    SigmoidGradsCPU sigmoid_grads_cpu {};

    LeakyReluCPU leaky_relu_cpu{};
    LeakyReluGradsCPU leaky_relu_grads_cpu {};

    MSE_CPU mse_cpu{};
    MSEGradsCPU mse_grads_cpu {};

    CrossEntropyCPU crossEntropy_cpu{};
    CrossEntropyGradsCPU crossEntropyGrads_cpu {};

    BinaryCrossEntropyCPU binaryCrossEntropy_cpu {};
    BinaryCrossEntropyGradsCPU binaryCrossEntropyGrads_cpu {};

    SGDUpdateCPU sgdUpdate_cpu{};
    AdamUpdateCPU adamUpdate_cpu{};

    SelfAttentionCPU self_attention_cpu{};
    SelfAttentionGradsCPU self_attention_grads_cpu {};
    MaskCPU mask_cpu{};

    RegisterPrimitives() {
        primitive_dispatcher().register_kernel<LinearAbstract>(&linear_layer_cpu, primitive_ops::linear,DispatchKey::CPU);
        primitive_dispatcher().register_kernel<LinearGradsAbstract>(&linear_grads_cpu, primitive_ops::linear, DispatchKey::CPU_Autodiff);

        primitive_dispatcher().register_kernel<WeightsInitAbstract>(&weights_init_cpu, primitive_ops::weightsInit,DispatchKey::CPU);

        primitive_dispatcher().register_kernel<ReluAbstract>(&relu_cpu, primitive_ops::relu, DispatchKey::CPU);
        primitive_dispatcher().register_kernel<ReluGradsAbstract>(&relu_grads_cpu, primitive_ops::relu, DispatchKey::CPU_Autodiff);

        primitive_dispatcher().register_kernel<SoftmaxAbstract>(&softmax_cpu, primitive_ops::softmax, DispatchKey::CPU);
        primitive_dispatcher().register_kernel<SoftmaxGradsAbstract>(&softmax_grads_cpu, primitive_ops::softmax, DispatchKey::CPU_Autodiff);

        primitive_dispatcher().register_kernel<TanhAbstract>(&tanh_cpu, primitive_ops::tanh, DispatchKey::CPU);
        primitive_dispatcher().register_kernel<TanhGradsAbstract>(&tanh_grads_cpu, primitive_ops::tanh, DispatchKey::CPU_Autodiff);

        primitive_dispatcher().register_kernel<GeluAbstract>(&gelu_cpu, primitive_ops::gelu, DispatchKey::CPU);
        primitive_dispatcher().register_kernel<GeluGradsAbstract>(&gelu_grads_cpu, primitive_ops::gelu, DispatchKey::CPU_Autodiff);

        primitive_dispatcher().register_kernel<SigmoidAbstract>(&sigmoid_cpu, primitive_ops::sigmoid, DispatchKey::CPU);
        primitive_dispatcher().register_kernel<SigmoidGradsAbstract>(&sigmoid_grads_cpu, primitive_ops::sigmoid, DispatchKey::CPU_Autodiff);

        primitive_dispatcher().register_kernel<LeakyReluAbstract>(&leaky_relu_cpu, primitive_ops::leaky_relu, DispatchKey::CPU);
        primitive_dispatcher().register_kernel<LeakyReluGradsAbstract>(&leaky_relu_grads_cpu, primitive_ops::leaky_relu, DispatchKey::CPU_Autodiff);

        primitive_dispatcher().register_kernel<MSEAbstract>(&mse_cpu, primitive_ops::mse, DispatchKey::CPU);
        primitive_dispatcher().register_kernel<MSEGradsAbstract>(&mse_grads_cpu, primitive_ops::mse, DispatchKey::CPU_Autodiff);

        primitive_dispatcher().register_kernel<CrossEntropyAbstract>(&crossEntropy_cpu, primitive_ops::crossEntropy, DispatchKey::CPU);
        primitive_dispatcher().register_kernel<CrossEntropyGradsAbstract>(&crossEntropyGrads_cpu, primitive_ops::crossEntropy, DispatchKey::CPU_Autodiff);

        primitive_dispatcher().register_kernel<BinaryCrossEntropyAbstract>(&binaryCrossEntropy_cpu, primitive_ops::binaryCrossEntropy, DispatchKey::CPU);
        primitive_dispatcher().register_kernel<BinaryCrossEntropyGradsAbstract>(&binaryCrossEntropyGrads_cpu, primitive_ops::binaryCrossEntropy, DispatchKey::CPU_Autodiff);

        primitive_dispatcher().register_kernel<SGDUpdateAbstract>(&sgdUpdate_cpu, primitive_ops::SGD, DispatchKey::CPU);
        primitive_dispatcher().register_kernel<AdamUpdateAbstract>(&adamUpdate_cpu, primitive_ops::Adam, DispatchKey::CPU);

        primitive_dispatcher().register_kernel<SelfAttentionImplAbstract>(&self_attention_cpu, primitive_ops::selfAttention, DispatchKey::CPU);
        primitive_dispatcher().register_kernel<MaskAbstract>(&mask_cpu, primitive_ops::SA_mask, DispatchKey::CPU);
        primitive_dispatcher().register_kernel<SelfAttentionGradsAbstract>(&self_attention_grads_cpu, primitive_ops::selfAttention, DispatchKey::CPU_Autodiff);
    }
};

namespace Forge {
    static RegisterPrimitives prim_reg;
}


