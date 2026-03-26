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
    SoftmaxCPU softmax_cpu{};
    TanhCPU tanh_cpu{};
    GeluCPU gelu_cpu{};
    SigmoidCPU sigmoid_cpu{};
    LeakyReluCPU leaky_relu_cpu{};

    RegisterPrimitives() {
        primitive_dispatcher().register_kernel<LinearAbstract>(&linear_layer_cpu, primitive_ops::linear,DispatchKey::CPU);
        primitive_dispatcher().register_kernel<WeightsInitCPU>(&weights_init_cpu, primitive_ops::weightsInit,DispatchKey::CPU);
        primitive_dispatcher().register_kernel<LinearGradsAbstract>(&linear_grads_cpu, primitive_ops::linear, DispatchKey::CPU_Autodiff);
        primitive_dispatcher().register_kernel<ReluAbstract>(&relu_cpu, primitive_ops::relu, DispatchKey::CPU);
        primitive_dispatcher().register_kernel<SoftmaxAbstract>(&softmax_cpu, primitive_ops::softmax, DispatchKey::CPU);
        primitive_dispatcher().register_kernel<TanhAbstract>(&tanh_cpu, primitive_ops::tanh, DispatchKey::CPU);
        primitive_dispatcher().register_kernel<GeluCPU>(&gelu_cpu, primitive_ops::gelu, DispatchKey::CPU);
        primitive_dispatcher().register_kernel<SigmoidAbstract>(&sigmoid_cpu, primitive_ops::sigmoid, DispatchKey::CPU);
        primitive_dispatcher().register_kernel<LeakyReluAbstract>(&leaky_relu_cpu, primitive_ops::leaky_relu, DispatchKey::CPU);
    }
};

namespace Forge {
    static RegisterPrimitives prim_reg;
}


