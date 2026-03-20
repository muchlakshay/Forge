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

    RegisterPrimitives() {
        primitive_dispatcher().register_kernel<LinearAbstract>(&linear_layer_cpu, primitive_ops::linear,
            DispatchKey::CPU);
        primitive_dispatcher().register_kernel<WeightsInitCPU>(&weights_init_cpu, primitive_ops::weightsInit,
            DispatchKey::CPU);
        primitive_dispatcher().register_kernel<LinearGradsAbstract>(&linear_grads_cpu, primitive_ops::linear,
            DispatchKey::CPU_Autodiff);
    }
};

namespace Forge {
    static RegisterPrimitives prim_reg;
}


