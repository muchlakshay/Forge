#pragma once

#include "../weights_init.h"
#include "../Linear/LinearLayerCPU.h"

namespace  Forge {
    struct RegisterPrimitives;
}

struct Forge::RegisterPrimitives {
    LinearCPU linear_layer_cpu;
    WeightsInitCPU weights_init_cpu;

    RegisterPrimitives() {
        primitive_dispatcher().register_kernel<LinearAbstract>(&linear_layer_cpu, primitive_ops::linear,
            DispatchKey::CPU);
        primitive_dispatcher().register_kernel<WeightsInitCPU>(&weights_init_cpu, primitive_ops::weightsInit,
            DispatchKey::CPU);
    }
};

namespace Forge {
    static RegisterPrimitives __;
}


