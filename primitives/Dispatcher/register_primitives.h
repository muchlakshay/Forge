#pragma once

#include "../weights_init.h"
#include "../Linear/LinearLayerCPU.h"

namespace  Forge {
    struct RegisterPrimitives;
}

struct Forge::RegisterPrimitives {
    LinearCPU linear_layer_cpu;

    RegisterPrimitives() {
        primitive_dispatcher().register_kernel<LinearAbstract>(&linear_layer_cpu, primitive_ops::linear,
            DispatchKey::CPU);
    }
};

namespace Forge {
    static RegisterPrimitives __;
}


