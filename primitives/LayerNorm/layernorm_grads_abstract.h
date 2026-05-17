#pragma once
#include "ops/Dispatcher/kernel_base.h"

namespace Forge {
    struct LayerNormGradsAbstract;
    class Tensor;
}

struct Forge::LayerNormGradsAbstract : Kernel {
    LayerNormGradsAbstract() : Kernel{ctti::type_id<LayerNormGradsAbstract>()} {}
    virtual void compute_grads(const Tensor& input, const Tensor& gamma, const Tensor& beta, const Tensor& opt) const = 0;
};