#pragma once
#include "tensor.h"
#include "ops/Dispatcher/kernel_base.h"

namespace Forge {
    struct LinearGradsAbstract;
}

struct Forge::LinearGradsAbstract : Kernel {
    LinearGradsAbstract() : Kernel {ctti::type_id<LinearGradsAbstract>()} {}
    virtual void compute_grads(Tensor& input, Tensor& weights, Tensor& bias, const Tensor& output) const = 0;
};
