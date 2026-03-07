#pragma once
#include "tensor/tensor.h"

namespace Forge {
    struct LinearAbstract;
}

struct Forge::LinearAbstract : Kernel {
    LinearAbstract() : Kernel{ctti::type_id<LinearAbstract>()} {}
    virtual void forward(const Tensor& input, const Tensor& output,
        const Tensor& weights, const Tensor& bias, bool using_bias) const = 0;
};