#pragma once
#include "tensor.h"
#include "../LinearAbstract.h"

namespace Forge {
    struct LinearCPU;
}

struct Forge::LinearCPU final : LinearAbstract {
    void forward(const Tensor& input, const Tensor& output, const Tensor& weights,
        const Tensor& bias, bool using_bias) const override;};
