#pragma once
#include "../LinearGradsAbstract.h"

namespace Forge {
    struct LinearGradsCPU;
}

struct Forge::LinearGradsCPU final :  LinearGradsAbstract {
    void compute_grads(const Tensor& input, const Tensor& weights, const Tensor& bias,
            const Tensor& output) const override;};