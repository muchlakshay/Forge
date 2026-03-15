#pragma once
#include "LinearGradsAbstract.h"

namespace Forge {
    struct LinearGradsCPU;
}

struct Forge::LinearGradsCPU final :  LinearGradsAbstract {
    void compute_grads(Tensor& input, Tensor& weights, Tensor& bias, const Tensor& output) const override {

    }
};