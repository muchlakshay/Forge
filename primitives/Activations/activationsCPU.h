#pragma once

#include "activationsAbstarct.h"
#include "activationsCPU.h"

namespace Forge {
    struct ReluCPU;
    struct SigmoidCPU;
    struct TanhCPU;
    struct SoftmaxCPU;
    struct LeakyReluCPU;
}

struct Forge::ReluCPU : ReluAbstract{
    void forward(const Tensor& input, Tensor& output) override {
        DISPATCH_ALL_TYPES(input.dtype(), Device::CPU, [&] {
           auto input_map {input.as_eigen<scalar_t>()};
            auto output_map {output.as_eigen<scalar_t>()};
            output_map = input_map.cwiseMax(static_cast<scalar_t>(0));
        });
    }
};
