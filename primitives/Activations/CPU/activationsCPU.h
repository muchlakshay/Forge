#pragma once

#include "../activationsAbstarct.h"

namespace Forge {
    struct ReluCPU;
    struct SigmoidCPU;
    struct TanhCPU;
    struct SoftmaxCPU;
    struct LeakyReluCPU;
    struct GeluCPU;
}

struct Forge::ReluCPU : ReluAbstract{void forward(const Tensor& input, Tensor& output) const override;};
struct Forge::SigmoidCPU : SigmoidAbstract{void forward(const Tensor& input, Tensor& output) const override;};
struct Forge::TanhCPU : TanhAbstract {void forward(const Tensor& input, Tensor& output) const override;};
struct Forge::LeakyReluCPU : LeakyReluAbstract {void forward(const Tensor& input, Tensor& output) const override;};
struct Forge::GeluCPU : GeluAbstract {void forward(const Tensor& input, Tensor& output) const override;};
struct Forge::SoftmaxCPU : SoftmaxAbstract{void forward(const Tensor& input, Tensor& output) const override;};