#pragma once
#include "../acvitvationsGradsAbstract.h"

namespace Forge {
    struct ReluGradsCPU;
    struct LeakyReluGradsCPU;
    struct SoftmaxGradsCPU;
    struct GeluGradsCPU;
    struct SigmoidGradsCPU;
    struct TanhGradsCPU;
}

struct Forge::ReluGradsCPU : ReluGradsAbstract {void compute_grads(const Tensor& preactivations,
    const Tensor& activations) const override;};
struct Forge::LeakyReluGradsCPU : LeakyReluGradsAbstract {void compute_grads(const Tensor& preactivations,
    const Tensor& activations) const override;};
struct Forge::GeluGradsCPU : GeluGradsAbstract {void compute_grads(const Tensor& preactivations,
    const Tensor& activations) const override;};
struct Forge::SoftmaxGradsCPU : SoftmaxGradsAbstract {void compute_grads(const Tensor& logits,
    const Tensor& activations) const override;};
struct Forge::SigmoidGradsCPU : SigmoidGradsAbstract {void compute_grads(const Tensor& preactivations,
    const Tensor& activations) const override;};
struct Forge::TanhGradsCPU : TanhGradsAbstract {void compute_grads(const Tensor& preactivations,
    const Tensor& activations) const override;};