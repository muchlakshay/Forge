#pragma once
#include "../layernorm_grads_abstract.h"

namespace Forge {
    struct LayerNormGradsCPU;
    class Tensor;
}

struct Forge::LayerNormGradsCPU :  LayerNormGradsAbstract {
    void compute_grads(const Tensor& input, const Tensor& gamma, const Tensor& beta, const Tensor& opt) const override;
};