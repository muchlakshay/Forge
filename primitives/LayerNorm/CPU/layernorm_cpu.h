#pragma once
#include "../layernorm_abstract.h"

namespace Forge {
    struct LayerNormImplCPU;
}

struct Forge::LayerNormImplCPU : LayerNormImplAbstract {
  void forward(const Tensor &input, const Tensor &gamma, const Tensor &beta, Tensor &opt) const override;
};
