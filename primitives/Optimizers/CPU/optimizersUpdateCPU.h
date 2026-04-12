#pragma once
#include "../optimizersUpdateAbstract.h"

namespace Forge {
    struct SGDUpdateCPU;
}

struct Forge::SGDUpdateCPU : SGDUpdateAbstract {void update(const std::vector<Tensor*>& params, float lr,
    float momentum_coef, const std::vector<Tensor>& V) const override;};