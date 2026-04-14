#pragma once
#include "../optimizersUpdateAbstract.h"

namespace Forge {
    struct SGDUpdateCPU;
    struct AdamUpdateCPU;
}

struct Forge::SGDUpdateCPU : SGDUpdateAbstract {void update(const std::vector<Tensor*>& params, float lr,
    float momentum_coef, const std::vector<Tensor>& V) const override;};

struct Forge::AdamUpdateCPU : AdamUpdateAbstract {
    void update(const std::vector<Tensor *> &params, float lr, const std::vector<Tensor> &V,
        const std::vector<Tensor> &M, float beta_1, float beta_2, int epoch) const override;
};