#pragma once
#include "ops/Dispatcher/kernel_base.h"
#include "tensor.h"
#include "../parameter.h"
#include <vector>

namespace Forge{
    struct SGDUpdateAbstract;
    struct AdamUpdateAbstract;
}

struct Forge::SGDUpdateAbstract : Kernel {
    SGDUpdateAbstract() : Kernel{ctti::type_id<SGDUpdateAbstract>()} {}
    virtual void update(const std::vector<Parameter>& params, float lr, float momentum_coef,
        const std::vector<Tensor>& V) const = 0;
};

struct Forge::AdamUpdateAbstract : Kernel {
    AdamUpdateAbstract() : Kernel{ctti::type_id<AdamUpdateAbstract>()} {}
    virtual void update(const std::vector<Parameter>& params, float lr, const std::vector<Tensor>& V,
        const std::vector<Tensor>& M, float beta_1, float beta_2, float decay_factor, int epoch) const = 0;
};