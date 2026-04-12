#pragma once
#include "ops/Dispatcher/kernel_base.h"
#include "tensor.h"
#include <vector>

namespace Forge{
    struct SGDUpdateAbstract;
}

struct Forge::SGDUpdateAbstract : Kernel {
    SGDUpdateAbstract() : Kernel{ctti::type_id<SGDUpdateAbstract>()} {}
    virtual void update(const std::vector<Tensor*>& params, float lr, float momentum_coef,
        const std::vector<Tensor>& V) const = 0;
};