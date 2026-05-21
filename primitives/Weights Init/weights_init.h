#pragma once
#include <random>
#include "ops/Dispatcher/kernel_base.h"
#include "tensor.h"

namespace Forge {
    struct WeightsInitAbstract;
    struct WeightsInitCPU;
}

struct Forge::WeightsInitAbstract : Kernel {
    WeightsInitAbstract() : Kernel{ctti::type_id<WeightsInitAbstract>()} {}
    virtual void initialize(Tensor& weights, const std::vector<std::size_t>& dims,
        Initializers initializer) const = 0;
};