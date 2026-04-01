#pragma once
#include "ops/Dispatcher/kernel_base.h"
#include "tensor.h"

namespace Forge {
    struct MSEAbstract;
    struct CrossEntropyAbstract;
    struct BinaryCrossEntropyAbstract;
}

struct Forge::MSEAbstract : Kernel {
    MSEAbstract() : Kernel {ctti::type_id<MSEAbstract>()} {}
    virtual void compute_loss(const Tensor& pred, const Tensor& ground_truth, Tensor& loss) const = 0;
};
struct Forge::CrossEntropyAbstract : Kernel {
    CrossEntropyAbstract() : Kernel {ctti::type_id<CrossEntropyAbstract>()} {}
    virtual void compute_loss(const Tensor& pred, const Tensor& ground_truth, Tensor& loss) const = 0;
};
struct Forge::BinaryCrossEntropyAbstract : Kernel {
    BinaryCrossEntropyAbstract() : Kernel {ctti::type_id<BinaryCrossEntropyAbstract>()} {}
    virtual void compute_loss(const Tensor& pred, const Tensor& ground_truth, Tensor& loss) const = 0;
};