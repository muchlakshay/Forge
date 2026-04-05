#pragma once
#include "lossFunctionsGrads.h"
#include "ops/Dispatcher/kernel_base.h"
#include "tensor.h"

namespace Forge {
    struct MSEGradsAbstract;
    struct CrossEntropyGradsAbstract;
    struct BinaryCrossEntropyGradsAbstract;
}

struct Forge::MSEGradsAbstract : Kernel {
    MSEGradsAbstract() : Kernel{ctti::type_id<MSEGradsAbstract>()}{}
    virtual void compute_grads(const Tensor& pred, const Tensor& ground_truth, const Tensor& loss) const = 0;
};
struct Forge::CrossEntropyGradsAbstract : Kernel {
    CrossEntropyGradsAbstract() : Kernel{ctti::type_id<CrossEntropyGradsAbstract>()}{}
    virtual void compute_grads(const Tensor& pred, const Tensor& ground_truth, const Tensor& loss) const = 0;
};
struct Forge::BinaryCrossEntropyGradsAbstract : Kernel {
    BinaryCrossEntropyGradsAbstract() : Kernel{ctti::type_id<BinaryCrossEntropyGradsAbstract>()}{}
    virtual void compute_grads(const Tensor& pred, const Tensor& ground_truth, const Tensor& loss) const = 0;
};