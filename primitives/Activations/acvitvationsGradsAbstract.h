#pragma once
#include "ops/Dispatcher/kernel_base.h"
#include "tensor.h"

namespace Forge {
    struct ReluGradsAbstract;
    struct SigmoidGradsAbstract;
    struct SoftmaxGradsAbstract;
    struct TanhGradsAbstract;
    struct LeakyReluGradsAbstract;
    struct GeluGradsAbstract;
}

struct Forge::ReluGradsAbstract : Kernel {
    ReluGradsAbstract () : Kernel{ctti::type_id<ReluGradsAbstract>()} {}
    virtual void compute_grads(const Tensor& preactivations, const Tensor& activations) const = 0;
};

struct Forge::SigmoidGradsAbstract : Kernel {
    SigmoidGradsAbstract () : Kernel{ctti::type_id<SigmoidGradsAbstract>()} {}
    virtual void compute_grads(const Tensor& preactivations, const Tensor& activations) const = 0;
};

struct Forge::SoftmaxGradsAbstract : Kernel {
    SoftmaxGradsAbstract () : Kernel{ctti::type_id<SoftmaxGradsAbstract>()} {}
    virtual void compute_grads(const Tensor& preactivations, const Tensor& activations) const = 0;
};

struct Forge::TanhGradsAbstract : Kernel {
    TanhGradsAbstract () : Kernel{ctti::type_id<TanhGradsAbstract>()} {}
    virtual void compute_grads(const Tensor& preactivations, const Tensor& activations) const = 0;
};

struct Forge::LeakyReluGradsAbstract : Kernel {
    LeakyReluGradsAbstract () : Kernel{ctti::type_id<LeakyReluGradsAbstract>()} {}
    virtual void compute_grads(const Tensor& preactivations, const Tensor& activations) const = 0;
};

struct Forge::GeluGradsAbstract : Kernel {
    GeluGradsAbstract () : Kernel{ctti::type_id<GeluGradsAbstract>()} {}
    virtual void compute_grads(const Tensor& preactivations, const Tensor& activations) const = 0;
};