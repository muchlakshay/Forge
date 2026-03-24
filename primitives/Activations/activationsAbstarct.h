#pragma once

#include "tensor/tensor.h"
#include "ops/Dispatcher/kernel_base.h"

namespace Forge {
    struct ReluAbstract;
    struct SigmoidAbstract;
    struct SoftmaxAbstract;
    struct TanhAbstract;
    struct LeakyReluAbstract;
}

struct Forge::ReluAbstract : Kernel{
    ReluAbstract() : Kernel{ctti::type_id<ReluAbstract>()}{}
    virtual void forward(const Tensor& input, Tensor& output) = 0;
};

struct Forge::SigmoidAbstract : Kernel{
    SigmoidAbstract() : Kernel{ctti::type_id<SigmoidAbstract>()}{}
    virtual void forward(const Tensor& input, Tensor& output) = 0;
};

struct Forge::SoftmaxAbstract : Kernel{
    SoftmaxAbstract() : Kernel{ctti::type_id<SoftmaxAbstract>()}{}
    virtual void forward(const Tensor& input, Tensor& output) = 0;
};

struct Forge::TanhAbstract : Kernel{
    TanhAbstract() : Kernel{ctti::type_id<TanhAbstract>()}{}
    virtual void forward(const Tensor& input, Tensor& output) = 0;
};

struct Forge::LeakyReluAbstract : Kernel{
    LeakyReluAbstract() : Kernel{ctti::type_id<LeakyReluAbstract>()}{}
    virtual void forward(const Tensor& input, Tensor& output) = 0;
};