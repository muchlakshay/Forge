#pragma once
#include "tensor.h"
#include "ops/Dispatcher/kernel_base.h"
#include "Linear/LinearLayer.h"

namespace Forge {
    struct SelfAttentionImplAbstract;
}

struct Forge::SelfAttentionImplAbstract : Kernel{
    SelfAttentionImplAbstract() : Kernel{ctti::type_id<SelfAttentionImplAbstract>()} {}
    virtual void forward(const Tensor &input, const Tensor &query_W, const Tensor &key_W, const Tensor &value_W, Tensor &output,
                         const Tensor &mask, const Tensor& Q_bias, const Tensor& K_bias, const Tensor &V_bias,
                         const Linear &linear, std::size_t heads, std::size_t d_model, bool using_mask) const = 0;
};