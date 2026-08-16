#pragma once
#include "Attention/Self Attention/self_attention_impl_abstract.h"

namespace Forge {
    struct SelfAttentionCPU;
}

struct Forge::SelfAttentionCPU : SelfAttentionImplAbstract {
    void forward(const Tensor &input, const Tensor &query_W, const Tensor &key_W, const Tensor &value_W, Tensor &output,
                 const Tensor &mask, const Tensor &Q_bias, const Tensor &K_bias, const Tensor &V_bias,
                 const Linear &linear, std::size_t heads, std::size_t d_model, bool using_mask) const override;
};