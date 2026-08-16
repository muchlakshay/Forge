#pragma once
#include "../self_attention_grads_abstract.h"

namespace Forge {
    struct SelfAttentionGradsCPU;
}

struct Forge::SelfAttentionGradsCPU : SelfAttentionGradsAbstract {
    void compute_grads(const Tensor &inp, const Tensor &Q_W, const Tensor &K_W, const Tensor &V_W, const Tensor& Q_bias,
        const Tensor& K_bias, const Tensor& V_bias, const Tensor &QcKs, const Tensor &atten_scores, const Tensor &AcVr,
        const  Tensor& mask, const Tensor& opt_l,
        const Tensor &opt) const override;
};