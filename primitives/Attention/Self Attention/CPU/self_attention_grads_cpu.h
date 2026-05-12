#pragma once
#include "../self_attention_grads_abstract.h"

namespace Forge {
    struct SelfAttentionGradsCPU;
}

struct Forge::SelfAttentionGradsCPU : SelfAttentionGradsAbstract {
    void compute_grads(Tensor &inp, Tensor &Q_W, Tensor &K_W, Tensor &V_W, Tensor &QcKs, Tensor &atten_scores, Tensor &AcVr,
        Tensor& mask, Tensor &opt) const override;
};