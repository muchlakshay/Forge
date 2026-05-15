#pragma once
#include <ctti/type_id.hpp>

#include "ops/Dispatcher/kernel_base.h"

namespace Forge {
    struct SelfAttentionGradsAbstract;
    class Tensor;
}

struct Forge::SelfAttentionGradsAbstract : Kernel {
    SelfAttentionGradsAbstract () : Kernel {ctti::type_id<SelfAttentionGradsAbstract>()} {}
    virtual void compute_grads(const Tensor &inp, const Tensor &Q_W, const Tensor &K_W, const Tensor &V_W,
    const Tensor &QcKs, const Tensor &atten_scores, const Tensor &AcVr, const  Tensor& mask, const Tensor& opt_l,
    const Tensor &opt) const = 0;
};