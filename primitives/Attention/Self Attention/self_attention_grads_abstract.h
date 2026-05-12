#pragma once
#include <ctti/type_id.hpp>

#include "ops/Dispatcher/kernel_base.h"

namespace Forge {
    struct SelfAttentionGradsAbstract;
    class Tensor;
}

struct Forge::SelfAttentionGradsAbstract : Kernel {
    SelfAttentionGradsAbstract () : Kernel {ctti::type_id<SelfAttentionGradsAbstract>()} {}
    virtual void compute_grads(Tensor& inp, Tensor& Q_W, Tensor& K_W, Tensor& V_W, Tensor& QcKs, Tensor& atten_scores,
        Tensor& AcVr, Tensor& mask, Tensor& opt) const = 0;
};