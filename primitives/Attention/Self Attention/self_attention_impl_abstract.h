#pragma once
#include "tensor.h"
#include "ops/Dispatcher/kernel_base.h"

namespace Forge {
    struct SelfAttentionImplAbstract;
}

struct Forge::SelfAttentionImplAbstract : Kernel{
    SelfAttentionImplAbstract() : Kernel{ctti::type_id<SelfAttentionImplAbstract>()} {}
    virtual void forward(Tensor input, Tensor query_W, Tensor key_W, Tensor value_W, Tensor output, bool mask) const = 0;
};