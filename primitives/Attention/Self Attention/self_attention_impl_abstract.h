#pragma once
#include "tensor.h"
#include "ops/Dispatcher/kernel_base.h"
#include "Linear/LinearLayer.h"

namespace Forge {
    struct SelfAttentionImplAbstract;
}

struct Forge::SelfAttentionImplAbstract : Kernel{
    SelfAttentionImplAbstract() : Kernel{ctti::type_id<SelfAttentionImplAbstract>()} {}
    virtual void forward(Tensor input, Tensor query_W, Tensor key_W, Tensor value_W, Tensor output, Linear& linear,
        std::size_t heads, bool mask) const = 0;
};