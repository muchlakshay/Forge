#pragma once
#include "ops/Dispatcher/kernel_base.h"

namespace Forge {
    struct MaskAbstract;
    class Tensor;
}

struct Forge::MaskAbstract : Kernel {
    MaskAbstract() : Kernel{ctti::type_id<MaskAbstract>()}{}
    virtual void generate(Tensor tensor, std::size_t seq_len) const = 0;
};