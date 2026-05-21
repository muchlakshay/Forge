#pragma once
#include "ops/Dispatcher/kernel_base.h"

namespace Forge {
    struct EmbeddingsGradsAbstract;
    class Tensor;
}

struct Forge::EmbeddingsGradsAbstract : Kernel {
    EmbeddingsGradsAbstract() : Kernel {ctti::type_id<EmbeddingsGradsAbstract>()} {}
    virtual void compute_grads(const Tensor& seq_ids, const Tensor& all_embeddings, const Tensor& opt) const = 0;
};