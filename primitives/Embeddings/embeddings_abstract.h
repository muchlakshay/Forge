#pragma once
#include "ops/Dispatcher/kernel_base.h"

namespace Forge {
    struct EmbeddingsImplAbstract;
    class Tensor;
}

struct Forge::EmbeddingsImplAbstract : Kernel {
    EmbeddingsImplAbstract() : Kernel(ctti::type_id<EmbeddingsImplAbstract>()) {
    }

    virtual void getEmbeddings(const Tensor &seq_ids, const Tensor &all_embeddings, Tensor &opt) const = 0;
};
