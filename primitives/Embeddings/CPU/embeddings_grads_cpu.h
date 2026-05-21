#pragma once
#include "../embeddings_grads_abstract.h"

namespace Forge {
    struct EmbeddingsGradsCPU;
}

struct Forge::EmbeddingsGradsCPU : EmbeddingsGradsAbstract {
    void compute_grads(const Tensor &seq_ids, const Tensor &all_embeddings, const Tensor &opt) const override;
};