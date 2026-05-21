#pragma once
#include "../embeddings_abstract.h"

namespace Forge {
    struct EmbeddingsImplCPU;
}

struct Forge::EmbeddingsImplCPU : EmbeddingsImplAbstract {
    void getEmbeddings(const Tensor& seq_ids, const Tensor& all_embeddings, Tensor& opt) const override;
};