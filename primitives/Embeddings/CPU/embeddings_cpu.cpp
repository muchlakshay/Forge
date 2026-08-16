#include "embeddings_cpu.h"
#include "tensor.h"

void Forge::EmbeddingsImplCPU::getEmbeddings(const Tensor &seq_ids, const Tensor& all_embeddings,  Tensor &opt) const {

    auto seq_ids_map {seq_ids.as_eigen<int, 1>()};
    std::size_t ids {seq_ids.size()};
    auto dtype {all_embeddings.dtype()};
    auto device {all_embeddings.device()};

    DISPATCH_ALL_TYPES(dtype, device, [&] {
        auto embd_map {all_embeddings.as_eigen<scalar_t, 2>()};
        auto opt_map {opt.as_eigen<scalar_t, 2>()};
        for (int id {}; id<ids; ++id) opt_map.chip(id, 0) = embd_map.chip(seq_ids_map(id), 0);
    });
}
