#include "embeddings_grads_cpu.h"

#include "tensor.h"

void Forge::EmbeddingsGradsCPU::compute_grads(const Tensor &seq_ids, const Tensor &all_embeddings, const Tensor &opt) const {
    using grads_t = float;

    std::size_t d_model {opt.shape()[1]};

    auto seq_ids_ptr {static_cast<int*>(seq_ids.data())};
    auto* embd_grads_ptr {static_cast<grads_t*>(all_embeddings.gradients().data())};
    auto* opt_grads_ptr {static_cast<grads_t*>(opt.gradients().data())};

    for (std::size_t seq{}; seq < opt.shape()[0]; ++seq) {
        auto* embd_grads_row {embd_grads_ptr + seq_ids_ptr[seq] * d_model};
        auto* opt_grads_row {opt_grads_ptr + seq * d_model};

        std::size_t d {};
        for (;d+3<d_model; d+=4) {
            embd_grads_row[d]   += opt_grads_row[d];
            embd_grads_row[d+1] += opt_grads_row[d+1];
            embd_grads_row[d+2] += opt_grads_row[d+2];
            embd_grads_row[d+3] += opt_grads_row[d+3];
        }
        for (; d<d_model; d++) embd_grads_row[d]   += opt_grads_row[d];
    }
}