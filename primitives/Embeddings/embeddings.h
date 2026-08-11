#pragma once
#include <cstddef>
#include <vector>
#include "tensor.h"
#include "parameter.h"

namespace Forge {
    class Embedding;
    class Tensor;
}

class Forge::Embedding {
    Tensor m_embeddings;
    std::size_t m_d_model {};
    std::size_t m_vocab_size {};
public:
    Embedding(std::size_t d_model, std::size_t vocab_size, Dtype dtype=Dtype::float32,
        bool need_grads=true, Initializers initializer = Initializers::xavier_normal, Device device=Device::CPU);
    Embedding() = default;

    Tensor operator()(const Tensor& seq_ids);
    auto& all_embeddings() { return m_embeddings; }
    auto parameters() {return std::vector{Parameter{&m_embeddings, true, "embd.w"}};}
    [[nodiscard]] auto d_model() const { return m_d_model; }
    [[nodiscard]] auto vocab_size() const { return m_vocab_size; }
};