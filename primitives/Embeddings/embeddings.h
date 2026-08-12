#pragma once
#include <cstddef>
#include <vector>
#include "tensor.h"
#include "parameter.h"
#include "instance_tracker.h"

namespace Forge {
    class Embedding;
    class Tensor;
}

class Forge::Embedding {
    Tensor m_embeddings;
    std::size_t m_d_model {};
    std::size_t m_vocab_size {};
    inline static InstanceTracker m_tracker;
    int m_cnt {};
public:
    Embedding(std::size_t d_model, std::size_t vocab_size, Dtype dtype=Dtype::float32,
        bool need_grads=true, Initializers initializer = Initializers::xavier_normal, Device device=Device::CPU);
    Embedding() {m_cnt=m_tracker.m_count; ++m_tracker.m_count;}

    Tensor operator()(const Tensor& seq_ids);
    auto& all_embeddings() { return m_embeddings; }
    auto parameters() {
        return std::vector{Parameter{&m_embeddings, true, std::format("embd.{}.w", m_cnt)}};
    }
    [[nodiscard]] auto d_model() const { return m_d_model; }
    [[nodiscard]] auto vocab_size() const { return m_vocab_size; }
};