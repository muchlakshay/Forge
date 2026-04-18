#pragma once
#include <cstddef>
#include "tensor.h"
#include "Weights Init/weights_init.h"

namespace Forge {
    class SelfAttention;
}

class Forge::SelfAttention {
    Tensor m_query_W {};
    Tensor m_key_W {};
    Tensor m_value_W {};
    bool m_mask {};
    DispatchKey m_dispatch_key {};
public:
    SelfAttention(std::size_t seq_len, std::size_t Q_K_dims, std::size_t V_dims, std::size_t heads, bool mask,
        Device device, Dtype=Dtype::float32, Initializers initializer=Initializers::he_normal);
    Tensor operator()(const Tensor& input);
    [[nodiscard]] auto query() const {return m_query_W;}
    [[nodiscard]] auto key() const {return m_key_W;}
    [[nodiscard]] auto value() const {return m_value_W;}
    [[nodiscard]] auto mask() const {return m_mask;}
    [[nodiscard]] auto dispatch_key() const {return m_dispatch_key;}
};