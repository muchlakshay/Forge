#pragma once
#include <cstddef>
#include "tensor.h"
#include "Linear/LinearLayer.h"
#include "Weights Init/weights_init.h"

namespace Forge {
    class SelfAttention;
}

class Forge::SelfAttention {
    Tensor m_query_W {};
    Tensor m_key_W {};
    Tensor m_value_W {};
    Tensor m_mask_ {};
    Linear m_linear;
    bool m_mask {};
    DispatchKey m_dispatch_key {};
    std::size_t m_heads{};
    std::size_t m_d_model{};
    Device m_device;
    Dtype m_dtype;
public:
    SelfAttention(std::size_t d_model, std::size_t Q_K_dims, std::size_t V_dims, std::size_t heads, bool mask,
        Device device, Dtype=Dtype::float32, Initializers initializer=Initializers::he_normal);
    Tensor operator()(const Tensor& input) const;
    void createMask(std::size_t seq_len);
    auto& mask() const {return m_mask_;}
    auto& linear() {return m_linear;}
    [[nodiscard]] auto& query() const {return m_query_W;}
    [[nodiscard]] auto& key() const {return m_key_W;}
    [[nodiscard]] auto& value() const {return m_value_W;}
    [[nodiscard]] auto using_mask() const {return m_mask;}
    [[nodiscard]] auto dispatch_key() const {return m_dispatch_key;}
    [[nodiscard]] auto device() const {return m_device;}
    [[nodiscard]] auto dtype() const {return m_dtype;}
    [[nodiscard]] auto heads() const {return m_heads;}
    [[nodiscard]] auto d_model() const {return m_d_model;}
    [[nodiscard]] auto useMask() const {return m_mask;}
};