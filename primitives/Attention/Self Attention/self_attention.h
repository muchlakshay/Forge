#pragma once
#include <cstddef>
#include "tensor.h"
#include "Linear/LinearLayer.h"
#include "Weights Init/weights_init.h"
#include "parameter.h"
#include "primitives/instance_tracker.h"

namespace Forge {
    class SelfAttention;
}

class Forge::SelfAttention {
    Tensor m_query_W {};
    Tensor m_key_W {};
    Tensor m_value_W {};
    Tensor m_mask_ {};
    Tensor m_Q_bias {};
    Tensor m_K_bias {};
    Tensor m_V_bias {};
    Linear m_linear;
    bool m_mask {};
    DispatchKey m_dispatch_key {};
    std::size_t m_heads{};
    std::size_t m_d_model{};
    Device m_device;
    Dtype m_dtype;
    inline static InstanceTracker m_tracker{};
    bool m_qkv_bias {};
    bool m_proj_bias {};
    bool m_need_grads {};
    int m_cnt {};
public:
    SelfAttention(std::size_t d_model, std::size_t Q_K_dims, std::size_t V_dims, std::size_t heads, bool mask,
        bool need_grads = true, bool qkv_bias=false, bool proj_bias=false, Device device=Device::CPU, Dtype=Dtype::float32,
        Initializers initializer=Initializers::he_normal);
    SelfAttention() {m_cnt=m_tracker.m_count; ++m_tracker.m_count;}
    Tensor operator()(const Tensor& input);
    void createMask(std::size_t seq_len);
    auto& mask() const {return m_mask_;}
    auto& linear() {return m_linear;}
    std::vector<Parameter> parameters();
    [[nodiscard]] auto& query() {return m_query_W;}
    [[nodiscard]] auto& key() {return m_key_W;}
    [[nodiscard]] auto& value() {return m_value_W;}
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
    [[nodiscard]] auto need_grads() const {return m_need_grads;}
    [[nodiscard]] auto qkv_bias() const {return m_qkv_bias;}
    [[nodiscard]] auto proj_bias() const {return m_proj_bias;}
    [[nodiscard]] auto& query_bias() {return m_Q_bias;}
    [[nodiscard]] auto& key_bias() {return m_K_bias;}
    [[nodiscard]] auto& value_bias() {return m_V_bias;}
};