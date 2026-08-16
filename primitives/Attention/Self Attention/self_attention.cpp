#include "self_attention.h"

#include "self_attention_impl_abstract.h"
#include "Dispatcher/primitive_dispatcher.h"
#include "Weights Init/weights_init.h"
#include "mask_abstract.h"

Forge::SelfAttention::SelfAttention(std::size_t d_model, std::size_t Q_K_dims, std::size_t V_dims, std::size_t heads,
    bool mask, bool need_grads, bool qkv_bias, bool proj_bias, Device device, Dtype dtype, Initializers initializer)
    : m_query_W {{heads, d_model, Q_K_dims}, dtype, need_grads, device},
    m_key_W{{heads, d_model, Q_K_dims}, dtype, need_grads, device},
    m_value_W {{heads, d_model, V_dims}, dtype, need_grads, device},
    m_linear{V_dims * heads, d_model, need_grads, proj_bias, initializer, dtype, device},
    m_mask{mask}, m_dispatch_key{device==Device::CPU?DispatchKey::CPU:DispatchKey::CUDA},
    m_heads{heads}, m_d_model{d_model}, m_device {device}, m_dtype{dtype}, m_qkv_bias{qkv_bias},
    m_proj_bias{proj_bias}, m_need_grads{need_grads}{

    m_cnt = m_tracker.m_count;
    ++m_tracker.m_count;

    const auto* init {primitive_dispatcher().lookup<WeightsInitAbstract>(primitive_ops::weightsInit, m_dispatch_key)};
    init->initialize(m_query_W, initializer);
    init->initialize(m_key_W, initializer);
    init->initialize(m_value_W, initializer);

    if (qkv_bias) {
        m_Q_bias = Tensor{{heads, Q_K_dims}, dtype, need_grads, device};
        m_K_bias = Tensor{{heads, Q_K_dims}, dtype, need_grads, device};
        m_V_bias = Tensor{{heads, V_dims}, dtype, need_grads, device};
        init->initialize(m_Q_bias, initializer);
        init->initialize(m_K_bias, initializer);
        init->initialize(m_V_bias, initializer);
    }
}

Forge::Tensor Forge::SelfAttention::operator()(const Tensor &input) {
    if (input.shape().size()>3) throw std::invalid_argument("input tensor must be rank 3 at max [batch, seq_len, d_model]");
    if (m_mask && m_mask_.shape().empty()) throw std::runtime_error("create a mask prior to forward pass");
    if (auto seq_len {input.shape()[0]}, mask_seq_len {m_mask_.shape()[0]}; m_mask_.shape()[0]!=seq_len)
        throw std::invalid_argument(
        std::format("mask with shape ({}, {}) is needed but got of shape ({}, {})", seq_len, seq_len, mask_seq_len, mask_seq_len)
    );
    Tensor output {};
    const auto* impl {primitive_dispatcher().lookup<SelfAttentionImplAbstract>(primitive_ops::selfAttention, m_dispatch_key)};
    auto start = std::chrono::steady_clock::now();
    impl->forward(input, m_query_W, m_key_W, m_value_W, output, m_mask_, m_Q_bias, m_K_bias,
        m_V_bias, m_linear, m_heads, m_d_model, m_mask);
    auto end = std::chrono::steady_clock::now();
    attention_time+=std::chrono::duration<float>{end - start}.count();
    return output;
}

void Forge::SelfAttention::createMask(std::size_t seq_len) {
    if (!m_mask) throw std::runtime_error("Cant create a mask, as this SelfAttention instance doesnt require one (mask=false)");
    auto M {Tensor::Zeros({seq_len, seq_len}, false, m_dtype, m_device)};
    const auto* mask_gen {primitive_dispatcher().lookup<MaskAbstract>(primitive_ops::SA_mask, M.dispatch_key())};
    mask_gen->generate(M, seq_len);
    m_mask_ = M;
}

std::vector<Forge::Parameter> Forge::SelfAttention::parameters() {
    auto linear_params {m_linear.parameters()};
    for (auto& p : linear_params) p.name = std::format("attn.{}.{}", m_cnt, p.name);

    std::vector params {
        Parameter{&m_query_W, true, std::format("attn.{}.q.w", m_cnt)},
        Parameter{&m_key_W, true, std::format("attn.{}.k.w", m_cnt)},
        Parameter{&m_value_W, true, std::format("attn.{}.v.w", m_cnt)},
        linear_params[0]
    };
    if (m_proj_bias) params.push_back(linear_params[1]);
    return params;
}

