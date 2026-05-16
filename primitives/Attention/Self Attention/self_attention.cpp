#include "self_attention.h"

#include "self_attention_impl_abstract.h"
#include "Dispatcher/primitive_dispatcher.h"
#include "Weights Init/weights_init.h"
#include "mask_abstract.h"

Forge::SelfAttention::SelfAttention(std::size_t d_model, std::size_t Q_K_dims, std::size_t V_dims, std::size_t heads,
    bool mask, Device device, Dtype dtype, Initializers initializer) : m_query_W {{heads, d_model, Q_K_dims}, dtype, true, device},
    m_key_W{{heads, d_model, Q_K_dims}, dtype, true, device}, m_value_W {{heads, d_model, V_dims}, dtype, true, device},
    m_linear{V_dims * heads, d_model, initializer, dtype, device, false},
    m_mask{mask}, m_dispatch_key{device==Device::CPU?DispatchKey::CPU:DispatchKey::CUDA}, m_heads{heads}, m_d_model{d_model},
    m_device {device}, m_dtype{dtype}{

    const auto* init {primitive_dispatcher().lookup<WeightsInitAbstract>(primitive_ops::weightsInit, m_dispatch_key)};
    init->initialize(m_query_W, m_query_W.shape(), initializer);
    init->initialize(m_key_W,   m_key_W.shape(),   initializer);
    init->initialize(m_value_W, m_value_W.shape(), initializer);
}

Forge::Tensor Forge::SelfAttention::operator()(const Tensor &input) const {
    if (input.shape().size()>3) throw std::invalid_argument("input tensor must be rank 3 at max [batch, seq_len, d_model]");
    if (m_mask && m_mask_.shape().empty()) throw std::runtime_error("create a mask prior to forward pass");
    if (auto seq_len {input.shape()[0]}, mask_seq_len {m_mask_.shape()[0]}; m_mask_.shape()[0]!=seq_len)
        throw std::invalid_argument(
        std::format("mask with shape ({}, {}) is needed but got of shape ({}, {})", seq_len, seq_len, mask_seq_len, mask_seq_len)
    );
    Tensor output {};
    const auto* impl {primitive_dispatcher().lookup<SelfAttentionImplAbstract>(primitive_ops::selfAttention, m_dispatch_key)};
    impl->forward(input, m_query_W, m_key_W, m_value_W, output, m_mask_, m_linear, m_heads, m_d_model, m_mask);
    return output;
}

void Forge::SelfAttention::createMask(std::size_t seq_len) {
    if (!m_mask) throw std::runtime_error("Cant create a mask, as this SelfAttention instance doesnt require one (mask=false)");
    auto M {Tensor::Zeros({seq_len, seq_len}, false, m_dtype, m_device)};
    const auto* mask_gen {primitive_dispatcher().lookup<MaskAbstract>(primitive_ops::SA_mask, M.dispatch_key())};
    mask_gen->generate(M, seq_len);
    m_mask_ = M;
}

