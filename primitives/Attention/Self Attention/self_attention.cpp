#include "self_attention.h"

#include "self_attention_impl_abstract.h"
#include "Dispatcher/primitive_dispatcher.h"
#include "Weights Init/weights_init.h"
#include <limits>

Forge::SelfAttention::SelfAttention(std::size_t d_model, std::size_t seq_len, std::size_t Q_K_dims, std::size_t V_dims, std::size_t heads,
    bool mask, Device device, Dtype dtype, Initializers initializer) : m_query_W {{heads, d_model, Q_K_dims}, dtype, true, device},
    m_key_W{{heads, d_model, Q_K_dims}, dtype, true, device}, m_value_W {{heads, d_model, V_dims}, dtype, true, device},
    m_linear{V_dims * heads, d_model, Initializers::xavier_normal, m_query_W.dtype(), m_query_W.device(), false},
    m_mask{mask}, m_dispatch_key{device==Device::CPU?DispatchKey::CPU:DispatchKey::CUDA}, m_heads{heads}{

    const auto* init {primitive_dispatcher().lookup<WeightsInitAbstract>(primitive_ops::weightsInit, m_dispatch_key)};
    init->initialize(m_query_W, m_query_W.shape(), initializer);
    init->initialize(m_key_W,   m_key_W.shape(),   initializer);
    init->initialize(m_value_W, m_value_W.shape(), initializer);

    if (mask) {
        auto M {Tensor::Zeros({seq_len, seq_len}, false, dtype, device)};
        DISPATCH_ALL_TYPES(dtype, device, [&] {
            auto* data_ptr {static_cast<scalar_t*>(M.data())};
            for (std::size_t i {}; i<seq_len; ++i) {
                for (std::size_t j {}; j<seq_len; ++j) {if (j>=i) data_ptr[i * seq_len + j] = -std::numeric_limits<scalar_t>::infinity();}
            }
        });
        m_mask_ = M;
    }
}

Forge::Tensor Forge::SelfAttention::operator()(const Tensor &input) const {
    Tensor output {input.shape(), input.dtype(), input.need_grads(), input.device()};
    const auto* impl {primitive_dispatcher().lookup<SelfAttentionImplAbstract>(primitive_ops::selfAttention, m_dispatch_key)};
    impl->forward(input, m_query_W, m_key_W, m_value_W, output, m_mask_, m_linear, m_heads, m_mask);
    return output;
}


