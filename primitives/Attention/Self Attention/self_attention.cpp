#include "self_attention.h"
#include "Dispatcher/primitive_dispatcher.h"
#include "Weights Init/weights_init.h"

Forge::SelfAttention::SelfAttention(std::size_t seq_len, std::size_t Q_K_dims, std::size_t V_dims, std::size_t heads,
    bool mask, Device device, Dtype dtype, Initializers initializer) : m_query_W {{heads, seq_len, Q_K_dims}, dtype, true, device},
    m_key_W{{heads, seq_len, Q_K_dims}, dtype, true, device}, m_value_W {{heads, seq_len, V_dims}, dtype, true, device},
    m_mask{mask}, m_dispatch_key{device==Device::CPU?DispatchKey::CPU:DispatchKey::CUDA}{

    const auto* init {primitive_dispatcher().lookup<WeightsInitAbstract>(primitive_ops::weightsInit, m_dispatch_key)};
    init->initialize(m_query_W, m_query_W.shape(), initializer);
    init->initialize(m_key_W,   m_key_W.shape(),   initializer);
    init->initialize(m_value_W, m_value_W.shape(), initializer);
}

