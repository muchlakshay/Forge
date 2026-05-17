#include "layernorm.h"
#include "layernorm_abstract.h"
#include "layernorm_grads_abstract.h"
#include "autograd/attach_node.h"
#include "Dispatcher/primitive_dispatcher.h"

Forge::LayerNorm::LayerNorm(std::size_t d_model, const Dtype dtype, const Device &device, const bool need_grads) :
    m_gamma{Tensor::Ones({d_model}, need_grads, dtype, device)}, m_beta{Tensor::Zeros({d_model}, need_grads, dtype, device)},
    m_d_model{d_model}, m_device{device}, m_dtype{dtype}, m_need_grads{need_grads} {}

Forge::Tensor Forge::LayerNorm::operator()(const Tensor &input) {
    if (input.shape().back() != m_d_model) throw std::invalid_argument(std::format("Input's d_model is not {}", m_d_model));
    if (input.device() != m_device) throw std::invalid_argument("Input resides on different device");

    Tensor opt{input.shape(), dtype(), m_need_grads || input.need_grads(), m_device};

    const auto* impl {primitive_dispatcher().lookup<LayerNormImplAbstract>(primitive_ops::LayerNorm, input.dispatch_key())};
    impl->forward(input, m_gamma, m_beta, opt);

    if (input.need_grads() || gamma().need_grads() || beta().need_grads()) {
        attach_node<LayerNormGradsAbstract, 3>(opt, m_device, primitive_dispatcher(), primitive_ops::LayerNorm,
            input, m_gamma, m_beta);
    }

    return opt;
}


