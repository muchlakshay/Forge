#pragma once
#include "tensor.h"
#include "parameter.h"

namespace Forge {
    class LayerNorm;
}

class Forge::LayerNorm {
    Tensor m_gamma;
    Tensor m_beta;
    std::size_t m_d_model;
    Dtype m_dtype;
    Device m_device;
    bool m_need_grads;
public:
    LayerNorm(const std::size_t d_model, const Dtype dtype, const Device& device, bool need_grads=true);
    LayerNorm() = default;
    Tensor operator()(const Tensor& input);
    auto& gamma() {return m_gamma;}
    auto& beta() {return m_beta;}
    auto parameters() {return std::vector{Parameter{&m_gamma, false, "ln.gamma"}, Parameter{&m_beta, false, "ln.beta"}};}
    [[nodiscard]] auto d_model() const {return m_d_model;}
    [[nodiscard]] auto d_device() const {return m_device;}
    [[nodiscard]] auto dtype() const {return m_dtype;}
};