#pragma once
#include "tensor/tensor.h"
#include "../Dispatcher/primitive_dispatcher.h"
#include "../Weights Init/weights_init.h"
#include "parameter.h"

namespace Forge {
    class Linear;
    struct LinearCPU;
}

class Forge::Linear {
    Tensor m_weights{};
    Tensor m_bias{};
    DispatchKey m_dispatch_key;
    std::size_t m_input_size, m_output_size;
    bool m_using_bias;
    Dtype m_dtype;
    Device m_device;
public:
    Linear(std::size_t input_size, std::size_t output_size, Initializers initializer=Initializers::xavier_normal,
        Dtype dtype=Dtype::float32, Device device=Device::CPU, bool bias=true);
    Linear() = default;
    Tensor operator()(const Tensor& input) const;
    auto parameters() {return std::vector{Parameter{&m_weights, true}, Parameter{&m_bias, false}};}
    [[nodiscard]] const auto& weights() const { return m_weights; }
    [[nodiscard]] const auto& bias() const { return m_bias; }
    [[nodiscard]] const auto& input_size() const { return m_input_size; }
    [[nodiscard]] const auto& output_size() const { return m_output_size; }
    [[nodiscard]] const auto& dispatch_key() const { return m_dispatch_key; }
    [[nodiscard]] const auto& device() const { return m_device; }
    [[nodiscard]] const auto& dtype() const { return m_dtype; }
    [[nodiscard]] auto& weights() {return m_weights;}
    [[nodiscard]] auto& bias() { return m_bias; }
};
