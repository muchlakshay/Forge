#pragma once
#include "tensor/tensor.h"
#include "primitive_dispatcher.h"
#include "weights_init.h"

namespace Forge {
    class Linear;
    struct LinearAbstract;
    struct LinearCPU;
}

class Forge::Linear {
    Tensor m_weights;
    Tensor m_bias;
    DispatchKey m_dispatch_key;
    std::size_t m_input_size, m_output_size;
    bool m_using_bias;
    Dtype m_dtype;
    Device m_device;
public:
    Linear(std::size_t input_size, std::size_t output_size, Initializers initializer=Initializers::xavier_normal,
        Dtype dtype=Dtype::float32, Device device=Device::CPU, bool bias=true);
    Tensor operator()(const Tensor& input) const;
    [[nodiscard]] const auto& weights() const { return m_weights; }
    [[nodiscard]] const auto& bias() const { return m_bias; }
    [[nodiscard]] const auto& input_size() const { return m_input_size; }
    [[nodiscard]] const auto& output_size() const { return m_output_size; }
    [[nodiscard]] const auto& dispatch_key() const { return m_dispatch_key; }
    [[nodiscard]] const auto& device() const { return m_device; }
    [[nodiscard]] const auto& dtype() const { return m_dtype; }
};

inline Forge::Linear::Linear(std::size_t input_size, std::size_t output_size, Initializers initializer, Dtype dtype,
    Device device, bool bias)
    : m_dispatch_key{device==Device::CPU?DispatchKey::CPU:DispatchKey::CUDA}, m_input_size{input_size},
    m_output_size{output_size}, m_using_bias{bias}, m_dtype{dtype}, m_device{device}{

    m_weights = Tensor{{output_size, input_size}, dtype, true, device};
    if (bias) m_bias = Tensor::Zeros({1, output_size}, true, dtype, device);
    static const auto* weight_initializer {primitive_dispatcher().lookup<WeightsInitAbstract>(
       primitive_ops::weightsInit, DispatchKey::CPU)};
    weight_initializer->initialize(m_weights, input_size, output_size, initializer);
}


inline Forge::Tensor Forge::Linear::operator()(const Tensor &input) const {
    if (input.dtype()!=m_dtype) throw std::invalid_argument("input tensor and Layer weights have different dtypes");
    if (auto inp_cols{input.shape()[input.size()-1]}; inp_cols == m_input_size) throw std::invalid_argument(
        std::format("invalid input tensor shape. Expected: {}, Got: {}", m_input_size, inp_cols));

    const auto* forward_fn {primitive_dispatcher().lookup<LinearAbstract>(primitive_ops::linear, m_dispatch_key)};
    auto output_shape {input.shape()};
    output_shape.back() = m_output_size;
    Tensor output {{output_shape}, m_dtype, input.need_grads(), m_device};
    forward_fn->forward(input, output, m_weights, m_bias, m_using_bias);
    return output;
}