#include "LinearLayer.h"
#include "autograd/attach_node.h"
#include "LinearGradsAbstract.h"
#include "LinearAbstract.h"

Forge::Linear::Linear(std::size_t input_size, std::size_t output_size, bool need_grads, bool bias, Initializers initializer, Dtype dtype,
Device device)
: m_dispatch_key{device==Device::CPU?DispatchKey::CPU:DispatchKey::CUDA}, m_input_size{input_size},
m_output_size{output_size}, m_using_bias{bias}, m_dtype{dtype}, m_device{device}, m_need_grads{need_grads}{
    m_cnt = m_tracker.m_count;
    ++m_tracker.m_count;
    m_weights = Tensor{{output_size, input_size}, dtype, m_need_grads, device};
    if (bias) m_bias = Tensor::Zeros({1, output_size}, m_need_grads, dtype, device);
    static const auto* weight_initializer {primitive_dispatcher().lookup<WeightsInitAbstract>(
       primitive_ops::weightsInit, DispatchKey::CPU)};
    weight_initializer->initialize(m_weights, initializer);
}


Forge::Tensor Forge::Linear::operator()(const Tensor &input) const {
    if (input.dtype()!=m_dtype) throw std::invalid_argument("input tensor and Layer weights have different dtypes");
    if (auto inp_cols{input.shape().back()}; inp_cols != m_input_size) throw std::invalid_argument(
        std::format("invalid input tensor shape. Expected: {}, Got: {}", m_input_size, inp_cols));

    const auto* forward_fn {primitive_dispatcher().lookup<LinearAbstract>(primitive_ops::linear, m_dispatch_key)};
    auto output_shape {input.shape()};
    output_shape.back() = m_output_size;

    bool need_grads {input.need_grads() || weights().need_grads() || (m_using_bias && m_bias.need_grads())};
    Tensor output {{output_shape}, m_dtype, need_grads, m_device};

    auto start = std::chrono::steady_clock::now();
    forward_fn->forward(input, output, m_weights, m_bias, m_using_bias);
    auto end = std::chrono::steady_clock::now();
    linear_time+=std::chrono::duration<float>{end - start}.count();

    attach_node<LinearGradsAbstract, 3>(output, m_device, primitive_dispatcher(), primitive_ops::linear,
        input, m_weights, m_bias);

    return output;
}