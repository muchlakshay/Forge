#include "activations.h"
#include "activationsAbstarct.h"
#include "acvitvationsGradsAbstract.h"
#include "../Dispatcher/primitive_dispatcher.h"
#include "autograd/attach_node.h"

Forge::Tensor Forge::Relu::operator()(const Tensor& input) const {
    Tensor output {input.shape(), input.dtype(), input.need_grads(), input.device()};
    const auto* relu {primitive_dispatcher().lookup<ReluAbstract>(primitive_ops::relu, input.dispatch_key())};
    relu->forward(input, output);
    if (output.need_grads()) {
        attach_node<ReluGradsAbstract, 1>(output, output.device(), primitive_dispatcher(), primitive_ops::relu, input);
    }
    return output;
}

Forge::Tensor Forge::Softmax::operator()(const Tensor& input) const {
    Tensor output {input.shape(), input.dtype(), input.need_grads(), input.device()};
    const auto* softmax {primitive_dispatcher().lookup<SoftmaxAbstract>(primitive_ops::softmax, input.dispatch_key())};
    softmax->forward(input, output);
    if (output.need_grads()) {
        attach_node<SoftmaxGradsAbstract, 1>(output, output.device(), primitive_dispatcher(), primitive_ops::softmax, input);
    }
    return output;
}

Forge::Tensor Forge::LeakyRelu::operator()(const Tensor& input) const {
    Tensor output {input.shape(), input.dtype(), input.need_grads(), input.device()};
    const auto* leaky_relu {primitive_dispatcher().lookup<LeakyReluAbstract>(primitive_ops::leaky_relu, input.dispatch_key())};
    leaky_relu->forward(input, output);
    if (output.need_grads()) {
        attach_node<LeakyReluGradsAbstract, 1>(output, output.device(), primitive_dispatcher(), primitive_ops::leaky_relu, input);
    }
    return output;
}

Forge::Tensor Forge::Tanh::operator()(const Tensor& input) const {
    Tensor output {input.shape(), input.dtype(), input.need_grads(), input.device()};
    const auto* tanh {primitive_dispatcher().lookup<TanhAbstract>(primitive_ops::tanh, input.dispatch_key())};
    tanh->forward(input, output);
    if (output.need_grads()) {
        attach_node<TanhGradsAbstract, 1>(output, output.device(), primitive_dispatcher(), primitive_ops::tanh, input);
    }
    return output;
}


Forge::Tensor Forge::Gelu::operator()(const Tensor& input) const {
    Tensor output {input.shape(), input.dtype(), input.need_grads(), input.device()};
    const auto* gelu {primitive_dispatcher().lookup<GeluAbstract>(primitive_ops::gelu, input.dispatch_key())};
    gelu->forward(input, output);
    if (output.need_grads()) {
        attach_node<GeluGradsAbstract, 1>(output, output.device(), primitive_dispatcher(), primitive_ops::gelu, input);
    }
    return output;
}

Forge::Tensor Forge::Sigmoid::operator()(const Tensor& input) const {
    Tensor output {input.shape(), input.dtype(), input.need_grads(), input.device()};
    const auto* sigmoid {primitive_dispatcher().lookup<SigmoidAbstract>(primitive_ops::sigmoid, input.dispatch_key())};
    sigmoid->forward(input, output);
    if (output.need_grads()) {
        attach_node<SigmoidGradsAbstract, 1>(output, output.device(), primitive_dispatcher(), primitive_ops::sigmoid, input);
    }
    return output;
}