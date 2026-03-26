#pragma once
#include "tensor.h"
#include "activationsAbstarct.h"
#include "acvitvationsGradsAbstract.h"
#include "../Dispatcher/primitive_dispatcher.h"
#include "autograd/attach_node.h"

namespace Forge {
    struct Relu;
    struct Softmax;
    struct LeakyRelu;
    struct Tanh;
    struct Gelu;
    struct Sigmoid;
};

struct Forge::Relu {
    Tensor operator()(const Tensor& input) const {
        Tensor output {input.shape(), input.dtype(), input.need_grads(), input.device()};
        const auto* relu {primitive_dispatcher().lookup<ReluAbstract>(primitive_ops::relu, input.dispatch_key())};
        relu->forward(input, output);
        if (output.need_grads()) {
            attach_node<ReluGradsAbstract, 1>(output, output.device(), primitive_dispatcher(), primitive_ops::relu, input);
        }
        return output;
    }
};

struct Forge::Softmax {
    Tensor operator()(const Tensor& input) const {
        Tensor output {input.shape(), input.dtype(), input.need_grads(), input.device()};
        const auto* softmax {primitive_dispatcher().lookup<SoftmaxAbstract>(primitive_ops::softmax, input.dispatch_key())};
        softmax->forward(input, output);
        if (output.need_grads()) {
            attach_node<SoftmaxGradsAbstract, 1>(output, output.device(), primitive_dispatcher(), primitive_ops::softmax, input);
        }
        return output;
    }
};

struct Forge::LeakyRelu {
    Tensor operator()(const Tensor& input) const {
        Tensor output {input.shape(), input.dtype(), input.need_grads(), input.device()};
        const auto* leaky_relu {primitive_dispatcher().lookup<LeakyReluAbstract>(primitive_ops::leaky_relu, input.dispatch_key())};
        leaky_relu->forward(input, output);
        if (output.need_grads()) {
            attach_node<LeakyReluGradsAbstract, 1>(output, output.device(), primitive_dispatcher(), primitive_ops::leaky_relu, input);
        }
        return output;
    }
};

struct Forge::Tanh {
    Tensor operator()(const Tensor& input) const {
        Tensor output {input.shape(), input.dtype(), input.need_grads(), input.device()};
        const auto* tanh {primitive_dispatcher().lookup<TanhAbstract>(primitive_ops::tanh, input.dispatch_key())};
        tanh->forward(input, output);
        if (output.need_grads()) {
            attach_node<TanhGradsAbstract, 1>(output, output.device(), primitive_dispatcher(), primitive_ops::tanh, input);
        }
        return output;
    }
};

struct Forge::Gelu {
    Tensor operator()(const Tensor& input) const {
        Tensor output {input.shape(), input.dtype(), input.need_grads(), input.device()};
        const auto* gelu {primitive_dispatcher().lookup<GeluAbstract>(primitive_ops::gelu, input.dispatch_key())};
        gelu->forward(input, output);
        if (output.need_grads()) {
            attach_node<GeluGradsAbstract, 1>(output, output.device(), primitive_dispatcher(), primitive_ops::gelu, input);
        }
        return output;
    }
};

struct Forge::Sigmoid {
    Tensor operator()(const Tensor& input) const {
        Tensor output {input.shape(), input.dtype(), input.need_grads(), input.device()};
        const auto* sigmoid {primitive_dispatcher().lookup<SigmoidAbstract>(primitive_ops::sigmoid, input.dispatch_key())};
        sigmoid->forward(input, output);
        if (output.need_grads()) {
            attach_node<SigmoidGradsAbstract, 1>(output, output.device(), primitive_dispatcher(), primitive_ops::sigmoid, input);
        }
        return output;
    }
};