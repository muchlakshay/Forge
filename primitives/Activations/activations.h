#pragma once
#include "tensor.h"
#include "activationsAbstarct.h"
#include "../Dispatcher/primitive_dispatcher.h"

namespace Forge {
    class Relu;
    class Softmax;
    class LeakyRelu;
    class Tanh;
    class Gelu;
    class Sigmoid;
};

class Forge::Relu {
    Tensor operator()(const Tensor& input) const {
        Tensor output {input.shape(), input.dtype(), input.need_grads(), input.device()};
        const auto* relu {primitive_dispatcher().lookup<ReluAbstract>(primitive_ops::relu, input.dispatch_key())};
        relu->forward(input, output);
        return output;
    }
};

class Forge::Softmax {
    Tensor operator()(const Tensor& input) const {
        Tensor output {input.shape(), input.dtype(), input.need_grads(), input.device()};
        const auto* softmax {primitive_dispatcher().lookup<SoftmaxAbstract>(primitive_ops::softmax, input.dispatch_key())};
        softmax->forward(input, output);
        return output;
    }
};

class Forge::LeakyRelu {
    Tensor operator()(const Tensor& input) const {
        Tensor output {input.shape(), input.dtype(), input.need_grads(), input.device()};
        const auto* leaky_relu {primitive_dispatcher().lookup<LeakyReluAbstract>(primitive_ops::leaky_relu, input.dispatch_key())};
        leaky_relu->forward(input, output);
        return output;
    }
};

class Forge::Tanh {
    Tensor operator()(const Tensor& input) const {
        Tensor output {input.shape(), input.dtype(), input.need_grads(), input.device()};
        const auto* tanh {primitive_dispatcher().lookup<TanhAbstract>(primitive_ops::tanh, input.dispatch_key())};
        tanh->forward(input, output);
        return output;
    }
};

class Forge::Gelu {
    Tensor operator()(const Tensor& input) const {
        Tensor output {input.shape(), input.dtype(), input.need_grads(), input.device()};
        const auto* gelu {primitive_dispatcher().lookup<GeluAbstract>(primitive_ops::gelu, input.dispatch_key())};
        gelu->forward(input, output);
        return output;
    }
};