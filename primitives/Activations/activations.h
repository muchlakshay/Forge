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

