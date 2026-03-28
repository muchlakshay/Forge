#pragma once
#include "tensor.h"

namespace Forge {
    struct Relu;
    struct Softmax;
    struct LeakyRelu;
    struct Tanh;
    struct Gelu;
    struct Sigmoid;
};

struct Forge::Relu {Tensor operator()(const Tensor& input) const;};
struct Forge::Softmax {Tensor operator()(const Tensor& input) const;};
struct Forge::LeakyRelu {Tensor operator()(const Tensor& input) const;};
struct Forge::Tanh {Tensor operator()(const Tensor& input) const;};
struct Forge::Gelu {Tensor operator()(const Tensor& input) const;};
struct Forge::Sigmoid {Tensor operator()(const Tensor& input) const;};