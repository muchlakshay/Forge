#pragma once
#include "tensor.h"

namespace Forge {
    struct LossFunction;
}

struct Forge::LossFunction {
    virtual Tensor operator()(const Tensor& predictions, const Tensor& ground_truth)= 0;
    virtual void backward() const = 0;
    virtual ~LossFunction() = default;
};
