#pragma once
#include "lossFunctionsBase.h"

namespace Forge {
    struct MSE;
    struct  CrossEntropy;
    struct  BinaryCrossEntropy;
}

struct Forge::MSE final : LossFunction { Tensor operator()(
    const Tensor& predictions, const Tensor& ground_truth) override;
};

struct Forge::CrossEntropy final : LossFunction {
    Tensor operator()(const Tensor& predictions, const Tensor& ground_truth) override;
};

struct Forge::BinaryCrossEntropy final : LossFunction {
    Tensor operator()(const Tensor& predictions, const Tensor& ground_truth) override;
};
