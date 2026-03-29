#pragma once
#include "lossFunctionsBase.h"

namespace Forge {
    class MSE;
    class CrossEntropy;
    class BinaryCrossEntropy;
}

class Forge::MSE final : LossFunction{
    const Tensor* m_loss {};
public:
    Tensor operator()(const Tensor& predictions, const Tensor& ground_truth) override;
    void backward() const override;
};

class Forge::CrossEntropy final : LossFunction{
    const Tensor* m_loss {};
public:
    Tensor operator()(const Tensor& predictions, const Tensor& ground_truth) override;
    void backward() const override;
};

class Forge::BinaryCrossEntropy final : LossFunction{
    const Tensor* m_loss {};
public:
    Tensor operator()(const Tensor& predictions, const Tensor& ground_truth) override;
    void backward() const override;
};
