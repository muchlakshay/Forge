#pragma once
#include "lossFunctionsBase.h"

namespace Forge {
    class MSE;
    class CrossEntropy;
    class BinaryCrossEntropy;
}

class Forge::MSE final : LossFunction {
    Tensor m_loss;
public:
    const Tensor& operator()(const Tensor& predictions, const Tensor& ground_truth) override;
    void backward() const override {if (m_loss.need_grads()) m_loss.backward(); }
};

class Forge::CrossEntropy final : LossFunction{
    Tensor m_loss;
public:
    const Tensor& operator()(const Tensor& predictions, const Tensor& ground_truth) override;
    void backward() const override {if (m_loss.need_grads()) m_loss.backward(); }
};

class Forge::BinaryCrossEntropy final : LossFunction{
    Tensor m_loss;
public:
    const Tensor& operator()(const Tensor& predictions, const Tensor& ground_truth) override;
    void backward() const override {if (m_loss.need_grads()) m_loss.backward(); }
};
