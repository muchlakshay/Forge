#pragma once
#include <vector>
#include "tensor.h"

namespace Forge {
    class SGD;
    class Adam;
}

class Forge::SGD {
    std::vector<Tensor*> m_parameters{};
    std::vector<Tensor> m_V;
    float m_learningRate {};
    float m_momentum_coef {};
public:
    explicit SGD(std::vector<Tensor *> parameters, float lr=0.01f, float momentum_coef=0.0f);
    void update() const;
    void clear_grads() const {for (const auto p : m_parameters) p->clear_grads();};
    void setLearningRate(const float lr) {m_learningRate = lr;}
    void setMomentumCoef(const float momentum_coef) {m_momentum_coef = momentum_coef;}
    [[nodiscard]] const auto& parameters() const {return m_parameters;}
    [[nodiscard]] auto learningRate() const {return m_learningRate;}
    [[nodiscard]] auto momentumCoef() const {return m_momentum_coef;}
};