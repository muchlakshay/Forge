#pragma once
#include <vector>
#include "tensor.h"
#include "../parameter.h"

namespace Forge {
    class SGD;
    class Adam;
}

class Forge::SGD {
    std::vector<Parameter> m_parameters{};
    std::vector<Tensor> m_V;
    float m_learningRate {};
    float m_momentum_coef {};
public:
    explicit SGD(std::vector<Parameter> parameters, float lr=0.01f, float momentum_coef=0.0f);
    void update() const;
    void clear_grads() const {for (const auto p : m_parameters) p.m_param_ptr->clear_grads();};
    void setLearningRate(const float lr) {m_learningRate = lr;}
    void setMomentumCoef(const float momentum_coef) {m_momentum_coef = momentum_coef;}
    [[nodiscard]] const auto& parameters() const {return m_parameters;}
    [[nodiscard]] auto learningRate() const {return m_learningRate;}
    [[nodiscard]] auto momentumCoef() const {return m_momentum_coef;}
};

class Forge::Adam {
    std::vector<Parameter> m_parameters{};
    std::vector<Tensor> m_V{};
    std::vector<Tensor> m_M{};
    float m_learningRate {};
    mutable int m_epoch{1};
    float m_beta_1 {}, m_beta_2 {};
public:
    explicit  Adam(const std::vector<Parameter>& parameters, float lr=0.01f, float beta_1=0.9f, float beta_2=0.999f);
    void clear_grads() const {for (const auto p : m_parameters) p.m_param_ptr->clear_grads();}
    void reset() const {m_epoch = 1;}
    void update() const;
    void setLearningRate(const float lr) {m_learningRate = lr;}
    void setBeta_1(const float beta_1) {m_beta_1 = beta_1;}
    void setBeta_2(const float beta_2) {m_beta_2 = beta_2;}
    [[nodiscard]] const auto& parameters() const {return m_parameters;}
    [[nodiscard]] const auto& firstMoment() const {return m_M;}
    [[nodiscard]] const auto& secondMoment() const {return m_M;}
    [[nodiscard]] auto beta_1() const {return m_beta_1;}
    [[nodiscard]] auto beta_2() const {return m_beta_2;}
    [[nodiscard]] auto learningRate() const {return m_learningRate;}
    [[nodiscard]] auto epoch() const {return m_epoch;}
};