#pragma once
#include <vector>
#include "tensor.h"
#include "../parameter.h"

namespace Forge {
    class SGD;
    class Adam;
    inline void check_in_range(const float x, float l, float u, std::string_view name) {
        if (x<l || x>u) throw std::invalid_argument(std::format("{} must be in range[{}, {}]", name, l, u));
    }
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
    void setMomentumCoef(const float momentum_coef) { check_in_range(momentum_coef, 0.f, 1.f, "Momentum Coefficient"); m_momentum_coef = momentum_coef;}
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
    float m_decayFactor {};
public:
    explicit  Adam(const std::vector<Parameter>& parameters, float lr=0.01f, float beta_1=0.9f, float beta_2=0.999f,
        float decay_factor=0.01f);
    void clear_grads() const {for (const auto p : m_parameters) p.m_param_ptr->clear_grads();}
    void reset() const {m_epoch = 1;}
    void update() const;
    void setLearningRate(const float lr) {m_learningRate = lr;}
    void setBeta_1(const float beta_1) {check_in_range(beta_1, 0.f, 1.f, "Beta 1"); m_beta_1 = beta_1;}
    void setBeta_2(const float beta_2) {check_in_range(beta_2, 0.f, 1.f, "Beta 2"); m_beta_2 = beta_2;}
    void setDecayFactor(const float decay_rate) {check_in_range(decay_rate, 0.f, 1.f, "Decay rate"); m_decayFactor = decay_rate;}
    [[nodiscard]] const auto& parameters() const {return m_parameters;}
    [[nodiscard]] const auto& firstMoment() const {return m_M;}
    [[nodiscard]] const auto& secondMoment() const {return m_M;}
    [[nodiscard]] auto beta_1() const {return m_beta_1;}
    [[nodiscard]] auto beta_2() const {return m_beta_2;}
    [[nodiscard]] auto learningRate() const {return m_learningRate;}
    [[nodiscard]] auto epoch() const {return m_epoch;}
    [[nodiscard]] auto decayFactor() const {return m_decayFactor;}
};