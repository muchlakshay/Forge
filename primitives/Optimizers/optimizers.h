#pragma once
#include <vector>
#include "tensor.h"

namespace Forge {
    class SGD;
}

class Forge::SGD {
    std::vector<Tensor*> m_parameters{};
    float m_learningRate {};
public:
    explicit SGD(std::vector<Tensor *> parameters, const float lr) : m_parameters{std::move(parameters)}, m_learningRate {lr}{}
    void update() const;
    void clear_grads() const {for (const auto p : m_parameters) p->clear_grads();};
    [[nodiscard]] auto parameters() const {return m_parameters;}
    [[nodiscard]] auto learningRate() const {return m_learningRate;}
};
