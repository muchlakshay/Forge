#pragma once
#include <vector>
#include "tensor.h"

namespace Forge {
    class SGD;
}

class Forge::SGD {
  const std::vector<Tensor*> m_parameters{};
  float m_learningRate {};
public:
  explicit SGD(const std::vector<Tensor*>& parameters, const float lr) : m_parameters(parameters), m_learningRate{lr} {}
    void update();
    void clear_grads();
    auto parameters() const {return m_parameters;}
    auto learningRate() const {return m_learningRate;}
};
