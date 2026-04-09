#pragma once
#include <vector>
#include "tensor.h"

namespace Forge {
    class SGD;
}

class Forge::SGD {
    std::vector<Tensor*> m_parameters{};
    float m_learningRate {};
    Device m_device{};
    Dtype m_dtype{};
public:
  explicit SGD(std::vector<Tensor*> parameters, const float lr=0.001f);
    void update();
    void clear_grads();
    auto parameters() const {return m_parameters;}
    auto learningRate() const {return m_learningRate;}
};
