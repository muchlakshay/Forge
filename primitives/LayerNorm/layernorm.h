#pragma once
#include "tensor.h"
#include "parameter.h"
#include "instance_tracker.h"

namespace Forge {
    class LayerNorm;
}

class Forge::LayerNorm {
    Tensor m_gamma;
    Tensor m_beta;
    std::size_t m_d_model;
    Dtype m_dtype;
    Device m_device;
    bool m_need_grads;
    inline static InstanceTracker m_tracker;
    int m_cnt {};
public:
    LayerNorm(const std::size_t d_model, const Dtype dtype, const Device& device, bool need_grads=true);
    LayerNorm() {m_cnt=m_tracker.m_count; ++m_tracker.m_count;}
    Tensor operator()(const Tensor& input);
    auto& gamma() {return m_gamma;}
    auto& beta() {return m_beta;}
    auto parameters() {
        return std::vector{Parameter{&m_gamma, false, std::format("ln.{}.w", m_cnt)},
            Parameter{&m_beta, false, std::format("ln.{}.b", m_cnt)}};
    }
    [[nodiscard]] auto d_model() const {return m_d_model;}
    [[nodiscard]] auto d_device() const {return m_device;}
    [[nodiscard]] auto dtype() const {return m_dtype;}
};