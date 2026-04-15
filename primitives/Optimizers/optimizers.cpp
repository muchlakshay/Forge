#include "optimizers.h"
#include "../Dispatcher/primitive_dispatcher.h"
#include "optimizersUpdateAbstract.h"
#include <algorithm>

namespace Forge {
    void invariants_checks_rmv_duplicates(std::vector<Parameter>& params);
}
void Forge::invariants_checks_rmv_duplicates(std::vector<Parameter>& params) {
    if (params.empty()) throw std::invalid_argument("No parameters provided");
    const auto device {params[0].m_param_ptr->device()};
    const auto dtype {params[0].m_param_ptr->dtype()};
    for (const auto p : params) {
        if (p.m_param_ptr->device() != device) throw std::invalid_argument("parameters have different device");
        if (p.m_param_ptr->dtype() != dtype) {throw std::invalid_argument("parameters have different dtype");}
    };
    std::ranges::sort(params, {}, &Parameter::m_param_ptr);
    const auto it {std::ranges::unique(params, {}, &Parameter::m_param_ptr).begin()};
    params.erase(it, params.end());
}

Forge::SGD::SGD(std::vector<Parameter> parameters, const float lr, const float momentum_coef)
    : m_parameters{std::move(parameters)}, m_learningRate {lr}, m_momentum_coef {momentum_coef}{
    invariants_checks_rmv_duplicates(m_parameters);

    check_in_range(momentum_coef, 0.f, 1.f, "Momentum coefficient");
    if (momentum_coef>0.0f) {
        for (std::size_t i{}; i<m_parameters.size(); ++i) {
            const auto param {m_parameters[i].m_param_ptr};
            m_V.push_back(Tensor::Zeros(param->shape(), false, param->dtype(), param->device()));
        }
    }
}

void Forge::SGD::update() const {
    auto dispatch_key {m_parameters[0].m_param_ptr->dispatch_key()};
    const auto* update_func {primitive_dispatcher().lookup<SGDUpdateAbstract>(primitive_ops::SGD, dispatch_key)};
    update_func->update(m_parameters, m_learningRate, m_momentum_coef, m_V);
}

Forge::Adam::Adam(const std::vector<Parameter> &parameters, float lr, float beta_1, float beta_2, float decay_factor)
    : m_parameters{parameters}, m_learningRate{lr}, m_beta_1 {beta_1}, m_beta_2{beta_2}, m_decayFactor {decay_factor}{

    invariants_checks_rmv_duplicates(m_parameters);
    check_in_range(beta_1, 0.f, 1.f, "Beta 1");
    check_in_range(beta_2, 0.f, 1.f, "Beta 2");

    for (std::size_t i{}; i<m_parameters.size(); ++i) {
        const auto param {m_parameters[i].m_param_ptr};
        m_V.push_back(Tensor::Zeros(param->shape(), false, param->dtype(), param->device()));
        m_M.push_back(Tensor::Zeros(param->shape(), false, param->dtype(), param->device()));
    }
}

void Forge::Adam::update() const {
    auto dispatch_key {m_parameters[0].m_param_ptr->dispatch_key()};
    const auto* update_func {primitive_dispatcher().lookup<AdamUpdateAbstract>(primitive_ops::Adam, dispatch_key)};
    update_func->update(m_parameters, m_learningRate, m_V, m_M, m_beta_1, m_beta_2, m_decayFactor, m_epoch);
    m_epoch++;
}
