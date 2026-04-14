#include "optimizers.h"
#include "../Dispatcher/primitive_dispatcher.h"
#include "optimizersUpdateAbstract.h"
#include <algorithm>

namespace Forge {
    void invariants_checks_rmv_duplicates(std::vector<Forge::Tensor *>& params);
}
void Forge::invariants_checks_rmv_duplicates(std::vector<Forge::Tensor *>& params) {
    if (params.empty()) throw std::invalid_argument("No parameters provided");
    const auto device {params[0]->device()};
    const auto dtype {params[0]->dtype()};
    for (const auto p : params) {
        if (p->device() != device) throw std::invalid_argument("parameters have different device");
        if (p->dtype() != dtype) {throw std::invalid_argument("parameters have different dtype");}
    };
    std::ranges::sort(params);
    const auto it {std::ranges::unique(params).begin()};
    params.erase(it, params.end());
}

Forge::SGD::SGD(std::vector<Tensor *> parameters, const float lr, const float momentum_coef)
    : m_parameters{std::move(parameters)}, m_learningRate {lr}, m_momentum_coef {momentum_coef}{
    invariants_checks_rmv_duplicates(m_parameters);

    if (momentum_coef<0.0f || momentum_coef>1.0f) throw std::invalid_argument("Momentum coefficient must be in interval [0, 1]");
    if (momentum_coef>0.0f) {
        for (std::size_t i{}; i<m_parameters.size(); ++i) {
            const auto param {m_parameters[i]};
            m_V.push_back(Tensor::Zeros(param->shape(), false, param->dtype(), param->device()));
        }
    }
}

void Forge::SGD::update() const {
    auto dispatch_key {m_parameters[0]->dispatch_key()};
    const auto* update_func {primitive_dispatcher().lookup<SGDUpdateAbstract>(primitive_ops::SGD, dispatch_key)};
    update_func->update(m_parameters, m_learningRate, m_momentum_coef, m_V);
}
