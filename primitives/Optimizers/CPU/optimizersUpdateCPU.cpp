#include "optimizersUpdateCPU.h"

void Forge::SGDUpdateCPU::update(const std::vector<Tensor *> &params, const float lr,
    float momentum_coef, const std::vector<Tensor>& V) const {
    using grads_t = float;
    const auto dtype {params[0]->dtype()};
    DISPATCH_ALL_TYPES(dtype, Device::CPU, [&] {
        for (std::size_t i {}; i<params.size(); ++i) {
            auto* param {params[i]};

            auto p_map {param->as_eigen<scalar_t>()};
            auto p_grads_map {param->gradients().as_eigen<grads_t>().template cast<scalar_t>()};
            auto learning_rate {static_cast<scalar_t>(lr)};

            if (momentum_coef>0.0f) {
                auto m_c {static_cast<scalar_t>(momentum_coef)};
                auto p_V_map {V[i].as_eigen<scalar_t>()};
                p_V_map = (p_V_map * m_c + p_grads_map).eval();
                p_map -= learning_rate * p_V_map;
            } else
                p_map -= learning_rate * p_grads_map;
        }
    });
}
