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

void Forge::AdamUpdateCPU::update(const std::vector<Tensor *> &params, float lr, const std::vector<Tensor> &V,
    const std::vector<Tensor> &M, float beta_1, float beta_2, int epoch) const {
    using grads_t = float;
    const auto dtype {params[0]->dtype()};
    DISPATCH_ALL_TYPES(dtype, Device::CPU, [&] {
        for (std::size_t i{}; i<params.size(); ++i) {\
            auto& p {*params[i]};
            Tensor temp {p.gradients().shape(), p.dtype(), false, p.device()};
            Tensor M_hat {p.gradients().shape(), p.dtype(), false, p.device()};
            auto p_map {params[i]->as_eigen<scalar_t>()};
            auto p_grads_map {params[i]->gradients().as_eigen<grads_t>(). template cast<scalar_t>()};
            auto V_map {V[i].as_eigen<scalar_t>()};
            auto M_map {M[i].as_eigen<scalar_t>()};
            auto temp_map {temp.as_eigen<scalar_t>()};
            auto M_hat_map {M_hat.as_eigen<scalar_t>()};
            auto& V_hat_map {temp_map};
            auto beta_1_c {static_cast<scalar_t>(beta_1)};
            auto beta_2_c {static_cast<scalar_t>(beta_2)};
            auto one {static_cast<scalar_t>(1)};
            auto learning_rate {static_cast<scalar_t>(lr)};
            auto e {static_cast<scalar_t>(1e-8)};

            auto denom_1 {static_cast<scalar_t>(1.0 - std::pow(beta_1, epoch))};
            auto denom_2 {static_cast<scalar_t>(1.0 - std::pow(beta_2, epoch))};

            temp_map = beta_1_c * M_map + (one - beta_1_c) * p_grads_map;
            M_map = temp_map;
            temp_map = beta_2_c * V_map + (one - beta_2_c) * p_grads_map.square();
            V_map = temp_map;

            M_hat_map =  M_map / denom_1;
            V_hat_map = V_map / denom_2;

            p_map -= learning_rate * (M_hat_map / (V_hat_map.sqrt() + e));
        }
    });
}
