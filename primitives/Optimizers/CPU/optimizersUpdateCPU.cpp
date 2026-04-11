#include "optimizersUpdateCPU.h"

void Forge::SGDUpdateCPU::update(const std::vector<Tensor *> &params, const float lr) const {
    using grads_t = float;
    const auto dtype {params[0]->dtype()};
    DISPATCH_ALL_TYPES(dtype, Device::CPU, [&] {
        for (auto p : params) {
            auto p_map {p->as_eigen<scalar_t>()};
            auto p_grads_map {p->gradients().as_eigen<grads_t>().template cast<scalar_t>()};
            auto learning_rate {static_cast<scalar_t>(lr)};
            p_map -= learning_rate * p_grads_map;
        }
    });
}
