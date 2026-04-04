#include "lossFunctionsCPU.h"

void Forge::MSE_CPU::compute_loss(const Tensor &pred, const Tensor &ground_truth, Tensor &loss) const {
    auto loss_map {loss.as_eigen<float>()};
    DISPATCH_ALL_TYPES(pred.dtype(), Device::CPU, [&] {
        auto pred_map {pred.as_eigen<scalar_t>()};
        auto ground_truth_map {ground_truth.as_eigen<scalar_t>()};
        loss_map = ((ground_truth_map-pred_map).square().sum()/pred_map.size()).template cast<float>;
    });
}


