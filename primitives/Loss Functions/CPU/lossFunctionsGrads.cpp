#include "lossFunctionsGradsCPU.h"

void Forge::MSEGradsCPU::compute_grads(const Tensor &pred, const Tensor &ground_truth, const Tensor& loss) const {
    using grads_t = float;
    DISPATCH_ALL_TYPES(pred.dtype(), Device::CPU, [&] {
        auto pred_map {pred.as_eigen<scalar_t>()};
        auto pred_map_f {pred_map.template cast<grads_t>()};
        auto pred_grads_map {pred.gradients().as_eigen<grads_t>()};
        auto gt_map_f {ground_truth.as_eigen<scalar_t>().template cast<grads_t>()};
        auto loss_grads_val {loss.gradients().as_eigen<grads_t>()(0, 0, 0, 0)};

        auto N {static_cast<grads_t>(pred_map.size())};

        pred_grads_map += pred_map_f.constant(grads_t(2) / N)
                        * (pred_map_f - gt_map_f)
                        * loss_grads_val;
    });
}