#include "lossFunctionsCPU.h"
#include <iostream>

void Forge::MSE_CPU::compute_loss(const Tensor &pred, const Tensor &ground_truth, Tensor &loss) const {
    DISPATCH_ALL_TYPES(pred.dtype(), Device::CPU, [&] {
        auto pred_map_f          {pred.as_eigen<scalar_t>()};
        auto ground_truth_map_f  {ground_truth.as_eigen<scalar_t>()};

        Eigen::Tensor<scalar_t, 0, Eigen::RowMajor> sum_val { (ground_truth_map_f - pred_map_f).square().sum()};
        loss.as_eigen<float>()(0, 0, 0, 0) = (static_cast<float>(sum_val())/static_cast<float>(pred.size()));
        });
}


