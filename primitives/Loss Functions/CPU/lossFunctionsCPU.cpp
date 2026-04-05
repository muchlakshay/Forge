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

void Forge::CrossEntropyCPU::compute_loss(const Tensor &pred, const Tensor &ground_truth, Tensor &loss) const {
    auto loss_map {loss.as_eigen<float>()};
    DISPATCH_ALL_TYPES(pred.dtype(), Device::CPU, [&] {
        auto pred_map {pred.as_eigen<scalar_t>()};
        auto ground_truth_map {ground_truth.as_eigen<scalar_t>()};

        auto& dims {pred_map.dimensions()};
        Eigen::array<Eigen::Index, 1> class_dim {3};
        Eigen::array<Eigen::Index, 4> keep_dim{dims[0], dims[1], dims[2], 1};
        Eigen::array<Eigen::Index, 4> bcast{1, 1, 1, dims[3]};
        Eigen::array<Eigen::Index, 1> batch_dim {2};

        auto row_max {pred_map.maximum(class_dim).reshape(keep_dim).broadcast(bcast)};
        auto shifted {pred_map - row_max};
        auto logsumexp {shifted.exp().sum(class_dim).log().reshape(keep_dim).broadcast(bcast)};
        auto log_softmax {shifted-logsumexp};

        auto NLL {-(ground_truth_map * log_softmax).sum()};
        Eigen::Tensor<scalar_t, 0, Eigen::RowMajor> NLL_val {NLL};
        loss_map(0, 0, 0, 0) = static_cast<float>(NLL_val()) / static_cast<float>(dims[batch_dim[0]]);
    });
}


