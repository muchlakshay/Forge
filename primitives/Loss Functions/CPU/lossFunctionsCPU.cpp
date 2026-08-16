#include "lossFunctionsCPU.h"
#include "execution_ctx.h"
#include <iostream>
#include "ops/Maths/CPU/max.h"

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
    Tensor log_softmax_eval {{pred.shape()}, pred.dtype(), false, pred.device()};
    Tensor row_max_eval {{pred.shape()}, pred.dtype(), false, pred.device()};
    Tensor logsumexp_eval {{pred.shape()}, pred.dtype(), false, pred.device()};
    Tensor shifted_eval {pred.shape(), pred.dtype(), false, pred.device()};

    DISPATCH_ALL_TYPES(pred.dtype(), Device::CPU, [&] {
        auto pred_map {pred.as_eigen<scalar_t>()};
        auto pdims {pred_map.dimensions()};
        pred_map = pred_map.reshape(Eigen::array<Eigen::Index, 4>{1, 1, pdims[0]*pdims[1]*pdims[2], pdims[3]});

        auto ground_truth_map {ground_truth.as_eigen<scalar_t>()};

        auto log_softmax_eval_map {log_softmax_eval.as_eigen<scalar_t>()};
        log_softmax_eval_map = log_softmax_eval_map.reshape(pred_map.dimensions());

        auto row_max_eval_map {row_max_eval.as_eigen<scalar_t>()};
        row_max_eval_map = row_max_eval_map.reshape(pred_map.dimensions());

        auto logsumexp_eval_map {logsumexp_eval.as_eigen<scalar_t>()};
        logsumexp_eval_map = logsumexp_eval_map.reshape(pred_map.dimensions());

        auto& dims {pred_map.dimensions()};
        Eigen::array<Eigen::Index, 1> class_dim {3};
        Eigen::array<Eigen::Index, 4> keep_dim{dims[0], dims[1], dims[2], 1};
        Eigen::array<Eigen::Index, 4> bcast{1, 1, 1, dims[3]};
        Eigen::array<Eigen::Index, 1> batch_dim {2};

        auto r_max {forge_max_AVX2<scalar_t>(pred, pred.shape().size()-1)};
        auto row_max {r_max.as_eigen<scalar_t>().broadcast(bcast)};
        forge_eval(row_max_eval_map, row_max);

        auto shifted_eval_map {shifted_eval.as_eigen<scalar_t>()};
        auto shifted {pred_map - row_max_eval_map};
        forge_eval(shifted_eval_map, shifted);
        // forge_exp_AVX2(static_cast<scalar_t>(shifted_eval.data()), shifted_eval_map.data());

        // auto start_out = std::chrono::steady_clock::now();
        auto logsumexp {shifted_eval_map.exp().sum(class_dim).log().reshape(keep_dim).broadcast(bcast)};
        forge_eval(logsumexp_eval_map, logsumexp);

        // auto end_out = std::chrono::steady_clock::now();
        forge_eval(log_softmax_eval_map, (shifted-logsumexp_eval_map));

        // auto start_in = std::chrono::steady_clock::now();
        if (ground_truth.dtype() == Dtype::int32 && ground_truth.shape().size()==(pred.shape().size()-1)) {

            auto idx_map {ground_truth.as_eigen<int>()};
            idx_map =  idx_map.reshape(Eigen::array<Eigen::Index, 4>{1, 1, 1, idx_map.size()});
            scalar_t NLL {};
            for (std::size_t B {}; B<idx_map.size(); B++) {
                NLL = NLL - log_softmax_eval_map(0, 0, B, idx_map(B));
            }

            Eigen::Tensor<scalar_t, 0, Eigen::RowMajor> NLL_val {};
            NLL_val.setValues({NLL});
            loss_map(0, 0, 0, 0) = static_cast<float>(NLL_val()) / static_cast<float>(idx_map.size());
            auto end_in = std::chrono::steady_clock::now();
            // std::cout<<"out: "<<(end_out-start_out).count()<<
            //     "\nin: "<<(end_in-start_in).count()<<"\n";
            return;
        }

        ground_truth_map = ground_truth_map.reshape(pred_map.dimensions());

        auto NLL {-(ground_truth_map * log_softmax_eval_map).sum()};
        Eigen::Tensor<scalar_t, 0, Eigen::RowMajor> NLL_val {NLL};
        loss_map(0, 0, 0, 0) = static_cast<float>(NLL_val()) / static_cast<float>(dims[batch_dim[0]]);
    });
}

void Forge::BinaryCrossEntropyCPU::compute_loss(const Tensor &pred, const Tensor &ground_truth, Tensor &loss) const {
    auto loss_map {loss.as_eigen<float>()};
    DISPATCH_ALL_TYPES(pred.dtype(), Device::CPU, [&] {
        auto pred_map {pred.as_eigen<scalar_t>()};
        auto ground_truth_map {ground_truth.as_eigen<scalar_t>()};
        auto zero {static_cast<scalar_t>(0)};
        auto one {pred_map.constant(static_cast<scalar_t>(1))};

        auto max_term {pred_map.cwiseMax(zero)};
        auto log_term {(one + (-pred_map.abs()).exp()).log()};
        auto BCE {max_term - pred_map*ground_truth_map+log_term};
        Eigen::Tensor<scalar_t, 0, Eigen::RowMajor> BCE_val {BCE.sum()};
        loss_map(0, 0, 0, 0) = {static_cast<float>(BCE_val())/static_cast<float>(pred_map.dimensions()[2])};

    });
}

