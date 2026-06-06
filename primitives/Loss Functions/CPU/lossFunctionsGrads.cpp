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

void Forge::CrossEntropyGradsCPU::compute_grads(const Tensor &pred, const Tensor &ground_truth, const Tensor &loss) const {
    using grads_t = float;
    Tensor softmax_eval {{pred.shape()}, Dtype::float32, false, pred.device()};
    Tensor row_max_eval {{pred.shape()}, Dtype::float32, false, pred.device()};

    DISPATCH_ALL_TYPES(pred.dtype(), Device::CPU, [&] {

        auto pred_map{pred.as_eigen<scalar_t>()};
        auto pred_map_f {pred_map.template cast<grads_t>()};
        auto gt_map_f {ground_truth.as_eigen<scalar_t>().template cast<grads_t>()};
        auto softmax_eval_map {softmax_eval.as_eigen<grads_t>()};

        auto pred_grads_map{pred.gradients().as_eigen<grads_t>()};
        auto loss_grads_val {loss.gradients().as_eigen<grads_t>()(0, 0, 0, 0)};
        auto row_max_eval_map {row_max_eval.as_eigen<grads_t>()};


        auto& dims {pred_map.dimensions()};
        Eigen::array<Eigen::Index, 1> class_dim {3};
        Eigen::array<Eigen::Index, 4> keep_dim  {dims[0], dims[1], dims[2], 1};
        Eigen::array<Eigen::Index, 4> bcast      {1, 1, 1, dims[3]};

        forge_eval(row_max_eval_map, (pred_map_f.maximum(class_dim).reshape(keep_dim).broadcast(bcast)));
        auto shifted{pred_map_f - row_max_eval_map};
        auto exp_s {shifted.exp()};
        forge_eval(softmax_eval_map, (exp_s / exp_s.sum(class_dim).reshape(keep_dim).broadcast(bcast)));

        auto N {static_cast<grads_t>(dims[0]*dims[1]*dims[2])};

        auto one_by_N {static_cast<grads_t>(1)/N};
        if (ground_truth.dtype() == Dtype::int32 && ground_truth.shape().size() == pred.shape().size()-1) {
            auto idx_map {ground_truth.as_eigen<int>()};
            auto one {static_cast<grads_t>(1.0f)};
            auto one_by_N {one/N};

            idx_map = idx_map.reshape(Eigen::array<Eigen::Index, 4>{1, 1, 1, idx_map.size()});
            softmax_eval_map = softmax_eval_map.reshape(
                Eigen::array<Eigen::Index, 4>{1, 1, dims[0]*dims[1]*dims[2], dims[3]});

            for (std::size_t B {}; B<idx_map.size(); ++B)  softmax_eval_map(0, 0, B, idx_map(0, 0, 0, B)) -= one;
            forge_eval(pred_grads_map, (pred_grads_map + softmax_eval_map.reshape(pred_map.dimensions()) * one_by_N * loss_grads_val));
            return;
        }

        forge_eval(pred_grads_map, (pred_grads_map + (softmax_eval_map - gt_map_f) * one_by_N * loss_grads_val));
    });
}

void Forge::BinaryCrossEntropyGradsCPU::compute_grads(const Tensor &pred, const Tensor &ground_truth, const Tensor &loss) const {
    using grads_t = float;
    DISPATCH_ALL_TYPES(pred.dtype(), Device::CPU, [&] {
        auto pred_map {pred.as_eigen<scalar_t>()};
        auto pred_map_f {pred_map.template cast<grads_t>()};
        auto pred_map_grads {pred.gradients().as_eigen<grads_t>()};
        auto gt_map_f {ground_truth.as_eigen<scalar_t>().template cast<grads_t>()};
        auto one {pred_map_f.constant(grads_t(1))};
        float loss_grads_val {loss.gradients().as_eigen<grads_t>()(0, 0, 0, 0)};

        auto sigmoid {one/(one+((-pred_map_f).exp()))};
        auto N {pred_map.dimensions()[2]};
        pred_map_grads += loss_grads_val * (sigmoid-gt_map_f).template cast<grads_t>()/static_cast<float>(N);
    });
}