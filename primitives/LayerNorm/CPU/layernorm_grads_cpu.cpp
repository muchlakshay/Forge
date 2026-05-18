#include "layernorm_grads_cpu.h"
#include "tensor.h"

void Forge::LayerNormGradsCPU::compute_grads(const Tensor &input, const Tensor &gamma, const Tensor &beta,
    const Tensor &opt) const {
    using grads_t = float;

    DISPATCH_ALL_TYPES(input.dtype(), input.device(), [&] {
        auto input_map {input.as_eigen<scalar_t, 3>()};
        auto gamma_map {gamma.as_eigen<scalar_t, 3>()};

        auto input_grads_map {input.gradients().as_eigen<grads_t, 3>()};
        auto opt_grads_map {opt.gradients().as_eigen<grads_t, 3>()};
        auto gamma_grads_map {gamma.gradients().as_eigen<grads_t, 3>()};
        auto beta_grads_map {beta.gradients().as_eigen<grads_t, 3>()};

        auto& inps {input_map.dimensions()};
        auto batch {inps[0]}, seq_len {inps[1]}, d_model {inps[2]};

        Eigen::array<Eigen::Index, 3> reshape_dims {batch, seq_len, 1};
        Eigen::array<Eigen::Index, 1> reduction_dim {2};
        Eigen::array<Eigen::Index, 3> bcast_dims {1, 1, d_model};
        Eigen::array<Eigen::Index, 2> reduction_dims {0, 1};
        Eigen::array<Eigen::Index, 3> bcast_dims_ {batch, seq_len, 1};

        auto row_mean {input_map.mean(reduction_dim).reshape(reshape_dims)};

        auto inp_sub_mean {input_map-row_mean.broadcast(bcast_dims)};
        auto row_variance {inp_sub_mean.square().mean(reduction_dim).reshape(reshape_dims)};

        auto eps {static_cast<scalar_t>(1e-5)};
        auto rstd {(row_variance+row_variance.constant(eps)).rsqrt()};
        auto normalized {inp_sub_mean * rstd.broadcast(bcast_dims)};


        gamma_grads_map+= (opt_grads_map * normalized.template cast<grads_t>()).sum(reduction_dims).reshape(bcast_dims);
        beta_grads_map += opt_grads_map.sum(reduction_dims);

        auto normalized_inp_grads {opt_grads_map * gamma_map.template cast<grads_t>().broadcast(bcast_dims_)};
        auto one_by_dmodel {normalized_inp_grads.constant(
            static_cast<grads_t>(1)/static_cast<grads_t>(d_model))};

        auto normalized_inp_grads_sum {normalized_inp_grads.sum(reduction_dim).reshape(reshape_dims).broadcast(bcast_dims)};
        auto rstd_casted {rstd.template cast<grads_t>().broadcast(bcast_dims)};
        auto d_model_tensor {normalized_inp_grads.constant(static_cast<grads_t>(d_model))};
        auto normalized_casted {normalized.template cast<grads_t>()};

        input_grads_map += one_by_dmodel * rstd_casted * ((d_model_tensor * normalized_inp_grads) - normalized_inp_grads_sum - (
            (normalized_inp_grads * normalized_casted).sum(reduction_dim).reshape(reshape_dims).broadcast(bcast_dims) * normalized_casted));
    });
}

// opt_grads (batch, seq_len, d_model) , normalized (batch, seq_len, d_model)