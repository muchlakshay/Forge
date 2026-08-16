#include "layernorm_cpu.h"

#include "tensor.h"
#include "ops/Dispatcher/dispatcher.h"

void Forge::LayerNormImplCPU::forward(const Tensor &input, const Tensor &gamma, const Tensor &beta, Tensor &opt) const {
    DISPATCH_ALL_TYPES(input.dtype(), input.device(), [&] {
        auto input_map {input.as_eigen<scalar_t, 3>()};
        auto gamma_map {gamma.as_eigen<scalar_t, 3>()};
        auto beta_map  { beta.as_eigen<scalar_t, 3>()};
        auto opt_map   { opt.as_eigen<scalar_t,  3>()};

        auto& inps {input_map.dimensions()};
        auto batch {inps[0]}, seq_len {inps[1]}, d_model {inps[2]};

        Eigen::array<Eigen::Index, 1> reduction_dim {2};
        Eigen::array<Eigen::Index, 3> reshape_dims {batch, seq_len, 1};

        auto row_mean {input_map.mean(reduction_dim).reshape(reshape_dims)};

        Eigen::array<Eigen::Index, 3> bcast_dims {1, 1, d_model};
        auto inp_sub_mean {input_map-row_mean.broadcast(bcast_dims)};
        auto row_variance {inp_sub_mean.square().mean(reduction_dim).reshape(reshape_dims)};

        auto eps {static_cast<scalar_t>(1e-5)};
        auto rstd {(row_variance+row_variance.constant(eps)).rsqrt()};
        auto normalized {inp_sub_mean * rstd.broadcast(bcast_dims)};

        Eigen::array<Eigen::Index, 3> bcast_dims_ {batch, seq_len, 1};

        forge_eval(opt_map, gamma_map.broadcast(bcast_dims_) * normalized + beta_map.broadcast(bcast_dims_));
    });
}
