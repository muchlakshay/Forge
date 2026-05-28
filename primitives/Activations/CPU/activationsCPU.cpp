#include "activationsCPU.h"


void Forge::ReluCPU::forward(const Tensor& input, Tensor& output) const{
    DISPATCH_ALL_TYPES(input.dtype(), Device::CPU, [&] {
       auto input_map {input.as_eigen<scalar_t>()};
        auto output_map {output.as_eigen<scalar_t>()};
        forge_eval(output_map, input_map.cwiseMax(static_cast<scalar_t>(0)));
    });
}

void Forge::SigmoidCPU::forward(const Tensor& input, Tensor& output) const{
    DISPATCH_ALL_TYPES(input.dtype(), Device::CPU, [&] {
        auto input_map {input.as_eigen<scalar_t>()};
        auto output_map {output.as_eigen<scalar_t>()};
        auto one {static_cast<scalar_t>(1)};
        forge_eval(output_map, one/(one+((-input_map).exp())));
    });
}

void Forge::TanhCPU::forward(const Tensor& input, Tensor& output) const{
    DISPATCH_ALL_TYPES(input.dtype(), Device::CPU, [&] {
        auto input_map {input.as_eigen<scalar_t>()};
        auto output_map {output.as_eigen<scalar_t>()};
        forge_eval(output_map, input_map.tanh());
    });
}

void  Forge::LeakyReluCPU::forward(const Tensor& input, Tensor& output) const{
    DISPATCH_ALL_TYPES(input.dtype(), Device::CPU, [&] {
        auto input_map {input.as_eigen<scalar_t>()};
        auto output_map {output.as_eigen<scalar_t>()};
        auto alpha {static_cast<scalar_t>(0.01)};
        forge_eval(output_map,(input_map > scalar_t(0)).select(input_map, alpha * input_map));
    });
}

void Forge::GeluCPU::forward(const Tensor& input, Tensor& output) const{
    DISPATCH_ALL_TYPES(input.dtype(), Device::CPU, [&] {
        auto input_map {input.as_eigen<scalar_t>()};
        auto output_map {output.as_eigen<scalar_t>()};
        const auto half = static_cast<scalar_t>(0.5);
        const auto sqrt_2_over_pi = static_cast<scalar_t>(0.7978845608028654);
        const auto coeff = static_cast<scalar_t>(0.044715);

        forge_eval(output_map, half*input_map*(
            static_cast<scalar_t>(1)+(sqrt_2_over_pi*(input_map+coeff*input_map.cube())).tanh()));
    });
}

void Forge::SoftmaxCPU::forward(const Tensor& input, Tensor& output) const{
    Tensor max_vals_eval {input.shape(), input.dtype(), false, input.device()};
    Tensor sum_exp_eval {input.shape(), input.dtype(), false, input.device()};

    DISPATCH_ALL_TYPES(input.dtype(), Device::CPU, [&] {
        auto input_map {input.as_eigen<scalar_t>()};
        auto output_map {output.as_eigen<scalar_t>()};
        auto max_vals_eval_map {max_vals_eval.as_eigen<scalar_t>()};
        auto sum_exp_eval_map {sum_exp_eval.as_eigen<scalar_t>()};
        Eigen::array<Eigen::Index, 1> reduce_axis {3};

        auto dims {input_map.dimensions()};
        Eigen::array<Eigen::Index, 4> kept_dims {dims[0], dims[1], dims[2], 1};
        Eigen::array<Eigen::Index, 4> bcast_dims {1, 1, 1, dims[3]};

        auto max_vals {input_map.maximum(reduce_axis).reshape(kept_dims).broadcast(bcast_dims)};
        forge_eval(max_vals_eval_map, max_vals);

        auto exp_vals {(input_map - max_vals_eval_map).exp()};
        auto sum_exp {exp_vals.sum(reduce_axis).reshape(kept_dims).broadcast(bcast_dims)};
        forge_eval(sum_exp_eval_map, sum_exp);

        auto opt {(exp_vals / sum_exp_eval_map)};
        forge_eval(output_map, opt);
    });
}
