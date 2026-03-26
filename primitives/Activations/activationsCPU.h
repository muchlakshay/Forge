#pragma once

#include "activationsAbstarct.h"
#include "activationsCPU.h"

namespace Forge {
    struct ReluCPU;
    struct SigmoidCPU;
    struct TanhCPU;
    struct SoftmaxCPU;
    struct LeakyReluCPU;
    struct GeluCPU;
}

struct Forge::ReluCPU : ReluAbstract{
    void forward(const Tensor& input, Tensor& output) override {
        DISPATCH_ALL_TYPES(input.dtype(), Device::CPU, [&] {
           auto input_map {input.as_eigen<scalar_t>()};
            auto output_map {output.as_eigen<scalar_t>()};
            output_map = input_map.cwiseMax(static_cast<scalar_t>(0));
        });
    }
};

struct Forge::SigmoidCPU : SigmoidAbstract{
    void forward(const Tensor& input, Tensor& output) override {
        DISPATCH_ALL_TYPES(input.dtype(), Device::CPU, [&] {
            auto input_map {input.as_eigen<scalar_t>()};
            auto output_map {output.as_eigen<scalar_t>()};
            auto one {static_cast<scalar_t>(1)};
            output_map = one/(one+((-input_map).exp()));
        });
    }
};

struct Forge::TanhCPU : TanhAbstract {
    void forward(const Tensor& input, Tensor& output) override {
        DISPATCH_ALL_TYPES(input.dtype(), Device::CPU, [&] {
            auto input_map {input.as_eigen<scalar_t>()};
            auto output_map {output.as_eigen<scalar_t>()};
            output_map = input_map.tanh();
        });
    }
};

struct Forge::LeakyReluCPU : LeakyReluAbstract {
    void forward(const Tensor& input, Tensor& output) override {
        DISPATCH_ALL_TYPES(input.dtype(), Device::CPU, [&] {
            auto input_map {input.as_eigen<scalar_t>()};
            auto output_map {output.as_eigen<scalar_t>()};
            auto alpha {static_cast<scalar_t>(0.01)};
            output_map = (input_map > scalar_t(0)).select(input_map, alpha * input_map);
        });
    }
};

struct Forge::GeluCPU : GeluAbstract {
    void forward(const Tensor& input, Tensor& output) override {
        DISPATCH_ALL_TYPES(input.dtype(), Device::CPU, [&] {
            auto input_map {input.as_eigen<scalar_t>()};
            auto output_map {output.as_eigen<scalar_t>()};
            const auto half = static_cast<scalar_t>(0.5);
            const auto sqrt_2_over_pi = static_cast<scalar_t>(0.7978845608028654);
            const auto coeff = static_cast<scalar_t>(0.044715);

            output_map = half*input_map*(
                static_cast<scalar_t>(1)+(
                        sqrt_2_over_pi*(input_map+coeff*input_map.cube())
                        ).tanh());
        });
    }
};

struct Forge::SoftmaxCPU : SoftmaxAbstract {
    void forward(const Tensor& input, Tensor& output) override {
        DISPATCH_ALL_TYPES(input.dtype(), Device::CPU, [&] {
            auto input_map {input.as_eigen<scalar_t>()};
            auto output_map {output.as_eigen<scalar_t>()};

            Eigen::array<Eigen::Index, 1> reduce_axis {3};

            auto dims {input_map.dimensions()};
            Eigen::array<Eigen::Index, 4> kept_dims {dims[0], dims[1], dims[2], 1};
            Eigen::array<Eigen::Index, 4> bcast_dims {1, 1, 1, dims[3]};

            auto max_vals {input_map.maximum(reduce_axis).reshape(kept_dims).broadcast(bcast_dims)};
            auto exp_vals {(input_map - max_vals).exp()};

            auto sum_exp {exp_vals.sum(reduce_axis).reshape(kept_dims).broadcast(bcast_dims)};
            output_map = exp_vals / sum_exp;
        });
    }
};