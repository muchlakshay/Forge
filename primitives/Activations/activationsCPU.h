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
