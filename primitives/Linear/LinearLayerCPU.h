#pragma once
#include "tensor.h"
#include "LinearAbstract.h"

namespace Forge {
    struct LinearCPU;
}

struct Forge::LinearCPU final : LinearAbstract {
    void forward(const Tensor& input, const Tensor& output, const Tensor& weights,
        const Tensor& bias, bool using_bias) const override {
        DISPATCH_ALL_TYPES(input.dtype(), Device::CPU, [&] {
            auto weights_map { weights.as_eigen<scalar_t>()};
            auto input_map   { input.as_eigen<scalar_t>()  };
            auto output_map  { output.as_eigen<scalar_t>() };

            Eigen::array<std::size_t , 4> shuffling {0, 1, 3, 2};
            auto transposed_weights {weights_map.shuffle(shuffling)};

            Eigen::array<Eigen::IndexPair<std::size_t>, 1> contraction_dims {Eigen::IndexPair<std::size_t>(3, 2)};
            output_map = input_map.contract(transposed_weights,
                contraction_dims).reshape(output_map.dimensions());
        });
    }
};
