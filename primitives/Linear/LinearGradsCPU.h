#pragma once
#include "LinearGradsAbstract.h"

namespace Forge {
    struct LinearGradsCPU;
}

struct Forge::LinearGradsCPU final :  LinearGradsAbstract {
    void compute_grads(Tensor& input, Tensor& weights, Tensor& bias, const Tensor& output) const override {
        using grads_t = float;
        const auto tensor_dtype {input.dtype()};
        Eigen::array<Eigen::IndexPair<std::size_t>, 1> contraction_dims{};

        auto output_grads_map {output.gradients().as_eigen<grads_t>()};
        DISPATCH_ALL_TYPES(tensor_dtype, Device::CPU, [&] {
            if (input.need_grads()) {
                auto input_grads_map {input.gradients().as_eigen<grads_t>()};
                contraction_dims[0] = Eigen::IndexPair<std::size_t>(3, 2);
                input_grads_map += output_grads_map.contract(weights.as_eigen<scalar_t>(),
                    contraction_dims).reshape(input_grads_map.dimensions());
            }
        });
    }
};