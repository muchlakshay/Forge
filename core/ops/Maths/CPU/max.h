#pragma once
#include <immintrin.h>
#include "tensor.h"
#include "transpose.h"
#include <numeric>

namespace Forge {
    template <typename DTYPE>
    Tensor forge_max_AVX2(const Tensor& A, int axis);
}

template<typename DTYPE>
Forge::Tensor Forge::forge_max_AVX2(const Tensor& A, int axis) {
    if constexpr (std::is_same_v<DTYPE, float>) {
        std::size_t step {8};
        Tensor temp {A};
        if (axis!=A.shape().size()-1) {
            auto new_shape {A.shape()};
            std::swap(new_shape[axis], new_shape.back());

            std::vector<int> permutation (A.shape().size());
            std::iota(permutation.begin(), permutation.end(), 0);
            std::swap(permutation[axis], permutation.back());

            temp = Tensor {new_shape, Dtype::float32, false};
            forge_transpose_AVX2(static_cast<DTYPE*>(A.data()), static_cast<DTYPE*>(temp.data()), A.strides(),
                A.shape(), permutation, A.size());
        }
        Tensor opt {{*(&temp.shape().back()-1)}, Dtype::float32, false};
        auto* data {static_cast<DTYPE*>(temp.data())};
        auto opt_data {static_cast<DTYPE*>(opt.data())};

    }
    return {};
}

