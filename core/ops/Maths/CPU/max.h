#pragma once
#include <immintrin.h>
#include "tensor.h"
#include "transpose.h"
#include <numeric>

namespace Forge {
    template <typename DTYPE>
    Tensor forge_max_AVX2(const Tensor& A, int axis);
    float _mm256_hmax_ps(__m256 x);
}

inline float Forge::_mm256_hmax_ps(__m256 x) {
    __m128 lower {_mm256_castps256_ps128(x)};
    __m128 upper {_mm256_extractf128_ps(x, 1)};
    __m128 max4 {_mm_max_ps(lower, upper)};
    __m128 max2 {_mm_max_ps(max4, _mm_shuffle_ps(max4, max4, _MM_SHUFFLE(1, 0, 3, 2)))};
    __m128 max  {_mm_max_ps(max2, _mm_shuffle_ps(max2, max2, _MM_SHUFFLE(0, 1, 0, 1)))};

    return _mm_cvtss_f32(max);
}

template<typename DTYPE>
Forge::Tensor Forge::forge_max_AVX2(const Tensor& A, int axis) {
    if constexpr (std::is_same_v<DTYPE, float>) {
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
        auto opt_shape {temp.shape()};
        opt_shape.back()=1;

        auto opt {Tensor::Zeros(opt_shape, false, Dtype::float32)};
        std::size_t step_size {8}, unroll_steps {4};
        std::size_t i {};

        auto* data {static_cast<DTYPE*>(temp.data())};
        auto opt_data {static_cast<DTYPE*>(opt.data())};

        auto last_axis_size {temp.shape().back()};

    }
    return {};
}

