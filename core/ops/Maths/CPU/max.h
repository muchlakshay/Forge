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

        auto opt {Tensor::Constant({opt_shape}, -10e5, false, Dtype::float32)};
        std::size_t step_size {8}, unroll_steps {4};
        std::size_t i {};

        auto* data {static_cast<DTYPE*>(temp.data())};
        auto opt_data {static_cast<DTYPE*>(opt.data())};

        auto last_axis_size {temp.shape().back()};

        for (std::size_t row {}; row<(temp.size()/last_axis_size); ++row) {
            for (; i+step_size*unroll_steps<A.size(); i+=step_size*unroll_steps) {
                auto row_offset {row*last_axis_size};
                __m256 x1 {_mm256_loadu_ps(row_offset+data+i)};
                __m256 x2 {_mm256_loadu_ps(row_offset+data+i+step_size)};
                __m256 x3 {row_offset+data+i+step_size*2};
                __m256 x4 {row_offset+data+i+step_size*3};

                float max1 {_mm256_hmax_ps(x1)};
                float max2 {_mm256_hmax_ps(x2)};
                float max3 {_mm256_hmax_ps(x3)};
                float max4 {_mm256_hmax_ps(x4)};

                if (opt_data[row]<max1) opt_data[row]=max1;
                if (opt_data[row+1]<max2) opt_data[row+1]=max2;
                if (opt_data[row+2]<max3) opt_data[row+2]=max3;
                if (opt_data[row+3]<max4) opt_data[row+3]=max4;
            }
        }
        if (i<temp.shape().back()) {
            for (std::size_t row {}; row<(temp.size()/last_axis_size); ++row) {
                for (; i<temp.shape().back(); i++) if (opt_data[row]<data[i]) opt_data[row]=data[i];
            }
        }
        if (axis!=A.shape().size()-1) {
            std::vector<int> permutation (A.shape().size());
            std::iota(permutation.begin(), permutation.end(), 0);
            std::swap(permutation[axis], permutation.back());
            auto new_opt_shape {opt.shape()};
            std::swap(new_opt_shape[axis], new_opt_shape.back());

            auto opt_trans {Tensor::FromHostPtr(static_cast<float*>(temp.data()), new_opt_shape, A.need_grads())};
            forge_transpose_AVX2(opt_data, static_cast<float*>(opt_trans.data()), opt.strides(), opt.shape(),
                permutation);
            return opt_trans;
        }
        return opt;
    }
    return {};
}

