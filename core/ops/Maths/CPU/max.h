#pragma once
#include <immintrin.h>
#include "tensor.h"
#include "transpose.h"
#include <numeric>
#include <omp.h>

namespace Forge {
    template <typename DTYPE>
    Tensor forge_max_AVX2(const Tensor& A, int axis);
    float _mm256_hmax_ps(__m256 x);
    double _mm256_hmax_pd(__m256d x);
}

inline float Forge::_mm256_hmax_ps(__m256 x) {
    __m128 lower {_mm256_castps256_ps128(x)};
    __m128 upper {_mm256_extractf128_ps(x, 1)};
    __m128 max4 {_mm_max_ps(lower, upper)};
    __m128 max2 {_mm_max_ps(max4, _mm_shuffle_ps(max4, max4, _MM_SHUFFLE(1, 0, 3, 2)))};
    __m128 max  {_mm_max_ps(max2, _mm_shuffle_ps(max2, max2, _MM_SHUFFLE(0, 1, 0, 1)))};

    return _mm_cvtss_f32(max);
}

inline double Forge::_mm256_hmax_pd(__m256d x) {
    __m128d lower {_mm256_castpd256_pd128(x)};
    __m128d upper {_mm256_extractf128_pd(x, 1)};
    __m128d max2 {_mm_max_pd(lower, upper)};
    __m128d max  {_mm_max_pd(max2, _mm_shuffle_pd(max2, max2, _MM_SHUFFLE2(1, 0)))};
    return _mm_cvtsd_f64(max);
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

        auto opt {Tensor::Constant({opt_shape}, -std::numeric_limits<float>::infinity(), false, Dtype::float32)};
        std::size_t step_size {8}, unroll_steps {4};
        std::size_t i {};
        std::size_t row{};

        auto* data {static_cast<DTYPE*>(temp.data())};
        auto opt_data {static_cast<DTYPE*>(opt.data())};

        auto last_axis_size {temp.shape().back()};
        auto total_rows {temp.size()/last_axis_size};

        // #pragma omp parallel for schedule(static) private(i)
        for (int _ = 0;row+unroll_steps<total_rows; row+=unroll_steps) {
            for (; i+step_size<last_axis_size; i+=step_size) {
                // std::cout<<"here\n";
                __m256 x1 {_mm256_loadu_ps(row*last_axis_size+data+i)};
                __m256 x2 {_mm256_loadu_ps((row+1)*last_axis_size+data+i)};
                __m256 x3 {_mm256_loadu_ps((row+2)*last_axis_size+data+i)};
                __m256 x4 {_mm256_loadu_ps((row+3)*last_axis_size+data+i)};

                float max1 {_mm256_hmax_ps(x1)};
                float max2 {_mm256_hmax_ps(x2)};
                float max3 {_mm256_hmax_ps(x3)};
                float max4 {_mm256_hmax_ps(x4)};

                if (opt_data[row]<max1) opt_data[row]=max1;
                if (opt_data[row+1]<max2) opt_data[row+1]=max2;
                if (opt_data[row+2]<max3) opt_data[row+2]=max3;
                if (opt_data[row+3]<max4) opt_data[row+3]=max4;
            }
            i=0;
        }
        if (row<total_rows) {
            for (int _=0;row<total_rows; row++) {
                for (std::size_t i {}; i+step_size<last_axis_size; i+=step_size) {
                    // std::cout<<"here\n";
                    __m256 x {_mm256_loadu_ps(row*last_axis_size+data+i)};
                    float max {_mm256_hmax_ps(x)};
                    if (opt_data[row]<max) opt_data[row]=max;
                }
            }
        }

        if (std::size_t j {last_axis_size-(last_axis_size%step_size)};j<last_axis_size) {
             // #pragma omp parallel for schedule(static)
            for (std::size_t row = 0; row<total_rows; ++row) {
                for (std::size_t i{j}; i<last_axis_size; i++) if (opt_data[row]<data[row*last_axis_size+i])
                    opt_data[row]=data[row*last_axis_size+i];
            }
        }

        if (axis!=A.shape().size()-1) {
            std::vector<int> permutation (A.shape().size());
            std::iota(permutation.begin(), permutation.end(), 0);
            std::swap(permutation[axis], permutation.back());
            auto new_opt_shape {opt.shape()};
            std::swap(new_opt_shape[axis], new_opt_shape.back());

            Tensor opt_trans{new_opt_shape, Dtype::float32, A.need_grads()};
            forge_transpose_AVX2(opt_data, static_cast<float*>(opt_trans.data()), opt.strides(), opt.shape(),
                permutation, opt.size());
            return opt_trans;
        }
        return opt;
    }
    // else if constexpr (std::is_same_v<DTYPE, double>) {
    //     Tensor temp {A};
    //     if (axis!=A.shape().size()-1) {
    //         auto new_shape {A.shape()};
    //         std::swap(new_shape[axis], new_shape.back());
    //
    //         std::vector<int> permutation (A.shape().size());
    //         std::iota(permutation.begin(), permutation.end(), 0);
    //         std::swap(permutation[axis], permutation.back());
    //
    //         temp = Tensor {new_shape, Dtype::float64, false};
    //         forge_transpose_AVX2(static_cast<DTYPE*>(A.data()), static_cast<DTYPE*>(temp.data()), A.strides(),
    //             A.shape(), permutation, A.size());
    //     }
    //     auto opt_shape {temp.shape()};
    //     opt_shape.back()=1;
    //
    //     auto opt {Tensor::Constant({opt_shape}, -std::numeric_limits<double>::infinity(), false, Dtype::float64)};
    //     std::size_t step_size {4}, unroll_steps {4};
    //     std::size_t i {};
    //     std::size_t row{};
    //
    //     auto* data {static_cast<DTYPE*>(temp.data())};
    //     auto opt_data {static_cast<DTYPE*>(opt.data())};
    //
    //     auto last_axis_size {temp.shape().back()};
    //     auto total_rows {temp.size()/last_axis_size};
    //
    //     // #pragma omp parallel for schedule(static) private(i)
    //     for (int _ = 0;row+unroll_steps<total_rows; row+=unroll_steps) {
    //         for (; i+step_size<last_axis_size; i+=step_size) {
    //             // std::cout<<"here\n";
    //             __m256d x1 {_mm256_loadu_pd(row*last_axis_size+data+i)};
    //             __m256d x2 {_mm256_loadu_pd((row+1)*last_axis_size+data+i)};
    //             __m256d x3 {_mm256_loadu_pd((row+2)*last_axis_size+data+i)};
    //             __m256d x4 {_mm256_loadu_pd((row+3)*last_axis_size+data+i)};
    //
    //             double max1 {_mm256_hmax_pd(x1)};
    //             double max2 {_mm256_hmax_pd(x2)};
    //             double max3 {_mm256_hmax_pd(x3)};
    //             double max4 {_mm256_hmax_pd(x4)};
    //
    //             if (opt_data[row]<max1) opt_data[row]=max1;
    //             if (opt_data[row+1]<max2) opt_data[row+1]=max2;
    //             if (opt_data[row+2]<max3) opt_data[row+2]=max3;
    //             if (opt_data[row+3]<max4) opt_data[row+3]=max4;
    //         }
    //         i=0;
    //     }
    //     if (row<total_rows) {
    //         for (int _=0;row<total_rows; row++) {
    //             for (std::size_t i {}; i+step_size<last_axis_size; i+=step_size) {
    //                 // std::cout<<"here\n";
    //                 __m256d x {_mm256_loadu_pd(row*last_axis_size+data+i)};
    //                 double max {_mm256_hmax_pd(x)};
    //                 if (opt_data[row]<max) opt_data[row]=max;
    //             }
    //         }
    //     }
    //
    //     if (std::size_t j {last_axis_size-(last_axis_size%step_size)};j<last_axis_size) {
    //          // #pragma omp parallel for schedule(static)
    //         for (std::size_t row = 0; row<total_rows; ++row) {
    //             for (std::size_t i{j}; i<last_axis_size; i++) if (opt_data[row]<data[row*last_axis_size+i])
    //                 opt_data[row]=data[row*last_axis_size+i];
    //         }
    //     }
    //
    //     if (axis!=A.shape().size()-1) {
    //         std::vector<int> permutation (A.shape().size());
    //         std::iota(permutation.begin(), permutation.end(), 0);
    //         std::swap(permutation[axis], permutation.back());
    //         auto new_opt_shape {opt.shape()};
    //         std::swap(new_opt_shape[axis], new_opt_shape.back());
    //
    //         Tensor opt_trans{new_opt_shape, Dtype::float64, A.need_grads()};
    //         forge_transpose_AVX2(opt_data, static_cast<double*>(opt_trans.data()), opt.strides(), opt.shape(),
    //             permutation, opt.size());
    //         return opt_trans;
    //     }
    //     return opt;
    // }
    return {};
}

