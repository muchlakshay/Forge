#pragma once
#include <immintrin.h>
#include <tensor/tensor.h>

namespace Forge {
    using size_vec = std::vector<size_t>;
    size_vec compute_strides(const size_vec& shape);
    template <typename DTYPE>
    void forge_transpose_AVX2(const DTYPE* A, DTYPE* B, const size_vec& strides_A,
        const size_vec& shape_A, const std::vector<int>& permutation, std::size_t size);
    std::vector<int> precalculate_offsets(
        const size_vec& strides_A, const size_vec& strides_B,
        const std::vector<int>& permutation, std::size_t size);
}

inline Forge::size_vec Forge::compute_strides(const size_vec &shape) {
    std::size_t stride {1}, idx {shape.size()-1};
    size_vec strides (shape.size());
    for (const auto dim : shape | std::views::reverse) {
        strides[idx--] = stride;
        stride *= dim;
    }
    return strides;
}

inline std::vector<int> Forge::precalculate_offsets(const size_vec &strides_A,
    const size_vec& strides_B, const std::vector<int>& permutation, std::size_t size) {
    std::vector<int> offsets (size);
    auto rank {strides_A.size()};
    std::vector<int> coords_B(rank);
    std::vector<int> coords_A(rank);

    for (std::size_t i {}; i<size; i++) {
        std::size_t flat_idx {i}, src_flat_idx {};
        for (int d{}; d<rank; d++) {
            coords_B[d] = flat_idx/strides_B[d];
            flat_idx %= strides_B[d];
            coords_A[permutation[d]] = coords_B[d];
        }

        for (std::size_t d{}; d < rank; d++)src_flat_idx += coords_A[d] * strides_A[d];
        offsets[i] = src_flat_idx;
    }
    return offsets;
}


template<typename DTYPE>
void Forge::forge_transpose_AVX2(const DTYPE *A, DTYPE *B, const size_vec &strides_A,
    const size_vec &shape_A, const std::vector<int> &permutation, std::size_t size) {
    if constexpr (std::is_same_v<DTYPE, float>) {
        std::size_t step {8};
        std::size_t end {size-step};
        size_vec shape_B;
        for (int i{}; i<shape_A.size(); i++) shape_B.push_back(shape_A[permutation[i]]);
        auto strides_B {compute_strides(shape_B)};
        const auto offsets {precalculate_offsets(strides_A, strides_B, permutation, size)};

        #pragma omp parallel for schedule(static)
        for (std::size_t i =0; i<=end; i+=step) {
            __m256i v_offsets {_mm256_loadu_si256(reinterpret_cast<const __m256i *>(&(offsets[i])))};
            __m256 v_data {_mm256_i32gather_ps(A, v_offsets, 4)};
            _mm256_storeu_ps(B+i, v_data);
        }

        const std::size_t remainder_start = (size / step) * step;
        for (auto i {remainder_start}; i < size; ++i) B[i] = A[offsets[i]];
    }
    if constexpr (std::is_same_v<double, DTYPE>) {
        // std::size_t step {4};
        // std::size_t end {size-step};
        // size_vec shape_B;
        // for (int i{}; i<shape_A.size(); i++) shape_B.push_back(shape_A[permutation[i]]);
        // auto strides_B {compute_strides(shape_B)};
        // const auto offsets {precalculate_offsets(strides_A, strides_B, permutation, size)};
        //
        // #pragma omp parallel for schedule(static)
        // for (std::size_t i =0; i<=end; i+=step) {
        //     __m256i v_offsets {_mm256_loadu_si256(reinterpret_cast<const __m256i *>(&(offsets[i])))};
        //     __m256d v_data {_mm256_i32gather_pd(A, v_offsets, 8)};
        //     _mm256_storeu_pd(B+i, v_data);
        // }
        //
        // const std::size_t remainder_start = (size / step) * step;
        // for (auto i {remainder_start}; i < size; ++i) B[i] = A[offsets[i]];
    }
}