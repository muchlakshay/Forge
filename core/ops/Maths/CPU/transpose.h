#include <immintrin.h>
#include <tensor/tensor.h>

namespace Forge {
    std::vector<std::size_t> compute_strides(const std::vector<std::size_t>& shape);
    template <typename DTYPE>
    void transpose_AVX2(const DTYPE* A, DTYPE* B, const std::vector<std::size_t>& strides_A,
        const std::vector<std::size_t>& shape_A, const std::vector<int>& permutation, std::size_t size);
    std::vector<int> precalculate_offsets(
        const std::vector<std::size_t>& strides_A, const std::vector<std::size_t>& strides_B,
        const std::vector<int>& permutation, std::size_t size);
}

inline std::vector<std::size_t> Forge::compute_strides(const std::vector<std::size_t> &shape) {
    std::size_t stride {1}, idx {shape.size()-1};
    std::vector<std::size_t> strides (shape.size());
    for (const auto dim : shape | std::views::reverse) {
        strides[idx--] = stride;
        stride *= dim;
    }
    return strides;
}

inline std::vector<int> Forge::precalculate_offsets(const std::vector<std::size_t> &strides_A,
    const std::vector<std::size_t>& strides_B, const std::vector<int>& permutation, std::size_t size) {
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
void Forge::transpose_AVX2(const DTYPE *A, DTYPE *B, const std::vector<std::size_t> &strides_A,
    const std::vector<std::size_t> &shape_A, const std::vector<int> &permutation, std::size_t size) {
    int step {8};
    std::size_t end {size-step};
    std::vector<std::size_t> shape_B;
    for (int i{}; i<shape_A.size(); i++) shape_B.push_back(shape_A[permutation[i]]);
    auto strides_B {compute_strides(shape_B)};
    const auto offsets {precalculate_offsets(strides_A, strides_B, permutation, size)};

#pragma omp parallel for schedule(static)
    for (int i {}; i<=end; i+=step) {
        __m256i v_offsets {_mm256_loadu_si256(reinterpret_cast<const __m256i *>(&(offsets[i])))};
        __m256 v_data {_mm256_i32gather_ps(A, v_offsets, 4)};
        _mm256_storeu_ps(B+i, v_data);
    }

    const std::size_t remainder_start = (size / step) * step;
    for (auto i {remainder_start}; i < size; ++i) B[i] = A[offsets[i]];
}