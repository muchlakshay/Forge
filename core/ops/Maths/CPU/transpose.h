#include <immintrin.h>
#include <tensor/tensor.h>

namespace Forge {
    std::vector<std::size_t> compute_strides(const std::vector<std::size_t>& shape);
    void transpose_AVX2(const Tensor& A, const Tensor& B, const std::vector<std::size_t>& transpose_axis);
    std::vector<int> precalculate_offsets(
        const std::vector<std::size_t>& strides_A, const std::vector<std::size_t>& strides_B,
        const std::vector<int>& permutation, std::size_t size);
}

inline std::vector<std::size_t> Forge::compute_strides(const std::vector<std::size_t> &shape) {
    std::size_t stride {1};
    std::vector<std::size_t> strides;
    for (const auto dim : shape | std::views::reverse) {
        strides.push_back(stride);
        stride *= dim;
    }
    return strides;
}

inline std::vector<int> Forge::precalculate_offsets(const std::vector<std::size_t> &strides_A,
    const std::vector<std::size_t>& strides_B, const std::vector<int>& permutation, std::size_t size) {
    std::vector<int> offsets (size);
    auto rank {strides.size()};
    std::vector<int> coords_B(rank);
    std::vector<int> coords_A(rank);

    for (int i {}; i<size; i++) {
        int flat_idx {i}, src_flat_idx {};
        for (int d{}; d<rank; d++) {
            coords_B[d] = flat_idx/strides_B[d];
            flat_idx %= strides_B[d];
            coords_A[permutation[d]] = coords_B[d];
            src_flat_idx += coords_A[d] * strides_A[d];
        }
        offsets[i] = src_flat_idx;
    }
}



