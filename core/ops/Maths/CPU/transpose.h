#include <immintrin.h>
#include <tensor/tensor.h>

namespace Forge {
    std::vector<std::size_t> compute_strides(const std::vector<std::size_t>& shape);
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
