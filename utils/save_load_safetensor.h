#pragma once
#include "tensor.h"
#include "parameter.h"
#include <rfl.hpp>
#include <rfl/json.hpp>
#include <fstream>

namespace Forge {
    std::string get_dtype(Dtype dtype);
    std::size_t get_dtype_size(Dtype dtype);

    template <typename T>
    void save(T& data_members, const std::string& filename);

    struct TensorMetadata {
        std::string dtype;
        std::vector<std::size_t> shape;
        std::vector<std::size_t> data_offsets;
    };
}
inline std::string Forge::get_dtype(const Dtype dtype) {
    switch (dtype) {
        case Dtype::float16: return "F16";
        case Dtype::float32: return "F32";
        case Dtype::float64: return "F64";
        default: throw std::invalid_argument("Unsupported dtype");
    }
}

inline std::size_t Forge::get_dtype_size(Dtype dtype) {
    switch (dtype) {
        case Dtype::float16: return 2;
        case Dtype::float32: return 4;
        case Dtype::float64: return 8;
        default: throw std::invalid_argument("Unsupported dtype");
    }
}



