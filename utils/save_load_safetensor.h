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


