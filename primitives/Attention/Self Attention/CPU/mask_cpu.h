#pragma once

#include "../mask_abstract.h"
#include "tensor.h"
#include <limits>

namespace Forge {
    struct MaskCPU;
}

struct Forge::MaskCPU : MaskAbstract {
    void generate(Tensor tensor, std::size_t seq_len) const override {
        DISPATCH_ALL_TYPES(tensor.dtype(), tensor.device(), [&] {
            auto* data_ptr {static_cast<scalar_t*>(tensor.data())};
            for (std::size_t i {}; i<seq_len; ++i) {
                for (std::size_t j {}; j<seq_len; ++j) {if (j>i) data_ptr[i * seq_len + j] = -static_cast<scalar_t>(10000);}
            }
        });
    }
};