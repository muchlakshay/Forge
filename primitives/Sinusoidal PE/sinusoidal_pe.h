#pragma once
#include <vector>
#include "tensor.h"

namespace Forge {
    Tensor sinusoidal_pe(std::size_t seq_len, std::size_t d_model, Dtype dtype=Dtype::float32,
        Device device=Device::CPU);
}