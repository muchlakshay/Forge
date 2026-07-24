#pragma once
#include <immintrin.h>
#include "tensor.h"
#include "transpose.h"

namespace Forge {
    template <typename DTYPE>
    Tensor forge_max_AVX2(const Tensor& A, int axis);
}


