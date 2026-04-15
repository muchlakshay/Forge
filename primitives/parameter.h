#pragma once
#include "tensor.h"

namespace Forge {
    struct Parameter {
        Tensor* m_param_ptr;
        bool need_decay {};
    };
}