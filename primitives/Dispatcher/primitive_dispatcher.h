#pragma once
#include "ops/Dispatcher/dispatcher.h"

namespace Forge{
    enum class primitive_ops{linear, weightsInit, relu, leaky_relu, sigmoid, tanh, softmax, gelu, Count};

    inline Dispatcher<primitive_ops> & primitive_dispatcher() {
        static Dispatcher<primitive_ops> dispatcher;
        return dispatcher;
    }
}