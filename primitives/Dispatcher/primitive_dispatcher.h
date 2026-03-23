#pragma once
#include "ops/Dispatcher/dispatcher.h"

namespace Forge{
    enum class primitive_ops{linear, weightsInit, Count};

    inline Dispatcher<primitive_ops> & primitive_dispatcher() {
        static Dispatcher<primitive_ops> dispatcher;
        return dispatcher;
    }
}