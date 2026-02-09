#pragma once
#include "../Dispatcher/dispatcher.h"
#include "utility_ops.h"

namespace Forge {
    template <ValidEnum T>
    void register_all_kernels(Dispatcher<T>& dispatcher);
}

template <ValidEnum T>
inline void Forge::register_all_kernels(Dispatcher<T> &dispatcher) {
    register_utility_kernels(dispatcher);
}
