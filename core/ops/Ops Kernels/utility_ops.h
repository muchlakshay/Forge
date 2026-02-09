#pragma once
#include "enums.h"
#include "ops/Dispatcher/dispatcher.h"

namespace Forge {
    void register_utility_kernels(Dispatcher<UtilityOps>& dispatcher);
}



