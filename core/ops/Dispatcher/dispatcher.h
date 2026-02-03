#pragma once
#include "kernel_base.h"
#include "tensor.h"

namespace Forge {
    class Dispatcher;
}

class Forge::Dispatcher {
public:
    enum class Ops{add, sub, matmul, OpsCount};
    enum class DispatchKey {CPU, CUDA, CPU_Autodiff, CUDA_Autodiff, DispatchKeysCount};
    Forge::Kernel* registry [Ops::OpsCount][DispatchKey::DispatchKeysCount] {nullptr};
};

