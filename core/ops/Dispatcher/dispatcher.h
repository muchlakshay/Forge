#pragma once
#include "kernel_base.h"
#include "tensor.h"

namespace Forge {
    class Dispatcher;
}

template <typename T>
concept KernelDerived = std::is_base_of_v<Forge::Kernel, T>;

class Forge::Dispatcher {
public:
    enum class Ops{add, sub, matmul, OpsCount};
    enum class DispatchKey {CPU, CUDA, CPU_Autodiff, CUDA_Autodiff, DispatchKeysCount};
    Forge::Kernel* registry [Ops::OpsCount][DispatchKey::DispatchKeysCount] {nullptr};
public:
    template <KernelDerived T>
    void register_kernel(const T* func_class, Ops op, DispatchKey dispatch_key);
    template <KernelDerived T>
     T* lookup(Ops op, DispatchKey dispatch_key);
};

