#pragma once
#include "kernel_base.h"
#include "tensor.h"

#define DISPATCH_ALL_TYPES(DTYPE, DEVICE, ...)\
[&] {                                  \
switch (DTYPE) {                       \
case Forge::DType::float16: {          \
if(DEVICE==Forge::Device::CUDA){       \
using scalar_t = __half;               \
__VA_ARGS__();                         \
}                                      \
else {                                 \
using scalar_t = Eigen::half;          \
 __VA_ARGS__();                        \
}                                      \
break ;                                \
}                                      \
case Forge::DType::float32: {          \
using scalar_t = float;                \
__VA_ARGS__();                         \
break;                                 \
}                                      \
case Forge::DType::float64: {          \
using scalar_t = double;               \
__VA_ARGS__();                         \
break;                                 \
}                                      \
case Forge::DType::int16: {            \
using scalar_t = std::int16_t;         \
__VA_ARGS__();                         \
break;                                 \
}                                      \
case Forge::DType::int32: {            \
using scalar_t = std::int32_t;         \
__VA_ARGS__();                         \
break;                                 \
}                                      \
case Forge::DType::int64: {            \
using scalar_t = std::int64_t;         \
__VA_ARGS__();                         \
break;                                 \
}                                      \
default: throw std::runtime_error("Unsupported type");\
}                                      \
}()

namespace Forge {
    class Dispatcher;
}

template <typename T>
concept KernelDerived = std::is_base_of_v<Forge::Kernel, T>;

class Forge::Dispatcher {
public:
    enum class Ops{add, sub, matmul, OpsCount};
    enum class DispatchKey {CPU, CUDA, CPU_Autodiff, CUDA_Autodiff, DispatchKeysCount};
    Forge::Kernel* m_registry [Ops::OpsCount][DispatchKey::DispatchKeysCount] {nullptr};
public:
    template <KernelDerived T>
    void register_kernel(T* func_class, Ops op, DispatchKey dispatch_key);
    template <KernelDerived T>
     T* lookup(Ops op, DispatchKey dispatch_key);
};

template <KernelDerived T>
inline void Forge::Dispatcher::register_kernel(T* func_class, Ops op, DispatchKey dispatch_key) {
    if (Kernel*& kernel {m_registry[static_cast<int>(op)][static_cast<int>(dispatch_key)]}; kernel == nullptr)
       kernel = func_class;
    else throw std::invalid_argument("Kernel already registered");
}

template<KernelDerived T>
inline T* Forge::Dispatcher::lookup(Ops op, DispatchKey dispatch_key) {
    Kernel* kernel {m_registry[static_cast<int>(op)][static_cast<int>(dispatch_key)]};
    if (kernel==nullptr)
        throw std::invalid_argument("Kernel not registered");

    if (kernel->derived_id() == ctti::type_id<T>)
        return static_cast<T*>(kernel);
    else throw std::invalid_argument("Invalid Op class passed to downcast to");
}
