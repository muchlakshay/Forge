#pragma once
#include "kernel_base.h"
#include <Eigen/Dense>

#define DISPATCH_ALL_TYPES(DTYPE, DEVICE, ...)\
[&]() {                                  \
switch (DTYPE) {                       \
case Forge::Dtype::float16: {          \
if(DEVICE==Forge::Device::CUDA){       \
throw std::runtime_error("float16 CUDA not implemented"); \
}                                      \
else {                                 \
using scalar_t = Eigen::half;          \
 __VA_ARGS__();                        \
}                                      \
break ;                                \
}                                      \
case Forge::Dtype::float32: {          \
using scalar_t = float;                \
__VA_ARGS__();                         \
break;                                 \
}                                      \
case Forge::Dtype::float64: {          \
using scalar_t = double;               \
__VA_ARGS__();                         \
break;                                 \
}                                      \
case Forge::Dtype::int16: {            \
using scalar_t = std::int16_t;         \
__VA_ARGS__();                         \
break;                                 \
}                                      \
case Forge::Dtype::int32: {            \
using scalar_t = std::int32_t;         \
__VA_ARGS__();                         \
break;                                 \
}                                      \
case Forge::Dtype::int64: {            \
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

template <typename T>
concept Enum = std::is_enum_v<T>;

class Forge::Dispatcher {
public:
    enum class Ops{add, sub, matmul, storage_backend, OpsCount};
    enum class DispatchKey {CPU, CUDA, CPU_Autodiff, CUDA_Autodiff, DispatchKeysCount};
private:
    Forge::Kernel* m_registry [static_cast<int>(Ops::OpsCount)][static_cast<int>(DispatchKey::DispatchKeysCount)] {nullptr};
    template <Enum T>
    static constexpr int to_idx(T x) {return static_cast<int>(x);}
public:
    template <KernelDerived T>
    void register_kernel(T* func_class, Ops op, DispatchKey dispatch_key);
    template <KernelDerived T>
     T* lookup(Ops op, DispatchKey dispatch_key);
};

template <KernelDerived T>
inline void Forge::Dispatcher::register_kernel(T* func_class, Ops op, DispatchKey dispatch_key) {
    if (Kernel*& kernel {m_registry[to_idx(op)][to_idx(dispatch_key)]}; kernel == nullptr)
       kernel = func_class;
    else throw std::invalid_argument("Kernel already registered");
}

template<KernelDerived T>
inline T* Forge::Dispatcher::lookup(Ops op, DispatchKey dispatch_key) {
    Kernel* kernel {m_registry[to_idx(op)][to_idx(dispatch_key)]};
    if (kernel==nullptr)
        throw std::invalid_argument("Kernel not registered");

    if (kernel->derived_id() == ctti::type_id<T>())
        return static_cast<T*>(kernel);
    else throw std::invalid_argument("Invalid Op class passed to downcast to");
}
