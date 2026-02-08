#pragma once
#include "kernel_base.h"
#include "enums.h"
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

template <typename T>
concept KernelDerived = std::is_base_of_v<Forge::Kernel, T>;

template <typename T>
concept ValidEnum = std::is_enum_v<T> && requires {T::Count;};

namespace Forge {
    template<ValidEnum T>
    class Dispatcher;
}

template <ValidEnum T>
class Forge::Dispatcher {
private:
    Forge::Kernel* m_registry [static_cast<int>(T::Count)][static_cast<int>(DispatchKey::DispatchKeysCount)] {nullptr};
    template<typename E>
    static constexpr int to_idx(E x) {return static_cast<int>(x);}
public:
    template <KernelDerived U>
    void register_kernel(U* func_class, T op, DispatchKey dispatch_key);
    template <KernelDerived U>
     U* lookup(T op, DispatchKey dispatch_key);
};

template <ValidEnum T>
template <KernelDerived U>
inline void Forge::Dispatcher<T>::register_kernel(U* func_class, T op, DispatchKey dispatch_key) {
    if (Kernel*& kernel {m_registry[to_idx(op)][to_idx(dispatch_key)]}; kernel == nullptr)
       kernel = func_class;
    else throw std::invalid_argument("Kernel already registered");
}

template <ValidEnum T>
template<KernelDerived U>
inline U* Forge::Dispatcher<T>::lookup(T op, DispatchKey dispatch_key) {
    Kernel* kernel {m_registry[to_idx(op)][to_idx(dispatch_key)]};
    if (kernel==nullptr)
        throw std::invalid_argument("Kernel not registered");

    if (kernel->derived_id() == ctti::type_id<U>())
        return static_cast<U*>(kernel);
    else throw std::invalid_argument("Invalid Op class passed to downcast to");
}

