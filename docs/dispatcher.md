# Dispatcher & Kernel System

The dispatcher is the routing layer of Forge — it's how operations get connected to the right implementation for a given device and data type. Instead of hardcoding operations inside the tensor class or using long chains of if-else statements, everything goes through a centralized dispatch mechanism.

The code lives in `core/ops/Dispatcher/`:
- `dispatcher.h` — The generic dispatcher template
- `kernel_base.h` — The base class all kernels inherit from

---

## The Problem This Solves

Imagine you're implementing a tensor `add` operation. You need different implementations for:
- CPU float32
- CPU float64
- CPU int32
- CUDA float32
- CUDA float64
- CPU float32 with autograd
- ...and so on

Without a dispatcher, you'd end up with:

```cpp
// DON'T do this
Tensor add(const Tensor& a, const Tensor& b) {
    if (a.device() == Device::CPU) {
        if (a.dtype() == Dtype::float32) {
            if (a.requires_grad()) {
                return add_cpu_float32_autograd(a, b);
            } else {
                return add_cpu_float32(a, b);
            }
        } else if (a.dtype() == Dtype::float64) {
            // ...
        }
    } else if (a.device() == Device::CUDA) {
        // more branching...
    }
}
```

This doesn't scale. Every new operation, device, or type multiplies the branching. And it couples everything together — changing how CUDA add works means touching the same function that handles CPU add.

The dispatcher replaces all of that with a lookup table.

---

## How the Dispatcher Works

### The 2D Registry

The dispatcher is a template class parameterized on an operation enum:

```cpp
template <ValidEnum T>
class Dispatcher {
    Kernel* m_registry[static_cast<int>(T::Count)][static_cast<int>(DispatchKey::DispatchKeysCount)] {nullptr};
};
```

It's a 2D array indexed by:
- **Row**: The operation (from the `T` enum — e.g., `UtilityOps::constant`, `MathOps::matmul`)
- **Column**: The dispatch key (e.g., `CPU`, `CUDA`, `CPU_Autodiff`)

Each cell holds a pointer to a `Kernel` object. The table is initialized to `nullptr` — kernels are registered explicitly.

The sizing comes from the enum conventions. Every operation enum must have a `Count` member as its last value, and `DispatchKey` has `DispatchKeysCount`. These serve as the array dimensions.

### ValidEnum Concept

The dispatcher requires its template parameter to satisfy:

```cpp
template <typename T>
concept ValidEnum = requires { T::Count; };
```

This means you can only instantiate a `Dispatcher` with an enum that has a `Count` member. This is a simple but effective compile-time check — if someone tries to use an enum without `Count`, the concept fails with a clear error.

The `Count` pattern is a standard C++ trick for getting the number of values in an enum without manually maintaining a separate constant. By always keeping `Count` as the last member, `static_cast<int>(T::Count)` gives the number of preceding values.

---

## Kernel Registration

### The Kernel Base Class

Every kernel inherits from `Kernel`:

```cpp
class Kernel {
    ctti::type_id_t m_derived_id{};
public:
    explicit Kernel(const ctti::type_id_t& derived_id) : m_derived_id(derived_id) {}
    ctti::type_id_t derived_id() const { return m_derived_id; }
};
```

The key field is `m_derived_id` — a compile-time type identifier from the CTTI library. When a kernel is constructed, it stores its actual derived type's ID. This is used later for type-safe downcasting.

### KernelDerived Concept

```cpp
template <typename T>
concept KernelDerived = std::is_base_of_v<Kernel, T>;
```

Only classes derived from `Kernel` can be registered with the dispatcher. This prevents accidentally registering a random object.

### Registration

```cpp
template <KernelDerived U>
void register_kernel(U* func_class, T op, DispatchKey dispatch_key) {
    m_registry[static_cast<int>(op)][static_cast<int>(dispatch_key)] = func_class;
}
```

Registration is straightforward: take a kernel pointer and put it in the right slot. The `KernelDerived` concept ensures type safety.

Example from `kernels_registrar.h`:

```cpp
// Register the ConstantCPU kernel for the "constant" operation on CPU
dispatcher.register_kernel(&constant_cpu, UtilityOps::constant, DispatchKey::CPU);
```

This means: "When someone asks for the `constant` operation on a `CPU` device, use `constant_cpu`."

### Lookup

```cpp
template <KernelDerived U>
U* lookup(T op, DispatchKey dispatch_key) const {
    Kernel* kernel = m_registry[static_cast<int>(op)][static_cast<int>(dispatch_key)];
    // Verify CTTI type ID matches U
    assert(kernel->derived_id() == ctti::type_id<U>());
    return static_cast<U*>(kernel);
}
```

Lookup is where the type safety really matters. When you call `lookup<ConstantCPU>(op, key)`:
1. The raw `Kernel*` is retrieved from the registry
2. The stored CTTI type ID is compared against the expected type `U`
3. If they match, a safe `static_cast` is performed
4. If they don't match, the assertion fires — you're looking for the wrong kernel type

This catches bugs like registering a `PrintCPU` kernel in the `constant` slot and then trying to use it as a `ConstantCPU`. Without the CTTI check, you'd get a silent `static_cast` to the wrong type and undefined behavior.

### Why CTTI Instead of dynamic_cast?

`dynamic_cast` is the standard C++ way to safely downcast. But:
- It requires RTTI (Runtime Type Information), which some projects disable for performance
- It has overhead — hash table lookups for the type info
- It returns `nullptr` on failure, which is easy to forget to check

CTTI gives us compile-time type IDs that are just integer comparisons. It's faster, works without RTTI, and the assertion makes failures loud and immediate.

---

## Dispatch Keys

The dispatch keys are defined in `core/enums.h`:

```cpp
enum class DispatchKey {
    CPU,              // CPU forward pass
    CUDA,             // GPU forward pass
    CPU_Autodiff,     // CPU backward pass
    CUDA_Autodiff,    // GPU backward pass
    DispatchKeysCount // Array sizing sentinel
};
```

A dispatch key represents a combination of device and mode (forward/backward). The tensor's dispatch key is set based on its device and whether it requires gradients:

| Device | Requires Grad | Dispatch Key |
|--------|--------------|--------------|
| CPU | No | `CPU` |
| CPU | Yes | `CPU_Autodiff` |
| CUDA | No | `CUDA` |
| CUDA | Yes | `CUDA_Autodiff` |

When an operation is dispatched, it uses the tensor's dispatch key to select the appropriate column in the registry. This means the same `add` operation enum value maps to different kernels depending on whether you're on CPU or GPU, and whether autograd is active.

### Why Separate Forward and Autodiff Keys?

I could have just had `CPU` and `CUDA` dispatch keys and checked the `requires_grad` flag separately. But separate keys have advantages:

1. **Different kernel implementations**: The autodiff kernel doesn't just compute the forward result — it also records the computation graph. Bundling this into the kernel means the graph recording happens at the same level as the computation, which is cleaner than wrapping forward kernels with autograd logic.

2. **No branching at dispatch time**: The dispatch key is computed once during tensor construction. Every subsequent dispatch is a direct table lookup — no `if (requires_grad)` checks.

3. **Future extensibility**: If I add quantized operations or distributed computation, they'd get their own dispatch keys. The 2D table naturally extends.

---

## How Operations Use the Dispatcher

Here's the typical pattern for implementing an operation:

### 1. Define the Kernel Abstract Class

```cpp
class ConstantAbstract : public Kernel {
public:
    ConstantAbstract() : Kernel(ctti::type_id<ConstantAbstract>()) {}
    virtual void setConstant(Tensor& tensor, /* args */) = 0;
};
```

### 2. Implement Device-Specific Kernels

```cpp
class ConstantCPU final : public ConstantAbstract {
public:
    void setConstant(Tensor& tensor, /* args */) override {
        // CPU-specific implementation
    }
};
```

### 3. Register with Dispatcher

```cpp
ConstantCPU constant_cpu;
dispatcher.register_kernel(&constant_cpu, UtilityOps::constant, DispatchKey::CPU);
```

### 4. Use in Tensor Code

```cpp
void Tensor::setConstant(float value) {
    auto* kernel = m_dispatcher.lookup<ConstantAbstract>(UtilityOps::constant, m_dispatch_key);
    kernel->setConstant(*this, value);
}
```

The tensor doesn't know or care whether it's on CPU or GPU. It just looks up the kernel and calls it.

---

## Kernels Registrar

`kernels_registrar.h` is the central location where all kernels are instantiated and registered with the dispatcher. It serves as the single point of truth for the kernel registry:

```cpp
// Create kernel instances
StorageBackendCPU storage_backend_cpu;
ConstantCPU constant_cpu;
InitializeCPU initialize_cpu;
PrintCPU print_cpu;
StorageCopyCPU storage_copy_cpu;

// Register all with the dispatcher
dispatcher.register_kernel(&storage_backend_cpu, UtilityOps::storage_backend, DispatchKey::CPU);
dispatcher.register_kernel(&constant_cpu, UtilityOps::constant, DispatchKey::CPU);
// ... etc.
```

I centralized registration rather than having each kernel register itself because:
- It's easier to see the full registry at a glance
- Registration order is explicit and predictable
- There's no static initialization order fiasco risk

---

## Design Decisions Summary

| Decision | Rationale |
|----------|-----------|
| 2D array registry (op × dispatch_key) | O(1) lookup, simple and cache-friendly |
| `ValidEnum` concept with `Count` pattern | Compile-time sizing of the registry array |
| CTTI for type-safe downcasting | Faster than `dynamic_cast`, works without RTTI, assertion on mismatch |
| `KernelDerived` concept constraint | Prevents registering non-kernel objects |
| Separate forward/autodiff dispatch keys | Clean separation of forward and backward kernels, no runtime branching |
| Centralized kernel registration | Single source of truth, predictable initialization |
| Inheritance-based kernel hierarchy | Abstract kernels define the interface, concrete kernels implement per-device |

---

## File Reference

| File | Lines | Purpose |
|------|-------|---------|
| `dispatcher.h` | ~92 | Generic 2D dispatcher template with registration and lookup |
| `kernel_base.h` | ~15 | Base `Kernel` class with CTTI type identification |
| `kernels_registrar.h` | ~20 | Central kernel instantiation and registration |
