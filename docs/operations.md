# Operations & Kernels

This document covers the concrete operation implementations — the kernels that actually do work on tensors. These live in `core/ops/Ops Kernels/` and are registered with the dispatcher to handle specific operations on specific devices.

Right now, Forge has utility operations (storage, initialization, constants, printing, copying) fully implemented. Math operations are the next milestone.

---

## The Type Dispatch Macro

Before diving into individual kernels, I need to explain the `DISPATCH_ALL_TYPES` macro because every kernel uses it. This is the mechanism that bridges the gap between the runtime `Dtype` enum and the compile-time template parameter.

```cpp
#define DISPATCH_ALL_TYPES(DTYPE, DEVICE, ...)
```

The problem: tensor dtype is a runtime value (`Dtype::float32`, `Dtype::float64`, etc.), but the storage is a template (`StorageCPU<float>`, `StorageCPU<double>`, etc.). We need to go from runtime to compile-time.

The macro generates a switch statement:

```cpp
switch (DTYPE) {
    case Dtype::float32: {
        using scalar_t = float;
        __VA_ARGS__  // Your code, with scalar_t available
        break;
    }
    case Dtype::float64: {
        using scalar_t = double;
        __VA_ARGS__
        break;
    }
    case Dtype::int32: {
        using scalar_t = int32_t;
        __VA_ARGS__
        break;
    }
    // ... all dtypes
}
```

Inside the macro body, `scalar_t` is a type alias for the C++ type corresponding to the dtype. This lets kernel code be generic:

```cpp
DISPATCH_ALL_TYPES(tensor.dtype(), tensor.device(), {
    auto* data = static_cast<scalar_t*>(tensor.data());
    // Now 'data' is correctly typed
});
```

I chose a macro over other approaches (like `std::visit` with a variant) because:
1. It's zero overhead — the compiler generates a jump table, same as you'd write by hand
2. The syntax is clean — you write the kernel code once, and it gets instantiated for all types
3. It's how PyTorch does it (`AT_DISPATCH_ALL_TYPES`), and it's a proven pattern

The downside is that it's a macro, with all the debugging and readability issues that implies. But the alternative — manually writing switch statements in every kernel — is worse.

---

## Utility Operations

These are the foundational operations that every tensor needs. They're registered under the `UtilityOps` enum:

```cpp
enum class UtilityOps {
    storage_backend,  // Create/manage storage
    initializers,     // Initialize from data
    constant,         // Fill with a constant value
    print,            // Format and display
    storage_copy,     // Deep copy storage
    Count             // Sentinel for dispatcher sizing
};
```

### 1. Storage Backend

**Abstract class**: `StorageBackend`
**CPU implementation**: `StorageBackendCPU`

This kernel creates and manages the underlying storage for tensors. It has two main operations:

#### `getStorageBackend()`

Creates a new storage object for a given dtype and byte count. This is what gets called when you construct a tensor — the tensor asks the dispatcher for a storage backend, and the kernel allocates the right type of storage.

```cpp
DISPATCH_ALL_TYPES(dtype, device, {
    return std::make_shared<StorageCPU<scalar_t>>(nbytes);
});
```

The type dispatch macro handles the mapping from `Dtype::float32` → `StorageCPU<float>`, `Dtype::float64` → `StorageCPU<double>`, etc.

#### `setView()`

Creates a view into existing storage — used by `operator[]` to implement tensor slicing. The view points to a sub-region of the parent's memory.

```cpp
DISPATCH_ALL_TYPES(dtype, device, {
    return std::make_shared<StorageCPU<scalar_t>>(
        static_cast<scalar_t*>(parent_data) + offset, view_nbytes
    );
});
```

The returned storage is in view mode (it won't free the memory on destruction).

### Why a Kernel for Storage?

Storage creation could be hardcoded in the tensor constructor. But making it a kernel means:
- Different devices get different storage implementations naturally
- The tensor constructor doesn't need type-specific code
- Adding a new storage backend (like pinned memory, or unified memory) is just registering a new kernel

---

### 2. Constant Fill

**Abstract class**: `ConstantAbstract`
**CPU implementation**: `ConstantCPU`

Fills a tensor with a constant value. Used by `Tensor::Constant()`, `Tensor::Ones()`, and `Tensor::Zeros()`.

```cpp
void setConstant(Tensor& tensor, DtypeVariant value) {
    DISPATCH_ALL_TYPES(tensor.dtype(), tensor.device(), {
        scalar_t typed_value = std::get<scalar_t>(value);
        auto* data = static_cast<scalar_t*>(tensor.data());
        for (size_t i = 0; i < tensor.size(); ++i) {
            data[i] = typed_value;
        }
    });
}
```

The value is passed as a `DtypeVariant` — a `std::variant` that can hold any of the supported scalar types. This allows type-safe value passing without templates at the call site.

I used `std::variant` here rather than a template parameter because the constant value's type might not match the tensor's dtype. For example, you might pass `5.0` (double) to a `float32` tensor. The kernel handles the conversion inside the dispatch.

---

### 3. Initialization

**Abstract class**: `InitializeAbstract`
**CPU implementation**: `InitializeCPU`

Copies data from a source pointer into tensor storage, with type conversion. This is what gets called when you do `tensor = {{1, 2, 3}, {4, 5, 6}}`.

```cpp
void initialize(void* dest, const void* src, Dtype dest_dtype, Dtype src_dtype, size_t count) {
    DISPATCH_ALL_TYPES(dest_dtype, Device::CPU, {
        using dest_t = scalar_t;
        DISPATCH_ALL_TYPES(src_dtype, Device::CPU, {
            using src_t = scalar_t;
            auto* d = static_cast<dest_t*>(dest);
            auto* s = static_cast<const src_t*>(src);
            for (size_t i = 0; i < count; ++i) {
                d[i] = static_cast<dest_t>(s[i]);
            }
        });
    });
}
```

Note the nested dispatch — both source and destination types are dispatched independently. This means every combination of source and destination types is handled (float→double, int→float, etc.). It's O(n²) code generation in the number of types, but the compiler eliminates dead branches, and the actual runtime is a single tight loop.

### Why Support Cross-Type Initialization?

In practice, you often have data in one type and want a tensor in another. Maybe your training data is `float64` but your model uses `float32`. Without cross-type initialization, you'd need an explicit cast step. With it, the tensor handles the conversion transparently.

---

### 4. Printing

**Abstract class**: `PrintAbstract`
**CPU implementation**: `PrintCPU`

Formats a tensor for display. This is what powers `std::cout << tensor`.

The printing kernel is more complex than you might expect because it needs to handle:
- Arbitrary dimensions (1D, 2D, 3D, 4D)
- Non-contiguous memory layouts (views with non-trivial strides)
- Proper bracket nesting and indentation
- Numeric formatting

The implementation uses a recursive approach that walks the tensor's shape and strides:

```cpp
void print(std::ostream& os, const Tensor& tensor) {
    DISPATCH_ALL_TYPES(tensor.dtype(), tensor.device(), {
        auto* data = static_cast<scalar_t*>(tensor.data());
        recursive_print(os, data, tensor.shape(), tensor.strides(), 0, 0);
    });
}
```

The recursive function handles each dimension by:
1. Opening a bracket `[`
2. Iterating along that dimension
3. Either printing a number (if at the innermost dimension) or recursing deeper
4. Closing the bracket `]`

Strides are used for indexing, so views print correctly — the printer follows the stride pattern to access the right elements, even if they're not contiguous in memory.

---

### 5. Storage Copy

**Abstract class**: `StorageCopyAbstract`
**CPU implementation**: `StorageCopyCPU`

Deep-copies tensor storage. Used by `Tensor::clone()`.

```cpp
std::shared_ptr<StorageAbstract> copy_storage(const Tensor& tensor) {
    DISPATCH_ALL_TYPES(tensor.dtype(), tensor.device(), {
        auto* src_storage = static_cast<StorageCPU<scalar_t>*>(tensor.storage().get());
        return std::make_shared<StorageCPU<scalar_t>>(src_storage->clone());
    });
}
```

This allocates fresh memory from the pool, copies the data byte-by-byte, and returns a new storage object. The clone is completely independent — modifying the original won't affect the copy.

---

## Math Operations (Planned)

`math_ops.h` exists as an empty placeholder. When math operations are implemented, they'll follow the exact same pattern:

1. Define an abstract kernel class (e.g., `AddAbstract`)
2. Implement CPU kernel (e.g., `AddCPU`)
3. Register with the dispatcher for `DispatchKey::CPU`
4. Optionally implement autodiff kernel for `DispatchKey::CPU_Autodiff`

The dispatcher and kernel infrastructure is ready — it's just the kernel implementations that need to be written. Operations I'm planning:

- **Element-wise**: add, subtract, multiply, divide, power
- **Reduction**: sum, mean, max, min
- **Linear algebra**: matmul, transpose
- **Activation functions**: relu, sigmoid, tanh, softmax
- **Loss functions**: MSE, cross-entropy

Each of these will be a kernel class with a forward method (for `DispatchKey::CPU`) and a backward method (for `DispatchKey::CPU_Autodiff`).

---

## The Kernel Hierarchy

All kernels follow a consistent inheritance pattern:

```
Kernel (kernel_base.h)
  ├── StorageBackend (abstract)
  │     └── StorageBackendCPU (concrete)
  ├── ConstantAbstract (abstract)
  │     └── ConstantCPU (concrete)
  ├── InitializeAbstract (abstract)
  │     └── InitializeCPU (concrete)
  ├── PrintAbstract (abstract)
  │     └── PrintCPU (concrete)
  └── StorageCopyAbstract (abstract)
        └── StorageCopyCPU (concrete)
```

The abstract level defines the interface (what the operation does). The concrete level provides the implementation (how it does it on a specific device). This two-level hierarchy means:
- The tensor code only knows about the abstract level
- Device-specific code is encapsulated in the concrete level
- Adding a new device means adding new concrete classes, not changing interfaces

---

## Design Decisions Summary

| Decision | Rationale |
|----------|-----------|
| `DISPATCH_ALL_TYPES` macro | Bridges runtime dtype to compile-time template, zero overhead |
| `std::variant` for constant values | Type-safe scalar passing without templates at call site |
| Nested dispatch for initialization | Supports all source→destination type combinations |
| Recursive printing | Handles arbitrary dimensionality with stride-aware indexing |
| Storage operations as kernels | Enables device-specific storage without tensor changes |
| Abstract + concrete kernel hierarchy | Clean separation of interface from device-specific implementation |
| Centralized registrar | Single place to see and manage all kernel registrations |

---

## File Reference

| File | Lines | Purpose |
|------|-------|---------|
| `utility_ops.h` | ~160 | All utility kernel implementations (storage, init, constant, print, copy) |
| `math_ops.h` | ~0 | Placeholder for future math operation kernels |
| `kernels_registrar.h` | ~20 | Kernel instantiation and dispatcher registration |
