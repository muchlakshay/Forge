# Tensor Subsystem

The `Tensor` class is the central data structure of Forge. It's defined in `core/tensor/tensor.h` and is the primary thing users interact with. Everything else — storage, dispatcher, autograd — exists to serve the tensor.

This document covers how the tensor works internally, the design decisions behind it, and what every significant piece of the implementation does.

---

## Overview

A `Tensor` in Forge is essentially metadata plus a shared pointer to storage. The tensor itself doesn't hold raw data — it holds a shape, strides, dtype, device, and a reference to a `StorageAbstract` that contains the actual bytes. This separation is fundamental to how views, cloning, and device abstraction work.

The tensor also carries an optional autograd node (for backpropagation) and a dispatch key (for routing operations to the right kernel).

---

## Class Layout

Here's what lives inside a `Tensor` object:

```cpp
class Tensor {
    Device m_device;                          // CPU or CUDA
    Dtype m_dtype;                            // float32, float64, int16, etc.
    std::shared_ptr<StorageAbstract> m_storage;  // Where the data actually lives
    std::vector<std::size_t> m_shape;         // Dimensions (e.g., {3, 4, 5})
    std::vector<std::size_t> m_strides;       // Element strides for each dimension
    std::size_t m_size;                       // Total number of elements
    bool m_need_grads;                        // Whether to track gradients
    DispatchKey m_dispatch_key;               // How to route operations
    std::shared_ptr<NodeAbstract> m_node;     // Computation graph node
    Tensor* m_grads;                          // Pointer to gradient tensor
};
```

Every field has a reason for being there. Let me go through the important ones.

### Shape and Strides

`m_shape` stores the dimensions — for a 3×4 matrix, it's `{3, 4}`. `m_strides` stores how many elements you need to skip to move one step along each dimension. For a contiguous 3×4 matrix, strides are `{4, 1}` — moving along the first dimension (rows) means jumping 4 elements, moving along the second dimension (columns) means jumping 1 element.

Strides are what make views work. When you slice a tensor with `tensor[i]`, the resulting tensor has the same storage but different strides and a different starting offset. The strides tell you how to traverse the sliced data correctly even though the underlying memory hasn't changed.

I use `std::vector` for shape and strides rather than fixed-size arrays because tensors can have arbitrary dimensionality. Yes, most neural network tensors are 2D-4D, but I didn't want to hard-code a dimension limit into the core data structure.

### Dispatch Key

The dispatch key determines which set of kernels gets used for operations on this tensor. For a CPU tensor without autograd, it's `DispatchKey::CPU`. For a CPU tensor that needs gradient tracking, it's `DispatchKey::CPU_Autodiff`.

The dispatch key is set during construction based on the device and whether gradients are needed. It's a simple mapping:
- CPU + no grads → `CPU`
- CPU + grads → `CPU_Autodiff`
- CUDA + no grads → `CUDA`
- CUDA + grads → `CUDA_Autodiff`

This means every tensor carries its routing information with it. When you pass a tensor to an operation, the operation can look at the dispatch key and immediately know which kernel to use, without checking device and grad flags separately.

---

## Construction

Tensors can be created in several ways, each serving a different use case.

### Shape-based Construction

```cpp
Tensor t {{3, 4}};  // 3×4 tensor, uninitialized
```

This creates a tensor with the given shape. The constructor:
1. Computes strides from the shape (row-major / C-contiguous layout)
2. Asks the dispatcher for a storage backend kernel
3. Allocates storage through that kernel (which ultimately goes through the memory pool)

The stride computation walks the shape in reverse, multiplying as it goes. For shape `{3, 4, 5}`, strides are `{20, 5, 1}`.

### Initializer List Construction

```cpp
Tensor t = {{1, 2, 3}, {4, 5, 6}};  // 2×3 tensor
```

This is where the template metaprogramming gets interesting. The tensor supports nested initializer lists up to 4 levels deep (1D through 4D). The shape is inferred at compile time through a recursive type trait:

```cpp
template <typename T>
struct ShapeInferer;

// Base case: scalar
template <>
struct ShapeInferer<float> {
    static void infer(auto& shape, auto list) { /* terminal */ }
};

// Recursive case: nested list
template <typename T>
struct ShapeInferer<std::initializer_list<T>> {
    static void infer(auto& shape, auto list) {
        shape.push_back(list.size());
        ShapeInferer<T>::infer(shape, *list.begin());
    }
};
```

The actual implementation uses variadic initializer list overloads (`init_list_1d`, `init_list_2d`, etc.) but the principle is the same — the compiler figures out the shape from the nesting depth and sizes.

After inferring the shape, the data is copied into storage using the `initializers` kernel from the dispatcher, which handles type conversion if the source type doesn't match the tensor's dtype.

### Factory Methods

```cpp
auto t1 = Tensor::Constant({3, 4}, 5.0f);   // All 5.0
auto t2 = Tensor::Ones({3, 4});              // All 1.0
auto t3 = Tensor::Zeros({3, 4});             // All 0.0
auto t4 = Tensor::Range({12}, 0.0f, 1.0f);   // Linspace-like
auto t5 = Tensor::Random({3, 4});            // Random values
```

These are static methods that create a tensor and fill it in one step. They all follow the same pattern: construct the tensor, then call the appropriate kernel through the dispatcher to fill the storage.

`FromHostPtr` is a special case — it creates a tensor that views existing host memory rather than allocating new storage. This is useful for wrapping data from other libraries without copying.

---

## Views and Indexing

One of the most important tensor operations is indexing, and Forge implements it as a view:

```cpp
Tensor row = tensor[1];  // View into the second row
```

When you index a tensor, you get back a new tensor that:
- Points to the **same** underlying storage (no copy)
- Has a **reduced** shape (one fewer dimension)
- Has adjusted strides and an offset into the original storage

This is implemented through the storage's view mechanism. The `operator[]` creates a new `StorageCPU` that points into the parent's memory at the appropriate offset. The parent's storage stays alive because both the original tensor and the view hold shared pointers to it (or the view's storage holds its own pointer to the data — the memory pool handles lifetime).

Views are essential for performance. Slicing a 1000×1000 tensor to get a row should be O(1), not O(1000). And since the view shares memory with the original, modifications through the view are visible in the original tensor (and vice versa).

---

## Reshape

```cpp
auto reshaped = tensor.reshape(12);          // 1D
auto reshaped = tensor.reshape(3, 4);        // 2D
auto reshaped = tensor.reshape(2, 3, 2);     // 3D
```

Reshape is implemented as a variadic template method that accepts up to 4 dimension arguments. It:
1. Validates that the total element count matches (via a fold expression: `(dims * ...) == m_size`)
2. Creates a new tensor sharing the same storage
3. Sets the new shape and recomputes strides

Since the storage is shared, reshape is also a view operation — zero cost, no data movement. The only constraint is that the tensor must be contiguous for reshape to work correctly (non-contiguous tensors would need a copy first, which isn't implemented yet).

I chose to implement reshape as a template with variadic arguments rather than taking a vector because:
- The number of dimensions is typically known at compile time
- Fold expressions give compile-time validation of the total size
- It's cleaner syntax: `reshape(3, 4)` vs `reshape({3, 4})`

The limit of 4 dimensions matches the Eigen tensor backend's maximum rank (which is used under the hood for mapping tensors to Eigen types).

---

## Type System

Forge supports six data types, defined in `enums.h`:

| Dtype | C++ Type | Size |
|-------|----------|------|
| `float16` | `Eigen::half` | 2 bytes |
| `float32` | `float` | 4 bytes |
| `float64` | `double` | 8 bytes |
| `int16` | `int16_t` | 2 bytes |
| `int32` | `int32_t` | 4 bytes |
| `int64` | `int64_t` | 8 bytes |

The type is stored as a `Dtype` enum in the tensor and used by the `DISPATCH_ALL_TYPES` macro to select the right template instantiation at runtime. When you create a tensor, it defaults to `float32` unless you specify otherwise.

The type dispatch macro is a critical piece — it takes a dtype enum value and generates a switch statement that instantiates the correct template. See [operations.md](operations.md) for details on how this works.

---

## Eigen Integration

Forge uses Eigen's unsupported Tensor module for the underlying numerical storage format compatibility. The `as_eigen<T>()` method maps a Forge tensor's raw memory to an Eigen 4D TensorMap:

```cpp
template <typename T>
Eigen::TensorMap<Eigen::Tensor<T, 4>> as_eigen() {
    // Maps the raw data pointer to an Eigen tensor view
}
```

This always produces a 4D Eigen tensor, padding missing dimensions with 1. So a 2D [3, 4] Forge tensor maps to a [3, 4, 1, 1] Eigen tensor. This standardization simplifies the kernel implementations — they always work with 4D tensors.

I chose Eigen's tensor module over writing my own numerical backend because Eigen is battle-tested, SIMD-optimized, and handles the numerical heavy lifting well. Forge's job is to provide the framework — dispatching, autograd, memory management — not to reimplement matrix multiplication.

---

## Cloning and Copying

```cpp
auto cloned = tensor.clone();  // Deep copy
```

`clone()` creates a completely independent copy of the tensor — new storage, new data, same shape and strides. It uses the `storage_copy` kernel through the dispatcher.

This is distinct from view operations (indexing, reshape) which share storage. The rule is:
- **Views** share storage. Changes propagate both ways.
- **Clones** own separate storage. They're independent after creation.

The `copy()` method is similar but is used for assignment — copying one tensor's data into another's existing storage.

---

## Printing

Forge overloads `operator<<` for tensors, so you can do:

```cpp
std::cout << tensor << std::endl;
```

The printing is handled by the `PrintCPU` kernel, which recursively formats the tensor respecting its shape and strides. It handles multi-dimensional tensors by printing nested brackets, with proper indentation and line breaks.

The formatting is aware of the tensor's actual memory layout (via strides), so it correctly prints non-contiguous tensors (like views) without needing to materialize them first.

---

## Design Decisions Summary

| Decision | Rationale |
|----------|-----------|
| Metadata + shared pointer to storage | Enables zero-cost views, polymorphic backends, safe ownership |
| Dispatch key stored in tensor | Each tensor carries its routing info — no need for external context |
| Strides as vectors | Supports arbitrary dimensionality without compile-time limits |
| Initializer lists up to 4D | Covers the common case (1D-4D) with compile-time shape inference |
| Factory methods as static functions | Clear, explicit construction semantics — you know what you're getting |
| Reshape via variadic templates | Compile-time validation, clean syntax, matches Eigen's 4D limit |
| Eigen integration for numerics | Leverages optimized library for computation, Forge handles the framework |
