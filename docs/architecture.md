# Architecture Overview

This document lays out the big picture of how Forge is structured and why I made the choices I did. If you're trying to understand the framework as a whole before diving into specific subsystems, start here.

---

## The Core Idea

Forge is a tensor computation framework built from scratch in C++20. The goal isn't to compete with PyTorch or TensorFlow on feature count — it's to build a framework with clean architecture, real engineering behind the memory management, and a design that makes it straightforward to extend.

Most tensor libraries start by hacking together a tensor class with some math operations and then bolt on autograd and GPU support later. That leads to messy abstractions and special cases everywhere. I wanted to do it the other way: build the infrastructure right from day one — the dispatcher, the memory pool, the computation graph — and then add operations on top of a solid foundation.

---

## High-Level Architecture

Here's how the major pieces fit together:

```
┌──────────────────────────────────────────────────────────┐
│                      User Code                           │
│            (Tensor creation, operations, backward)       │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│                    Tensor Class                          │
│     (core/tensor/tensor.h)                               │
│     - Shape, strides, dtype, device metadata             │
│     - Shared pointer to storage                          │
│     - Shared pointer to autograd node                    │
│     - Dispatch key for routing operations                │
└────────┬──────────────┬──────────────────┬───────────────┘
         │              │                  │
         ▼              ▼                  ▼
┌────────────┐  ┌──────────────┐  ┌────────────────────┐
│  Storage   │  │  Dispatcher  │  │    Autograd        │
│  Backend   │  │  System      │  │    Engine          │
│            │  │              │  │                    │
│ - Abstract │  │ - 2D lookup  │  │ - Node abstraction │
│   storage  │  │   table      │  │ - Computation      │
│ - CPU impl │  │ - Kernel     │  │   graph building   │
│ - Memory   │  │   registry   │  │ - Backward pass    │
│   pool     │  │ - Type-safe  │  │ - Gradient         │
│ (radix     │  │   dispatch   │  │   propagation      │
│  tree)     │  │              │  │                    │
└────────────┘  └──────┬───────┘  └────────────────────┘
                       │
                       ▼
              ┌────────────────┐
              │   Op Kernels   │
              │                │
              │ - Utility ops  │
              │ - Math ops     │
              │   (planned)    │
              └────────────────┘
```

The tensor is at the center. It holds references to its storage (where the data lives in memory), knows its dispatch key (which tells the system where to route operations), and optionally holds a node in the computation graph (for autograd).

When you do something with a tensor — initialize it, reshape it, print it — the tensor doesn't do the work itself. It asks the dispatcher to look up the right kernel for the operation given the tensor's device and type, and that kernel does the actual computation.

---

## Why This Layering Matters

### Tensor ≠ Storage

The tensor class is metadata + a pointer to storage. Multiple tensors can share the same storage — that's how views work. When you do `tensor[1]`, you get back a new tensor object that points into the same underlying memory with adjusted strides and shape. No data copy happens.

This separation is critical because:
- It makes views zero-cost. Slicing a tensor is just creating a new metadata wrapper.
- It allows different storage backends (CPU, CUDA in the future) without changing the tensor interface.
- It enables shared ownership semantics through `std::shared_ptr`, so storage lives as long as any tensor references it.

### Tensor ≠ Operations

The tensor doesn't contain any operation logic. It doesn't know how to add, multiply, or compute gradients. All of that lives in kernel classes that are registered with the dispatcher.

This is a deliberate architectural choice. If you put operation logic in the tensor class, you end up with a 10,000-line monolith. By keeping operations separate:
- Adding a new operation means writing a new kernel class and registering it. You never touch `tensor.h`.
- Supporting a new device (like CUDA) means writing new kernels for that device and registering them with the appropriate dispatch key. Again, the tensor class doesn't change.
- Testing operations is isolated — you test the kernel, not the entire tensor machinery.

### Dispatcher = The Routing Layer

The dispatcher is what connects tensors to operations. It's a 2D lookup table indexed by (operation, dispatch_key). When you need to, say, initialize a tensor with constants, the code asks the dispatcher: "give me the `constant` kernel for `CPU`." The dispatcher returns a pointer to the right kernel, and the code calls it.

This pattern is inspired by how PyTorch's dispatcher works, but simplified and built with C++20 features (concepts, compile-time type information) instead of runtime type registries and string-based lookups.

---

## Design Decisions

### Why C++20?

C++20 gives us concepts, which are a game-changer for template-heavy code like this. Instead of relying on SFINAE tricks or `static_assert` with cryptic error messages, I can write constraints directly:

```cpp
template <typename T>
concept TensorStorageType = std::is_trivially_copyable_v<T> && std::is_standard_layout_v<T>;
```

This means if you try to create a `StorageCPU<std::string>`, you get a clear error at the concept level, not a wall of template instantiation errors. Concepts also make the dispatcher's type safety much cleaner — `KernelDerived` and `ValidEnum` constraints are readable and enforceable.

### Why Header-Only?

A few reasons:
1. **Zero build complexity for users.** Include the header, compile your code, done. No need to build and link a separate library.
2. **Templates require it anyway.** Most of the code is heavily templated (the dispatcher, storage, autograd nodes), so the implementations need to be in headers regardless.
3. **Compiler optimization.** With everything in headers, the compiler can see the full picture and inline aggressively. For a performance-critical tensor library, this matters.

The tradeoff is longer compile times, but for a framework of this size, it's not a problem.

### Why Shared Pointers for Storage?

Tensors use `std::shared_ptr<StorageAbstract>` for their underlying storage. This enables:
- **View semantics**: Multiple tensors can point to the same storage. When one goes out of scope, the storage survives as long as any tensor references it.
- **Safe ownership**: No manual memory management at the tensor level. The storage is freed when the last tensor referencing it is destroyed.
- **Polymorphism**: The tensor holds a base pointer, so it can work with CPU storage, GPU storage, or any future backend without knowing the concrete type.

I chose shared pointers over unique pointers specifically because views are a first-class feature. A unique pointer would force moves on every view creation, which defeats the purpose.

### Why a Custom Memory Pool?

Calling `malloc` and `free` for every tensor allocation is expensive, especially for the small-to-medium allocations that are common in neural network workloads (think: intermediate activations, gradient buffers). A memory pool amortizes allocation cost by:
- Pre-allocating chunks of memory
- Dividing them into size-classed bins
- Reusing freed blocks instead of returning them to the OS

The radix-tree lookup for deallocation is a specific choice — it gives O(1) pointer-to-bin mapping without storing metadata adjacent to the allocated block (which would mess with alignment and cache behavior). See [storage.md](storage.md) for the full deep-dive.

### Why a Dispatcher Instead of Virtual Functions?

Virtual functions are the standard C++ approach for runtime polymorphism, but they have problems for this use case:
- **Single dispatch**: Virtual functions dispatch on one type (the object). Tensor operations need to dispatch on both the operation type *and* the device/dtype — that's multi-dispatch.
- **Rigid hierarchy**: Adding a new operation with virtual functions means adding a new virtual method to a base class. That changes the vtable and forces recompilation of everything.
- **No compile-time verification**: You can forget to override a virtual method and get silent bugs.

The dispatcher approach gives us a flat registry that can be extended by registering new kernels without modifying any existing classes. Combined with CTTI for type verification, it catches mismatches at lookup time rather than at runtime through UB.

---

## Data Flow: Forward Pass

Here's what happens when you create and use a tensor:

1. **Construction**: `Tensor t {{3, 4}}` creates a tensor with shape [3, 4]. The constructor asks the dispatcher for a `storage_backend` kernel, which allocates storage through the memory pool.

2. **Initialization**: `t = {{1,2,3,4}, {5,6,7,8}, {9,10,11,12}}` triggers the initializer list overload. The shape is inferred at compile time through recursive template type traits. The data is copied into storage via the `initializers` kernel.

3. **Operations** (future): `auto c = a + b` would look up the `add` kernel in the dispatcher for the appropriate dispatch key, execute it, and create a new tensor with the result. If autograd is enabled, it would also attach a computation graph node.

4. **View creation**: `t[1]` creates a new tensor pointing into `t`'s storage at an offset, with reduced dimensionality. No data copy.

## Data Flow: Backward Pass

1. **Graph construction**: During the forward pass, if `requires_grad` is set, each operation attaches a `Node` to the output tensor via `attach_node`. The node stores references to the parent tensors and a pointer to the gradient kernel.

2. **Backward trigger**: Calling `tensor.backward()` starts the reverse pass from that tensor's node.

3. **Gradient computation**: Each node calls its stored gradient function (`m_gradFn->compute_grads()`), which computes gradients with respect to its parents and stores them.

4. **Recursive propagation**: After computing local gradients, the node calls `backward()` on each parent's node, propagating the chain rule through the graph.

---

## Module Dependencies

```
enums.h ◄──── Everything depends on this (Dtype, Device, DispatchKey)
    │
    ├──► Storage (needs Dtype for type dispatch, Device for backend selection)
    │
    ├──► Dispatcher (needs DispatchKey for routing, ValidEnum concept)
    │
    ├──► Autograd (needs DispatchKey and operation enums)
    │
    └──► Tensor (aggregates everything above)
```

There's a clear dependency direction: enums → storage/dispatcher/autograd → tensor. Nothing below depends on something above. This avoids circular dependencies and keeps the include graph clean.

---

## What's Intentionally Missing

Some things are absent by design at this stage:

- **CUDA support**: The enum values and dispatch keys exist, but no CUDA kernels are implemented. The architecture supports it — you'd register CUDA kernels with `DispatchKey::CUDA` and implement a `StorageCUDA` class. But rushing into CUDA before the CPU path is solid would be premature.

- **Math operations**: `math_ops.h` is an empty placeholder. The dispatcher and kernel infrastructure is ready — I just haven't written the actual math kernels yet. When I do, they'll slot in without any architectural changes.

- **Broadcasting**: Not implemented yet. This is a tensor-level feature that will need careful design to integrate with the dispatcher and autograd graph.

These aren't oversights — they're intentional scope boundaries for the current phase of development. The infrastructure is built to accommodate them cleanly when the time comes.
