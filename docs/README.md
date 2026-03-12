# Forge Documentation

Welcome to the Forge documentation. This is a comprehensive guide covering every subsystem, every design decision, and the reasoning behind the way things are built. I wrote Forge as a modern C++20 header-only tensor framework — something that's clean, extensible, and built from scratch with real engineering principles rather than hacks on top of hacks.

If you're reading this, you probably want to understand how Forge works under the hood. Each document here dives deep into a specific subsystem. I'd recommend starting with the architecture overview and then exploring whichever subsystem interests you most.

---

## Documentation Index

| Document | What It Covers |
|----------|---------------|
| [Architecture Overview](architecture.md) | The big picture — how every piece fits together, and the core design philosophy driving the project |
| [Tensor Subsystem](tensor.md) | The `Tensor` class itself — construction, initialization, views, reshape, type system, and how it ties into everything else |
| [Storage & Memory Pool](storage.md) | How memory is actually managed — the storage abstraction, CPU storage backend, and the radix-tree memory pool allocator |
| [Autograd Engine](autograd.md) | The reverse-mode automatic differentiation system — computation graphs, nodes, backward pass, and gradient propagation |
| [Dispatcher & Kernel System](dispatcher.md) | How operations get routed to the right kernel — the dispatcher pattern, dispatch keys, and the registration mechanism |
| [Operations & Kernels](operations.md) | The concrete kernel implementations — storage backends, initialization, constants, printing, copying, and the type dispatch macro |
| [Build System & Dependencies](build.md) | How to build the project, what external libraries are used, platform support, and compiler requirements |
| [Roadmap & Future Work](roadmap.md) | What's done, what's not, what's planned, and where contributions would help most |

---

## Quick Overview

Forge is a header-only C++20 tensor framework. The entire library lives in header files under `core/`, and you include what you need. There's no separate compilation step for the library itself — just include the headers and you're good.

The core idea: build a tensor library that's properly engineered from day one. That means a real dispatcher system (not just if-else chains), a proper memory pool (not just malloc/free everywhere), a computation graph that's built at the type level (not runtime string matching), and storage abstractions that allow different backends without rewriting tensor logic.

Right now, the framework has solid infrastructure — tensors, storage, memory management, dispatcher, and the autograd graph skeleton. Math operations and CUDA support are the next big milestones.

### Project Structure

```
Forge/
├── core/                          # The entire framework lives here
│   ├── enums.h                    # Global type definitions (Dtype, Device, DispatchKey)
│   ├── autograd/                  # Reverse-mode automatic differentiation
│   │   ├── node_abstract.h        # Abstract node interface
│   │   ├── node.h                 # Templated computation graph node
│   │   └── attach_node.h          # Utility for wiring nodes into the graph
│   ├── ops/                       # Operations infrastructure
│   │   ├── Dispatcher/            # The dispatcher pattern implementation
│   │   │   ├── dispatcher.h       # Generic dispatcher with 2D registry
│   │   │   └── kernel_base.h      # Base class for all kernels
│   │   └── Ops Kernels/           # Concrete operation implementations
│   │       ├── kernels_registrar.h # Kernel registration entry point
│   │       ├── utility_ops.h      # Storage, init, print, copy kernels
│   │       └── math_ops.h         # Math operations (placeholder)
│   └── tensor/                    # Tensor data structure
│       ├── tensor.h               # The Tensor class (main API surface)
│       └── Storage/               # Memory management
│           ├── storage_abstract.h # Storage interface
│           ├── storage_cpu.h      # CPU storage implementation
│           ├── memory_pool_abstract.h # Memory pool interface
│           └── memory_pool_cpu.h  # Radix-tree based CPU memory pool
├── primitives/                    # Primitive type definitions (placeholder)
│   └── primitives.h
├── tests/                         # Test suite
│   └── main.cpp                   # Basic tensor functionality tests
├── CMakeLists.txt                 # Build configuration
└── LICENSE                        # MIT License
```

---

## Philosophy

A few principles that guide how I build Forge:

1. **Type safety at compile time, flexibility at runtime.** C++20 concepts enforce constraints where it matters (kernel types, storage layouts, enum validity), while the dispatcher gives runtime flexibility for device/dtype selection.

2. **No hidden allocations.** Every allocation goes through the memory pool. You can see exactly where memory comes from and where it goes.

3. **Operations are decoupled from data.** The tensor doesn't know how to multiply matrices — the dispatcher routes that to the right kernel. This keeps the tensor class clean and makes it trivial to add new operations or backends.

4. **Header-only by design.** No build complexity for users. Include the header, and you're done. The compiler handles the rest.

5. **Build infrastructure first, features second.** It's tempting to rush into implementing matmul and convolutions, but without a proper dispatcher, memory pool, and autograd graph, you end up with a mess. Forge builds the foundation first.
