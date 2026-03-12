# Roadmap & Future Work

This document captures where Forge is today, what's planned, and what's still open. It's as much a design document as a roadmap — I want to be transparent about what's done, what's stubbed, and what the priorities are.

---

## Current Status

### What's Solid

These subsystems are implemented, tested (at least at a basic level), and architecturally sound:

**Core Tensor Class** — Shape, strides, dtype, device, construction from initializer lists, factory methods (Constant, Ones, Zeros, Range, Random), views via operator[], reshape, clone, copy, Eigen integration. This is the foundation and it works.

**Storage Subsystem** — Abstract storage interface, CPU storage with owned and view modes, proper lifetime management through shared pointers. Storage creation is dispatched through the kernel system, so adding new backends is clean.

**Memory Pool** — The radix-tree based CPU memory pool is the most complex piece of the codebase. It handles allocation, deallocation, size-class binning, and cache prefetching. It's production-quality infrastructure, even if the rest of the framework is still growing.

**Dispatcher** — The 2D lookup table with CTTI-based type verification works. Kernels register cleanly, lookups are O(1), and type safety is enforced at dispatch time. This is the backbone for all operations.

**Utility Operations** — Storage backend, initialization, constant fill, printing, and storage copy are all implemented as proper kernels and registered with the dispatcher.

**Autograd Graph Structure** — The node abstraction, concrete templated node, and the attach_node mechanism are in place. The infrastructure for building computation graphs during the forward pass exists.

### What's Stubbed

These files exist but contain no real implementation:

- **`math_ops.h`** — Empty. No math operation kernels yet.
- **`primitives.h`** — Empty. Was meant for primitive type definitions but hasn't been needed yet.

### What's Missing Entirely

These features don't have any code yet:

- **CUDA support** — The enum values exist (`Device::CUDA`, `DispatchKey::CUDA`, `DispatchKey::CUDA_Autodiff`) but no CUDA storage, memory pool, or kernels exist.
- **Math operations** — No add, multiply, matmul, conv, activation functions, or loss functions.
- **Gradient kernels** — No backward implementations for any operations (because no operations exist yet).
- **Broadcasting** — Tensor operations on mismatched shapes aren't handled.
- **Optimizer** — No SGD, Adam, or other parameter update algorithms.
- **Serialization** — No way to save or load tensors to/from disk.
- **Error handling** — Minimal validation and error reporting.

---

## Priority Roadmap

Here's the order I plan to tackle things, and why.

### Phase 1: Core Math Operations

**Priority: Highest**

The framework infrastructure is in place, but without math operations, you can't actually compute anything useful. This is the critical gap.

**Planned operations**:
1. **Element-wise arithmetic**: add, subtract, multiply, divide
2. **Matrix multiplication**: matmul (this is the big one — the fundamental operation of neural networks)
3. **Reductions**: sum, mean, max, min along dimensions
4. **Activation functions**: ReLU, sigmoid, tanh, softmax

Each operation follows the same pattern:
- Define abstract kernel class
- Implement CPU kernel
- Register with dispatcher for `DispatchKey::CPU`
- Write corresponding backward kernel
- Register backward with `DispatchKey::CPU_Autodiff`

The dispatcher and kernel infrastructure make this systematic — it's implementation work, not architectural work.

### Phase 2: Autograd Completion

**Priority: High (depends on Phase 1)**

With math operations in place, the autograd engine needs:
1. **Gradient kernels** for every forward operation
2. **Gradient accumulation** — when a tensor is used in multiple operations, its gradients need to be summed, not overwritten
3. **Graph cleanup** — mechanism to detach or clear the computation graph after backward()
4. **Gradient clipping** — prevent exploding gradients during training

### Phase 3: Training Infrastructure

**Priority: Medium**

Once you can compute forward and backward passes:
1. **Optimizers**: SGD, Adam, AdaGrad, RMSProp
2. **Loss functions**: MSE, Cross-Entropy, Binary Cross-Entropy
3. **Learning rate schedulers**: Step, exponential, cosine annealing
4. **Parameter groups**: Applying different learning rates to different parts of a model

### Phase 4: Model Building Blocks

**Priority: Medium**

1. **Linear layers**: fully connected layers
2. **Convolution**: 1D, 2D convolutions
3. **Pooling**: max pooling, average pooling
4. **Normalization**: batch norm, layer norm
5. **Dropout**: regularization during training

### Phase 5: CUDA Support

**Priority: Lower than you'd think**

GPU support is important for performance, but:
- The CPU path needs to be complete and correct first
- CUDA development is significantly more complex (memory management across devices, kernel launches, synchronization)
- Having a correct CPU reference implementation makes CUDA debugging much easier

When CUDA support comes, the architecture is ready:
- `StorageCUDA` implementing `StorageAbstract`
- `MemoryPoolCUDA` with device memory management
- CUDA kernels registered with `DispatchKey::CUDA`
- Data transfer mechanisms between CPU and CUDA

### Phase 6: Quality of Life

**Priority: Ongoing**

1. **Broadcasting**: NumPy-style shape broadcasting for operations
2. **Serialization**: Save/load tensors and models
3. **Error handling**: Meaningful error messages for shape mismatches, type errors, etc.
4. **Comprehensive testing**: Unit tests for every operation, edge cases, memory leak testing
5. **Benchmarks**: Performance tracking to prevent regressions

---

## Build System Improvements

The CMakeLists.txt needs work:

- **Remove hardcoded paths**: Use `FetchContent` or `find_package` for Eigen and CTTI
- **Add install targets**: So users can `find_package(Forge)` in their own CMake projects
- **Add test targets**: Separate test executables using CTest or Google Test
- **CI/CD pipeline**: GitHub Actions for building and testing on Linux, macOS, and Windows
- **Compiler warnings**: Enable `-Wall -Wextra -Wpedantic` (or MSVC equivalents)

---

## Known Technical Debt

Things that work but should be improved:

1. **Thread safety**: The memory pool singleton is not thread-safe. For multi-threaded training, it needs either a lock or per-thread pools.

2. **Memory pool fragmentation**: The current binning strategy works well for common sizes but could waste memory for allocations that are just above a power-of-2 boundary.

3. **Reshape contiguity**: `reshape()` assumes the tensor is contiguous. Non-contiguous tensors (views) should either error clearly or automatically materialize (copy to contiguous memory) before reshaping.

4. **Gradient tensor lifetime**: `m_grads` is a raw pointer (`Tensor*`), which has ownership concerns. This should probably be a `std::unique_ptr<Tensor>` or stored inline with an "initialized" flag.

5. **4D limit**: The Eigen integration forces a 4D tensor maximum (padding smaller tensors with dimensions of 1). For some workloads, higher-dimensional tensors (5D for video, batched 3D convolutions) would be useful.

6. **No in-place operations**: All operations create new tensors. In-place variants (like `add_()` in PyTorch) would reduce memory pressure.

---

## Contribution Areas

If you're interested in contributing, here are areas where help would be most impactful:

### Easy Wins
- **Unit tests**: The test suite is minimal. Adding tests for edge cases (zero-size tensors, single-element tensors, large tensors) would be valuable.
- **CMake improvements**: Making the build system cross-platform without manual path editing.
- **Documentation examples**: More code examples showing how to use each feature.

### Medium Effort
- **Element-wise operations**: Addition, subtraction, multiplication, division — these follow a clear pattern and the infrastructure is ready.
- **Reduction operations**: Sum, mean, max along dimensions.
- **Error handling**: Adding proper validation and error messages throughout.

### Significant Effort
- **Matrix multiplication**: Getting matmul right with proper shape validation, broadcasting, and good performance.
- **CUDA backend**: Implementing StorageCUDA, MemoryPoolCUDA, and CUDA kernels.
- **Autograd gradient kernels**: Each forward operation needs a corresponding backward implementation.

---

## Design Principles Going Forward

As Forge grows, these principles should guide new additions:

1. **Everything goes through the dispatcher.** No operation should be hardcoded in the tensor class. If it's a computation, it's a kernel.

2. **CPU first, CUDA second.** Every operation gets a CPU implementation first. The CUDA version comes later, and the CPU version serves as the reference for correctness testing.

3. **Test at the kernel level.** Tests should exercise individual kernels, not just the tensor API. This isolates bugs and makes them easier to fix.

4. **Memory goes through the pool.** No raw `new` or `malloc` for tensor data. The memory pool exists for a reason — use it.

5. **Concepts over SFINAE.** Anywhere a template constraint is needed, use a C++20 concept. They're readable, they give good error messages, and they document the intent.

6. **Keep the tensor class thin.** The tensor is metadata + a pointer to storage. Logic belongs in kernels, not in the tensor.
