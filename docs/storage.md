# Storage & Memory Pool Subsystem

This document covers how Forge manages memory — from the abstract storage interface down to the radix-tree based memory pool that handles actual allocations. Memory management is one of the most critical parts of a tensor framework, and I spent considerable time getting this right.

The code lives in `core/tensor/Storage/`:
- `storage_abstract.h` — The interface that all storage backends implement
- `storage_cpu.h` — The CPU storage implementation
- `memory_pool_abstract.h` — The interface for memory pool allocators
- `memory_pool_cpu.h` — The radix-tree based CPU memory pool

---

## Storage Abstraction

### Why an Abstraction?

The `StorageAbstract` class is the base interface for all storage backends:

```cpp
class StorageAbstract {
public:
    virtual void* data() = 0;
    virtual std::size_t nbytes() = 0;
    virtual ~StorageAbstract() = default;
};
```

It's intentionally minimal — just `data()` and `nbytes()`. The tensor doesn't need to know whether its data lives on the CPU, GPU, or some future accelerator. It just needs a pointer to bytes and a size.

This abstraction is what allows the same `Tensor` class to work with different backends. When CUDA support is added, there'll be a `StorageCUDA` class implementing this same interface, and the tensor won't need any changes.

I kept the interface this small because:
1. Every virtual call has overhead. Minimizing the interface means fewer virtual dispatch points.
2. Backend-specific methods (like typed access) are handled through downcasting with CTTI verification in the dispatcher, not through the abstract interface.
3. Storage is a low-level building block — it should be simple and fast.

---

## CPU Storage Implementation

`StorageCPU` is the concrete implementation for CPU tensors. It's a template class parameterized on the element type:

```cpp
template <TensorStorageType T>
class StorageCPU final : public StorageAbstract {
    T* m_data;
    std::size_t m_nbytes;
    bool m_is_view;
    // ...
};
```

### The TensorStorageType Concept

The template parameter is constrained by a C++20 concept:

```cpp
template <typename T>
concept TensorStorageType = std::is_trivially_copyable_v<T> && std::is_standard_layout_v<T>;
```

This ensures you can only create storage for types that:
- Can be safely copied with `memcpy` (trivially copyable)
- Have a predictable, C-compatible memory layout (standard layout)

This rules out types like `std::string` or `std::vector` — you can't store those in a tensor, and for good reason. Tensor data needs to be a flat array of uniform-sized elements for SIMD operations, GPU transfers, and memory pooling to work.

### Owned vs. View Mode

`StorageCPU` has two modes, controlled by the `m_is_view` flag:

**Owned mode** (default): The storage allocates memory from the memory pool and owns it. When the storage is destroyed, the memory is returned to the pool.

**View mode**: The storage points to memory owned by someone else. When destroyed, it does *not* free the memory.

This distinction is what makes tensor views zero-cost. When you do `tensor[i]`, a new `StorageCPU` is created in view mode, pointing into the parent's memory. The parent's storage handles deallocation.

```cpp
~StorageCPU() {
    if (!m_is_view && m_data) {
        MemoryPoolCPU::instance().free(m_data);
    }
}
```

The destructor only frees memory if the storage actually owns it. Simple but important.

### Key Methods

- **`data()`**: Returns `void*` to the raw bytes (virtual interface)
- **`data_()`**: Returns `T*` — typed access for internal use
- **`nbytes()`**: Returns total byte count
- **`size()`**: Returns element count (`nbytes / sizeof(T)`)
- **`clone()`**: Creates a new owned storage with a deep copy of the data

`clone()` is used when you explicitly want a copy (like `tensor.clone()`). It allocates fresh memory from the pool and copies the data, creating a fully independent storage.

---

## The CPU Memory Pool

This is where things get interesting. The memory pool in `memory_pool_cpu.h` is a serious piece of infrastructure — about 225 lines of carefully designed allocation code.

### Why a Custom Allocator?

In neural network training, you're constantly allocating and freeing tensors — forward activations, backward gradients, optimizer states, temporary buffers. If every allocation goes to `malloc`/`free`, you're paying:
1. **System call overhead**: `malloc` may need to request memory from the OS
2. **Fragmentation**: Frequent alloc/free cycles fragment the heap
3. **No size-class optimization**: `malloc` doesn't know that most of your allocations are powers of 2

A memory pool solves all three:
1. Memory is pre-allocated in large chunks, so individual allocations are just pointer bumps
2. Size-classed bins eliminate fragmentation within each class
3. Freed blocks are reused immediately, without going back to the OS

### Architecture

The memory pool uses three key structures:

#### 1. Size-Classed Bins

There are 64 bins, each handling a specific allocation size (power-of-2 aligned). When you request memory:
1. The size is rounded up to the next power of 2
2. A bin index is computed from the size
3. The appropriate bin provides a free block

If a bin is empty, a new chunk of memory is allocated from the OS (64 KB default) and divided into blocks of the bin's size.

```
Bin 0:  [block][block][block][block]...  (smallest size class)
Bin 1:  [block][block][block]...         (2× smallest)
Bin 2:  [block][block]...                (4× smallest)
...
Bin 63: [block]                          (largest size class)
```

Each bin maintains a linked list of free blocks. Allocation pops from the front; deallocation pushes to the front. Both are O(1).

#### 2. Radix Tree for Deallocation

Here's the clever part. When you free a pointer, you need to figure out which bin it belongs to. Most allocators solve this by storing metadata adjacent to the allocation (like `malloc` does — there's a header before every block). But that wastes memory and messes with alignment.

Instead, I use a 3-level radix tree that maps pointers to bins:

```
Address bits:  [  Level 0 (9 bits)  |  Level 1 (9 bits)  |  Level 2 (9 bits)  ]
                       │                     │                     │
                       ▼                     ▼                     ▼
               ┌───────────┐         ┌───────────┐         ┌───────────┐
               │ 512 slots │──►      │ 512 slots │──►      │ 512 slots │──► bin index
               └───────────┘         └───────────┘         └───────────┘
```

When a chunk is allocated for a bin, the chunk's address is recorded in the radix tree. When `free()` is called, the pointer is looked up in the tree to find which bin to return it to. The lookup is O(1) — just bit-shifting and array indexing, no hash tables or tree traversals.

Each level of the tree has 512 entries (2⁹). Three levels gives 27 bits of addressable space, which is enough for the chunk sizes we're working with. The tree nodes are allocated lazily — most of the tree stays null for typical workloads.

#### 3. Platform-Aware Aligned Allocation

The pool uses aligned allocation under the hood:

```cpp
#ifdef _MSC_VER
    void* ptr = _aligned_malloc(size, alignment);
    _aligned_free(ptr);
#else
    void* ptr = std::aligned_alloc(alignment, size);
    std::free(ptr);
#endif
```

Windows and Unix have different APIs for aligned allocation, so the pool handles both. The alignment matters for SIMD operations — unaligned memory access is slower (or causes faults on some architectures).

### Cache Prefetching

The pool uses compiler-specific prefetch hints to improve cache behavior:

```cpp
#if defined(_MSC_VER)
    #define FORGE_PREFETCH(addr) _mm_prefetch((const char*)(addr), _MM_HINT_T0)
#elif defined(__GNUC__) || defined(__clang__)
    #define FORGE_PREFETCH(addr) __builtin_prefetch((addr), 0, 3)
#endif
```

When traversing the free list during allocation, prefetching the next block while processing the current one hides memory latency. This is a micro-optimization but it matters when you're doing millions of allocations per second during training.

### Singleton Pattern

The memory pool is a singleton:

```cpp
static MemoryPoolCPU& instance() {
    static MemoryPoolCPU pool;
    return pool;
}
```

There's one pool per process. I chose a singleton because:
- Memory reuse is most effective with a single pool (freed blocks from one part of the code are immediately available to another)
- Multiple pools would fragment the available free blocks
- The pool's state (radix tree, bins) is expensive to maintain, so having one instance is efficient

The downside is that it's not thread-safe as currently implemented — that's a future improvement.

### Hard Clear

```cpp
void hard_clear_cache() {
    // Deallocates ALL memory held by the pool
}
```

This is a nuclear option — it frees every chunk the pool has allocated. Useful for testing and for controlled teardown, but you'd never call this during normal operation.

---

## Design Decisions Summary

| Decision | Rationale |
|----------|-----------|
| Minimal `StorageAbstract` interface | Fast virtual dispatch, clean abstraction boundary |
| `TensorStorageType` concept constraint | Prevents storing non-POD types that would break SIMD and GPU transfers |
| Owned vs. view mode in `StorageCPU` | Zero-cost tensor views without ownership confusion |
| Radix-tree for pointer-to-bin lookup | O(1) deallocation without metadata overhead |
| Power-of-2 bin sizes | Simplifies size-class math, reduces fragmentation |
| Platform-specific aligned allocation | Windows/Unix compatibility with proper alignment for SIMD |
| Singleton memory pool | Maximizes memory reuse across the entire application |
| Cache prefetching macros | Micro-optimization for high-throughput allocation patterns |

---

## File Reference

| File | Lines | Purpose |
|------|-------|---------|
| `storage_abstract.h` | ~11 | Pure virtual storage interface |
| `storage_cpu.h` | ~67 | CPU storage with owned/view modes |
| `memory_pool_abstract.h` | ~10 | Memory pool interface |
| `memory_pool_cpu.h` | ~225 | Radix-tree based CPU memory pool |
