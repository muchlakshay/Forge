# Build System & Dependencies

This document covers how to build Forge, what external libraries it depends on, and the platform support story. Forge is header-only, so there's no library build step — but you still need to compile your code that uses it, and you need the dependencies in place.

---

## Build System: CMake

Forge uses CMake (minimum version 3.31) as its build system. The configuration is in `CMakeLists.txt` at the project root.

### Current Configuration

```cmake
cmake_minimum_required(VERSION 3.31)
project(Forge)
set(CMAKE_CXX_STANDARD 20)

add_executable(main tests/main.cpp)
target_include_directories(main PUBLIC
    <path-to-eigen>/eigen-3.4.0
    core/tensor
    <path-to-ctti>/ctti/include
    core/
)
```

The build currently compiles only the test file (`tests/main.cpp`) into an executable. Since Forge is header-only, the headers are pulled in through `#include` directives — there's no separate library target.

### Include Paths

The build needs four include paths:
1. **Eigen 3.4.0** — For the tensor storage backend and numerical operations
2. **`core/tensor`** — For `tensor.h` and the `Storage/` headers
3. **CTTI** — For compile-time type information used in the dispatcher
4. **`core/`** — For `enums.h` and the `autograd/` and `ops/` headers

### Building

Standard CMake workflow:

```bash
mkdir build && cd build
cmake ..
cmake --build .
```

This produces a `main` executable that runs the test suite.

> **Note**: The current CMakeLists.txt has hardcoded paths for Eigen and CTTI pointing to a specific Windows development machine. You'll need to update these paths to match your local setup. I plan to add proper `find_package` or `FetchContent` support to make this automatic, but for now it's manual.

---

## Dependencies

### Eigen 3.4.0

**What it is**: A C++ template library for linear algebra — matrices, vectors, numerical solvers, and related algorithms.

**Why I use it**: Forge uses Eigen's unsupported CXX11 Tensor module specifically. This provides the numerical tensor type that Forge maps its data to via `as_eigen<T>()`. Eigen handles the SIMD-optimized element-wise operations, reductions, and contractions.

**What Forge uses from Eigen**:
- `Eigen::Tensor<T, 4>` — 4D tensor type
- `Eigen::TensorMap` — View into raw memory as an Eigen tensor
- `Eigen::half` — 16-bit floating point type (used for `Dtype::float16`)

**Why not write my own numerics?**: Because Eigen is one of the most optimized linear algebra libraries available. It has years of SIMD tuning, cache-aware algorithms, and platform-specific optimizations. Writing my own would be a multi-year effort to reach the same performance. Forge's value is in the framework — dispatcher, autograd, memory management — not in reimplementing matrix multiplication.

**Version**: 3.4.0 specifically, because that's the version with the unsupported Tensor module API I'm targeting. Newer versions may work but haven't been tested.

### CTTI (Compile-Time Type Information)

**What it is**: A C++ library that provides type identification at compile time, without requiring RTTI (Runtime Type Information).

**Why I use it**: The dispatcher needs to verify that when you look up a kernel and cast it to a specific type, the types match. CTTI provides `ctti::type_id<T>()` which gives a unique identifier for any type at compile time.

**What Forge uses from CTTI**:
- `ctti::type_id<T>()` — Returns a compile-time type identifier
- `ctti::type_id_t` — The type ID type, used for storage and comparison

**Why not use RTTI/dynamic_cast?**: Two reasons:
1. **Performance**: RTTI-based `dynamic_cast` involves hash table lookups. CTTI type IDs are integer comparisons.
2. **Portability**: Some embedded and game development toolchains disable RTTI. CTTI works everywhere C++14 (or later) is supported.

---

## C++20 Requirements

Forge requires C++20. Here's what it uses from the standard:

### Concepts

Used extensively for compile-time type constraints:
- `TensorStorageType` — Ensures storage types are trivially copyable and standard layout
- `KernelDerived` — Ensures kernel types inherit from `Kernel`
- `ValidEnum` — Ensures operation enums have a `Count` member
- `isRefTo` — Checks reference-to-type relationships

Without concepts, these constraints would be enforced through SFINAE or `static_assert`, both of which give worse error messages.

### Fold Expressions

Used in reshape for compile-time size validation:
```cpp
static_assert((dims * ...) == m_size);  // Product of all dimensions must equal total size
```

### Structured Bindings

Used for cleaner destructuring in various places.

### `constexpr` Improvements

C++20 expanded what can be `constexpr`, which is used in several places for compile-time computation.

### std::bit_ceil / Bit Manipulation

The memory pool uses bit manipulation for power-of-2 rounding of allocation sizes.

---

## Platform Support

### Compilers

Forge is designed to work with:
- **MSVC** (Visual Studio 2019+) — Primary development compiler
- **GCC 10+** — Full C++20 support
- **Clang 12+** — Full C++20 support

The code has explicit platform detection for compiler-specific features:

```cpp
#if defined(_MSC_VER)
    // MSVC-specific code
#elif defined(__GNUC__) || defined(__clang__)
    // GCC/Clang-specific code
#endif
```

### Operating Systems

- **Windows** — Primary development platform, uses `_aligned_malloc` / `_aligned_free`
- **Linux** — Supported via `std::aligned_alloc` / `std::free`
- **macOS** — Should work with Clang, but not actively tested

### Memory Allocation

The memory pool uses platform-specific aligned allocation:

| Platform | Allocate | Deallocate |
|----------|----------|------------|
| Windows (MSVC) | `_aligned_malloc(size, alignment)` | `_aligned_free(ptr)` |
| Unix (GCC/Clang) | `std::aligned_alloc(alignment, size)` | `std::free(ptr)` |

Alignment is important for SIMD operations — SSE requires 16-byte alignment, AVX requires 32-byte alignment. The memory pool handles this automatically.

### Cache Prefetching

The memory pool uses compiler-specific prefetch intrinsics:

| Compiler | Prefetch Macro |
|----------|---------------|
| MSVC / Intel | `_mm_prefetch((const char*)(addr), _MM_HINT_T0)` |
| GCC / Clang | `__builtin_prefetch((addr), 0, 3)` |

---

## Project Structure for Building

```
Forge/
├── CMakeLists.txt          # Build configuration
├── core/                   # Header-only library (no compilation needed)
│   ├── enums.h
│   ├── autograd/
│   ├── ops/
│   └── tensor/
├── tests/
│   └── main.cpp            # Compiled to test executable
└── <external>/
    ├── eigen-3.4.0/        # Must be available on include path
    └── ctti/include/        # Must be available on include path
```

The external dependencies (Eigen and CTTI) need to be downloaded separately and their include paths configured in CMakeLists.txt. There's no package manager integration yet — it's manual setup.

### Setting Up Dependencies

1. **Eigen**: Download from https://eigen.tuxfamily.org/ (version 3.4.0). Extract somewhere and note the path.
2. **CTTI**: Clone from https://github.com/Manu343726/ctti. Note the `include/` path.
3. **Update CMakeLists.txt**: Replace the hardcoded paths with your local paths.

### Future Build Improvements

Things I plan to improve:
- **CMake `FetchContent`**: Automatically download Eigen and CTTI during the build
- **`find_package` support**: For system-installed dependencies
- **Install target**: So Forge can be installed and used as a `find_package`-able library
- **Multiple build targets**: Separate test, benchmark, and example executables
- **CI/CD**: Automated building and testing on multiple platforms

---

## Design Decisions Summary

| Decision | Rationale |
|----------|-----------|
| CMake as build system | Industry standard for C++, cross-platform |
| C++20 standard | Concepts, fold expressions, and other features dramatically improve code quality |
| Header-only library | Zero build complexity for users, enables aggressive inlining |
| Eigen for numerics | Battle-tested, SIMD-optimized, saves years of development |
| CTTI for type IDs | Faster than RTTI, works without runtime type info enabled |
| Platform-specific allocation | Proper alignment for SIMD on both Windows and Unix |
