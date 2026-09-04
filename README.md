# Forge: A Deep Learning Framework from Scratch

## Overview

Forge is a deep learning framework that I built from the ground up to really understand how modern AI and frameworks work
behind the scenes. it gives you the tools you need to build, train and fine-tune.

i didn’t just want to use pytorch and trust what happens inside it. i wanted to see the internal details of how it actually works.

to be sure everything was right i ran a check. when i used pretrained weights and greedy decoding Forge’s GPT-2 model produced exactly the same tokens, as Hugging Face’s `transformers`. same outputs. same results. that gave me confidence that the whole system was working as intended.

# Documentation

these documentations are still basic, further work on them will be done to make them as detailed as possible

- [Tensor](docs/tensor.md)
- [Linear](docs/linear.md)
- [Activations](docs/activations.md)
- [Loss Functions](docs/loss_functions.md)
- [Optimizers](docs/optimizers.md)
- [Embeddings](docs/embeddings.md)
- [Self-Attention](docs/Self-Attention.md)
- [LayerNorm](docs/LayerNorm.md)
- [Tokenizer](docs/tokenizer.md)
- [Loading/Saving](docs/loading_saving.md)

## Installation

### Requirements

CMake 3.50+

a C++20 compiler (windows: MinGW-w64) (linux: GCC or Clang)

-OpenBLAS

-A CPU with AVX2 support (any x86-64 CPU from roughly the last 10 years)

reflect-cpp, ctti and Eigen are downloaded automatically using CMake `FetchContent`
no setup for them is needed

### windows
1. install OpenBLAS or grab a prebuilt binary if u dont have one from
   [OpenBLAS releases page](https://github.com/OpenMathLib/OpenBLAS/releases) and extract it, like in `C:/Libs/OpenBLAS`


2. clone the repo:
```bash
   git clone https://github.com/muchlakshay/Forge
   cd Forge
```

3. configure and build:
```bash
   cmake -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release -DOPENBLAS_ROOT="C:/Libs/OpenBLAS"
   cmake --build build
```

in case OpenBLAS is not installed in `C:/Libs/OpenBLAS`, set `-DOPENBLAS_ROOT` to the path
wherever you have extracted it.

4. `Forge` builds as a static library at `build/libForge.a`.

### Linux

1. install OpenBLAS and compiler toolchain:
```bash
   sudo apt install build-essential cmake libopenblas-dev
```

2. Clone the repo:
```bash
   git clone https://github.com/muchlakshay/Forge
   cd Forge
```

3. Configure and build:
```bash
   cmake -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build
```
in linux the `OPENBLAS_ROOT` defaults to `/usr`, which is where `apt` installs it so
no extra flag needed unless you built OpenBLAS from somewhere else.

4. `Forge` builds as a static library at `build/libForge.a`.

### Building the test executables

Forge has two test executables - `gpt2` and 'mnsit' that shows the framework in action
and they are not built by default.


so a plain build only produces the Forge library. to build the tests too:

```bash
cmake -B build -DFORGE_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

it adds the test executables in `build/`. prebuilt version of these test executables are
for both linux and windows is available on the [releases page](https://github.com/muchlakshay/Forge/releases/tag/0.1)


### Linking Against Forge And OpenBLAS Runtime

Forge links against OpenBLAS as a shared library (will be changed in future to a link against the static lib),
so the OpenBLAS runtime needs to be available when running an executable that uses Forge.

on windows, copy `libopenblas.dll` from `OPENBLAS_ROOT/bin` into the same directory as your executable.
otherwise windows will fail to start the program because the dll cant be found.

on linux, you can install OpenBLAS system wide with `sudo apt install libopenblas0`, or make sure `libopenblas.so` or `libopenblas.so.0`
is available through your `LD_LIBRARY_PATH`

the repository's `tests/CMakeLists.txt` already handles the windows dll copy for the `gpt2` and `mnist` exes.
if you're building your own executable and linking it against Forge, you'll need to handle the runtime library yourself !!!


### Tested Platforms

| Platform | Arch  | Tested OS Version              | Most Thoroughly Tested |
|----------|-------|----------------------------------|:-----------------------:|
| Windows  | x64   | Windows 11 Pro version 25H2         | yes                      |
| Linux    | x64   | Ubuntu 26.04 (via WSL2)            | yes                      |

## Architecture & Components

### Math Backend & Performance

Forge currently uses **Eigen** for its tensor and linear algebra operations.
it works but Eigens expression templates and lazy evaluation can become a bottleneck,
especially during the reduction ops, in places like cross entropy loss, it backward pass and same with the softmax and it backward pass kernel.

the plan is to move the linear algebra work to OpenBLAS and replace some of the element-wise operations with
custom AVX2 vectorized kernels as OpenBLAS lacks elementwise kernels.

### Tensor

Forge has its Tensor` abstraction, supporting tensors of up to 4 dimensions. it is built around the needs of deep learning rather than trying to be a general-purpose tensor library.
as generally only at max 4 dim tensors are used in deep learning.

### Management of Memory

Forge uses a CPU radix tree based memory allocator designed around the frequent allocation and deallocation patterns
that happen during model training.

### Supported Operations & DL Primitives

Forge currently includes the building blocks needed for common neural networks and transformers.
this includes Linear, LayerNorm, Embeddings,  sinusoidal positional encoding, multi-head self-attention (with optional masking)
and a gpt2 style BPE tokenizer.

for activations Forge supports RELU, GELU, Sigmoid, Tanh, Softmax and LeakyRELU.

the loss functions currently supported are cross entropy loss, binary cross entropy and mean squared error (MSE).
cross entropy and binary cross entropy include fused softmax and fused sigmoid respectively.

Forge also has SGD, SGD with Momentum, Adam and AdamW optimizers

### Backends

currently Forge runs on the CPU with Eigen handling most of the math operations.

A CUDA backend is planned for GPU acceleration along with optimized OpenBLAS and custom vectorized kernels for the CPU backend.

### Use Cases

Forge is mainly built as a learning project. its meant for understanding how deep learning frameworks work internally
experimenting with architectures and building neural networks from the lower level pieces.
