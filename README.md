# Forge: A Deep Learning Framework from Scratch

- [Overview](#overview)
- [Installation](#installation)
- [Tensor](#tensor-documentation)
  - [Internal Components](#internal-components)
  - [Basic Creation](#basic-creation)
  - [Creating Tensors with Specific Values](#creating-tensors-with-specific-values)
  - [Initializing from Existing Data](#initializing-from-existing-data)
  - [Reshaping](#reshaping)
  - [Cloning and Copying](#cloning-and-copying)
  - [Indexing](#indexing)
  - [Eigen support](#direct-element-access-with-eigen)
  - [Tensor Properties and Inspection](#tensor-properties-and-inspection)
  - [Backpropagation / auto-differentitation](#setting-up-tensors-for-backpropagation)
  - [Advance Gradient Management](#advance-gradient-management)
- [Linear Layer](#linear-layer-documentation)
  - [Overview](#linear-class-overview)
  - [Internal Components](#internal-components)
  - [Basic Constructor](#basic-constructor)
  - [Weight Initialization](#weight-initialization-strategies)
  - [Forward Pass](#forward-pass)
  - [Accessing Parameters](#accessing-weights-and-bias)
  - [Layer Properties](#layer-properties)
- [Activations](#activations)
  - [RELU](#forgerelu)
  - [Sigmoid](#forgesigmoid)
  - [Tanh](#forgetanh)
  - [LeakyRELU](#forgeleakyrelu)
  - [GELU](#forgegelu)
  - [Softmax](#forgesoftmax)
- [Loss functions](#loss-functions-documentation)
  - [Mean Squared Error](#mean-squared-error-mse)
  - [Cross-Entropy Loss](#cross-entropy-loss)
  - [Binary Cross-Entropy Loss](#binary-cross-entropy-bce)
- [Optimizers](#optimizers-documentation)
  - [Stochastic Gradient Descent (SGD)](#forgesgd)
  - [Adaptive Moment Estimation (Adam)](#forgeadam)
- [LayerNorm](#layernorm)
- [Self-Attention](#self-attention)
- [Embeddings](#embeddings)
- [Tokenizer](#tokenizer)
- [Parameters extraction & Model Load/Save (safetensors)](#reflection-based-parameters-&-model-load-save-safetensors)

## Overview

Forge implements core deep learning abstractions and primitives, enabling you to build, train, and optimize neural network models efficiently. The framework is designed for simplicity, making it suitable for learning use cases.

Most people learn deep learning by calling `model.fit()` and trusting PyTorch got the internals right. I wanted to know *why* it's right - how a tensor actually sits in memory, why GEMM dominates a forward pass, why attention needs a causal mask and what breaks silently if you get a transpose wrong.

So Forge is a deep learning framework built from scratch. Every primitive here - Linear, LayerNorm, SelfAttention, Optimizers, and some of the the AVX2 kernels underneath - exists because I wrote it myself and verified it against a real reference.

The proof: Forge's GPT-2, loaded with real pretrained weights, matches Hugging Face's `transformers` **token-for-token** under greedy decoding. Not close - exact. That's what convinces me this actually taught me how modern AI works, not just how to produce something that looks like it does.

## Installation

### Requirements

- CMake 3.20+
- A C++20 compiler
  - Windows: MinGW-w64 (tested with the toolchain bundled in CLion)
  - Linux: GCC or Clang
- OpenBLAS
- A CPU with AVX2 support (any x86-64 CPU from roughly the last 10 years)

Eigen, reflect-cpp and ctti are fetched automatically via CMake's `FetchContent` -
no manual setup needed for either.

### Windows

1. Install OpenBLAS. If you don't already have it, grab a prebuilt release
   from the [OpenBLAS releases page](https://github.com/OpenMathLib/OpenBLAS/releases)
   and extract it somewhere, e.g. `C:/Libs/OpenBLAS`.

2. Clone the repo:
```bash
   git clone https://github.com/muchlakshay/Forge
   cd Forge
```

3. Configure and build:
```bash
   cmake -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release -DOPENBLAS_ROOT="C:/Libs/OpenBLAS"
   cmake --build build
```
   If OpenBLAS is somewhere other than `C:/Libs/OpenBLAS`, point `-DOPENBLAS_ROOT`
   at wherever you extracted it.

4. `Forge` builds as a static library at `build/libForge.a`, with headers
   under `core/` and `primitives/` in the repo itself.

### Linux

1. Install OpenBLAS and a compiler toolchain:
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
   `OPENBLAS_ROOT` defaults to `/usr`, which matches where `apt` installs it -
   no extra flag needed unless you built OpenBLAS from source somewhere custom.

4. `Forge` builds as a static library at `build/libForge.a`.

### Building the test executables

Forge includes `gpt2` and `mnist` test executables that demonstrate the
framework in action. They're off by default
so a plain build only produces the library. To build them too:

```bash
cmake -B build -DFORGE_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

This adds `gpt2`/`mnist` (Linux) or `gpt2.exe`/`mnist.exe` (Windows) to
`build/`. Prebuilt version of these for windows are also available on the
[Releases page](https://github.com/muchlakshay/Forge/releases/tag/0.1) if you'd rather skip building them yourself.

### Linking against Forge

Since `Forge` links OpenBLAS as a shared library rather than statically, any
executable you link against `Forge` needs the OpenBLAS runtime library
available at runtime, not just at link time:

- **Windows:** copy `libopenblas.dll` (found under `OPENBLAS_ROOT/bin`) into
  the same folder as your built executable. Without it, the executable will
  fail to launch with a missing-DLL error.
- **Linux:** either install OpenBLAS system-wide (`sudo apt install
  libopenblas0`), or ensure `libopenblas.so`/`libopenblas.so.0` is somewhere
  on your `LD_LIBRARY_PATH`.

The `tests/CMakeLists.txt` in this repo already handles this automatically
for the `gpt2`/`mnist` executables via a post-build copy step on Windows -
if you're linking your own executable against `Forge` outside that setup,
you'll need to do this step yourself.

## Architecture & Components

### Math Backend & Performance Roadmap
Currently, Forge uses **Eigen** as its underlying math library for tensor operations. However, **Eigen's abstraction and lazy evaluation (expression templates) introduce significant performance bottlenecks**, especially during backpropagation. 

**Performance Optimization Plan:**
- Replace Eigen with **OpenBLAS** for optimized linear algebra operations
- Implement **custom vectorized element-wise kernels** for backward pass computations
- These optimizations will dramatically improve training speed and throughput

### Tensor Abstraction
- **Multi-dimensional tensors** supporting up to **4 dimensions** for handling batch data, spatial dimensions, and feature channels
- Optimized for deep learning use cases (not a general-purpose tensor library)
- Core abstraction layer for all neural network computations

### Memory Management
- **CPU radix tree-based memory allocator** for efficient memory allocation and deallocation
- Optimized for the allocation patterns typical in deep learning workflows

## Supported Operations & Layers

### Deep Learning Primitives
- **Linear Layer**: Fully connected layer with learnable weights and biases
- **Layer Normalization**: Normalization across features
- **Embedding Layer**: Token/index embedding lookups
- **Sinusoidal Encoding**: Positional encodings for sequence models
- **Multi-Head Self Attention**: Core component for transformer-based architectures
- BPE Tokenizer

### Activation Functions
- ReLU
- GELU
- Sigmoid
- Tanh
- Softmax

### Loss Functions
- Cross-Entropy Loss, with fused softmax (for multi-class classification)
- Binary Cross-Entropy Loss, with fused softmax)
- Mean Squared Error (MSE)

### Optimizers
- **SGD**: Standard stochastic gradient descent
- **SGD with Momentum**: Accelerated gradient-based optimization
- **Adam**: Adaptive learning rate optimizer
- **AdamW**: Adam with weight decay regularization

## Backend Support

### Current
- **CPU Backend** (with Eigen-based math, performance optimizations in progress)

### Planned
- **CUDA Backend** (GPU acceleration for NVIDIA devices)
- **Optimized kernels** (OpenBLAS + custom vectorized kernels)

## Use Cases
Forge is ideal for:
- Learning deep learning systems implementation from first principles
- Experimenting with neural network architectures

# Tensor Documentation

## Tensor Class Overview

The **Tensor** class is the core abstraction in Forge for multi-dimensional data. It encapsulates:
- **Data storage** via CPU memory allocator (radix tree-based)
- **Shape and stride information** for multi-dimensional indexing
- **Automatic differentiation** support for backpropagation
- **Data type (dtype)** and **device** information

### Internal Components

```cpp
class Tensor {
    Device m_device;                          // CPU or GPU
    Dtype m_dtype;                            // float32, float64, int32, int16, etc.
    std::shared_ptr<StorageAbstract> m_storage; // Actual data buffer
    std::vector<std::size_t> m_shape;         // Dimensions: (batch, height, width, channels)
    std::vector<std::size_t> m_strides;       // Row-major strides for efficient indexing
    std::size_t m_size;                       // Total element count
    bool m_need_grads;                        // Requires gradients during backward?
    DispatchKey m_dispatch_key;               // CPU/CUDA dispatch key
    mutable std::shared_ptr<NodeAbstract> m_node;   // Autograd computation graph node
    std::shared_ptr<Tensor> m_grads;          // Accumulated gradients tensor
};
```


## Creating and Initializing Tensors

### Basic Creation

```cpp
#include "Forge.h"
using namespace Forge;

// Create empty tensor
Tensor t;

// Create tensor with shape (batch=32, seq_len=128, features=64)
std::vector<std::size_t> shape = {32, 128, 64};
Tensor embeddings(shape, Dtype::float32, true, Device::CPU);
```

### Creating Tensors with Specific Values

```cpp
// Create batch of 10 samples with 5 features, initialized to zero
Tensor zeros = Tensor::Zeros({10, 5}, true);

// Create ones matrix for bias initialization
Tensor ones = Tensor::Ones({1, 64}, true);

// Create constant tensor with value 0.5
Tensor half_tensor = Tensor::Constant({8, 8}, 0.5f, true);

// Create range tensor: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Tensor indices = Tensor::Range(0, 10, 1, true);
```

### Initializing from Existing Data

```cpp
// Convert numpy/C array to Forge tensor
float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
Tensor from_data = Tensor::FromHostPtr(data, {2, 3}, true);

// Direct initializer list initialization
Tensor matrix({2, 3});
matrix = {{1.0f, 2.0f, 3.0f}, 
          {4.0f, 5.0f, 6.0f}};
```

## Tensor Manipulation

### Reshaping

```cpp
// Create flat tensor
Tensor flat = Tensor::Range(0, 24, 1, true);  // [0, 1, 2, ..., 23]

// Reshape to (2, 3, 4) - batch=2, height=3, width=4
Tensor reshaped = flat.reshape(2, 3, 4);
// Elements remain the same, just different view

// Another reshape: (2, 3, 4) -> (6, 4)
Tensor flattened = reshaped.reshape(6, 4);
```

### Cloning and Copying

```cpp
Tensor original = Tensor::Ones({3, 3}, true);

// Deep clone - independent copy
Tensor independent_copy = original.clone();

// Copy data from one tensor to another
Tensor destination({3, 3});
destination.copy(original);
```

### Indexing

```cpp
Tensor batch = Tensor::Zeros({32, 128, 64});  // batch_size=32

// Get first sample from batch
Tensor first_sample = batch[0];  // Shape: (128, 64)

// Get tenth sample
Tensor tenth_sample = batch[9];  // Shape: (128, 64)
```

## Direct Element Access with Eigen

Forge integrates with Eigen for efficient element-wise operations:

```cpp
// Create a weight matrix for linear layer
Tensor weights({64, 32}, true);  // 64 input features, 32 output features
weights = 0.1f;

// Get Eigen map for direct element access
auto w_map = weights.as_eigen<float, 2>();

// Set specific elements
w_map(0, 0) = 0.5f;   // weights[0][0]
w_map(10, 5) = -0.3f; // weights[10][5]

// Element-wise operations via Eigen
w_map = w_map * 2.0f;  // Scale all weights by 2
w_map = w_map + 0.01f; // Add small constant (regularization)
```

## Tensor Properties and Inspection

```cpp
Tensor t = Tensor::Range(0, 60, 1);
t = t.reshape(3, 4, 5);

// Get shape information
const auto& shape = t.shape();           // {3, 4, 5}
std::cout << "Shape: [" << shape[0] << ", " << shape[1] << ", " << shape[2] << "]\n";

// Get memory layout
const auto& strides = t.strides();       // {20, 5, 1} for row-major layout

// Data type and device
Dtype dtype = t.dtype();                 // Dtype::float32
Device device = t.device();              // Device::CPU

// Memory information
std::size_t total_elements = t.size();   // 60
void* data_ptr = t.data();               // Raw pointer to buffer

// Gradient requirement
bool requires_grad = t.need_grads();     // true or false
```

## Automatic Differentiation

### Setting up Tensors for Backpropagation

```cpp
// Input features
Tensor x {Tensor::Ones({32, 10}, true)};  // 32 samples, 10 features
Linear linear {10, 32};

// Forward pass (simplified)
Tensor output = linear(x);  

//trigger a backward pass
output.backward()

//access gradients of x
std::cout<<x.gradients();

```


### Advance Gradient Management

```cpp
Tensor output = Linear(x);

// Backward with graph retention (for multiple backward passes)
output.backward(true);  // keep_graph=true

// Use gradients...
auto grad1 = x.gradients().clone();

// Clear and do another backward
x.clear_grads();
output.backward(true);  // Second backward pass

// Now gradients accumulated again
```

---

# Linear Layer Documentation

## Linear Class Overview

The **Linear** layer is a fully connected (dense) layer that applies an affine transformation to the input. It's one of the fundamental building blocks in neural networks, implementing the operation:

```
output = input @ weights^T + bias
```

Where:
- **input** has shape `(batch_size, ..., input_features)` (supports up to 4D tensors)
- **weights** has shape `(input_features, output_features)`
- **bias** has shape `(1, 1, 1, output_features)` (broadcasted)
- **output** has shape `(batch_size, ..., output_features)`

### Internal Components

```cpp
class Linear {
    Tensor m_weights;              // Shape: (input_size, output_size)
    Tensor m_bias;                 // Shape: (1, 1, 1, output_size)
    DispatchKey m_dispatch_key;    // CPU or GPU dispatch key
    std::size_t m_input_size;      // Number of input features
    std::size_t m_output_size;     // Number of output features
    bool m_using_bias;             // Use bias or not
    Dtype m_dtype;                 // Data type (float32, float64, etc.)
    Device m_device;               // Device (CPU or GPU)
};
```

## Creating a Linear Layer

### Basic Constructor

```cpp
#include "Forge.h"
using namespace Forge;

// Create a linear layer: 128 input features -> 64 output features
// Uses Xavier Normal initialization by default
Linear linear_1(128, 64);

// With custom data type
Linear linear_2(256, 128, Initializers::he_normal, Dtype::float32);

// Without bias term
Linear linear_no_bias(100, 50, Initializers::xavier_normal, Dtype::float32, Device::CPU, false);
```

### Constructor Parameters

```cpp
Linear(
    std::size_t input_size,                    // Number of input features
    std::size_t output_size,                   // Number of output features
    Initializers initializer = Initializers::xavier_normal,  // Weight initialization
    Dtype dtype = Dtype::float32,              // Data type
    Device device = Device::CPU,               // Device (CPU/GPU)
    bool bias = true                           // Use bias term?
);
```

### Weight Initialization Strategies

```cpp
// Xavier Normal: Good for networks with sigmoid/tanh activations
Linear layer_xavier(128, 64, Initializers::xavier_normal);

// He Normal: Recommended for ReLU-based networks
Linear layer_he(128, 64, Initializers::he_normal);

// Xavier Uniform: Uniform distribution variant
Linear layer_xu(128, 64, Initializers::xavier_uniform);

// He Uniform: He initialization with uniform distribution
Linear layer_hu(128, 64, Initializers::he_uniform);
```

## Using Linear Layers

### Forward Pass

```cpp
// Create layer and input
Linear fc(784, 128);
Tensor input({32, 784});  // batch_size=32, features=784

// Forward pass using operator()
Tensor output = fc(input);  // Returns shape: (32, 128)

std::cout << "Input shape: ";
for (auto s : input.shape()) std::cout << s << " ";
std::cout << "\nOutput shape: ";
for (auto s : output.shape()) std::cout << s << " ";
std::cout << "\n";
```

### Accessing Weights and Bias

```cpp
Linear layer(64, 32);

// Get read-only references
const auto& weights = layer.weights();     // Shape: (64, 32)
const auto& bias = layer.bias();           // Shape: (1, 1, 1, 32)

// Get mutable references for optimization
auto& weights_mut = layer.weights();
auto& bias_mut = layer.bias();

// Inspect dimensions
std::size_t in_size = layer.input_size();   // 64
std::size_t out_size = layer.output_size(); // 32

// Print shapes
std::cout << "Weights shape: " << weights.shape()[0] << " x " << weights.shape()[1] << "\n";
std::cout << "Bias shape: " << bias.size() << " elements\n";
```

## Layer Properties

```cpp
Linear layer(512, 256);

// Query layer metadata
std::cout << "Layer configuration:\n";
std::cout << "  Input size: " << layer.input_size() << "\n";
std::cout << "  Output size: " << layer.output_size() << "\n";
std::cout << "  Data type: " << static_cast<int>(layer.dtype()) << "\n";
std::cout << "  Device: " << static_cast<int>(layer.device()) << "\n";

// Weights storage
const auto& w = layer.weights();
std::cout << "  Weights total elements: " << w.size() << "\n";
std::cout << "  Bias total elements: " << layer.bias().size() << "\n";
```
---
# Activations

This module provides the activation functions used inside models. Each one is a small stateless functor - construct it (or use a temporary) and call it like a function on a `Tensor`. Gradients are wired up automatically whenever the input requires them, so you never call a grads function directly; it runs as part of the normal `.backward()` pass.


## `Forge::Relu`

```cpp
Tensor operator()(const Tensor& input) const;
```

Returns a new tensor of the same shape as `input`, with ReLU applied element-wise.

### Formula

```
Relu(x) = max(x, 0)
```

### Example

```cpp
Forge::Relu relu;

Tensor x = ...;     // any shape
Tensor y = relu(x); // same shape as x
y.backward();
```

## `Forge::Sigmoid`

```cpp
Tensor operator()(const Tensor& input) const;
```

Returns a new tensor of the same shape as `input`, with the sigmoid (logistic) function applied element-wise.

### Formula

```
Sigmoid(x) = 1 / (1 + exp(-x))
```

### Example

```cpp
Forge::Sigmoid sigmoid;

Tensor x = ...;        // any shape
Tensor y = sigmoid(x); // same shape as x, values in (0, 1)
y.backward();
```

## `Forge::Tanh`

```cpp
Tensor operator()(const Tensor& input) const;
```

Returns a new tensor of the same shape as `input`, with the hyperbolic tangent applied element-wise.

### Formula

```
Tanh(x) = tanh(x)
```

### Example

```cpp
Forge::Tanh tanh_;

Tensor x = ...;      // any shape
Tensor y = tanh_(x); // same shape as x, values in (-1, 1)
y.backward();
```

## `Forge::LeakyRelu`

```cpp
Tensor operator()(const Tensor& input) const;
```

Returns a new tensor of the same shape as `input`, with Leaky ReLU applied element-wise.

### Formula

```
LeakyRelu(x) = x        if x > 0
             = a * x    otherwise        # a = 0.01, fixed internally
```

### Example

```cpp
Forge::LeakyRelu leaky_relu;

Tensor x = ...;            // any shape
Tensor y = leaky_relu(x);  // same shape as x
y.backward();
```

### Note

The negative slope `a` is hardcoded at `0.01` - there's currently no constructor parameter to change it.


## `Forge::Gelu`

```cpp
Tensor operator()(const Tensor& input) const;
```

Returns a new tensor of the same shape as `input`, with GELU applied element-wise, using the tanh-based approximation (the same one used in GPT-style implementations).

### Formula

```
Gelu(x) = 0.5 * x * (1 + tanh(k * (x + c * x^3)))

  k = sqrt(2 / pi) = 0.7978845608028654
  c = 0.044715
```

### Example

```cpp
Forge::Gelu gelu;

Tensor x = ...;     // any shape
Tensor y = gelu(x); // same shape as x
y.backward();
```

### Note

This is the tanh approximation, not the exact erf-based formula (`0.5 * x * (1 + erf(x / sqrt(2)))`). Results are very close but not bit-identical to an exact GELU implementation.

## `Forge::Softmax`

```cpp
Tensor operator()(const Tensor& input) const;
```

Computes softmax along the **last axis** of a **4D** tensor, returning a tensor of the same shape. This is the same shape convention used by attention scores, e.g. `(batch, heads, seq_len, seq_len)`.

### Formula

For each row `x` along the last axis:

```
Softmax(x)_i = exp(x_i - max(x)) / sum_j(exp(x_j - max(x)))
```

The `- max(x)` subtraction is purely for numerical stability and doesn't change the result.

### Example

```cpp
Forge::Softmax softmax;

Tensor scores = ...;            // shape (batch, heads, seq_len, seq_len)
Tensor probs = softmax(scores); // same shape, normalized over the last axis
probs.backward();
```

### Note

`Softmax` specifically reduces over axis index `3` (the last of four dims) - passing a 2D or 3D tensor won't normalize correctly. If you need softmax over a different rank/axis, reshape first.

---

## General notes

- **All six are stateless** - there's nothing to configure or store, so `Forge::Relu{}(x)` works just as well as keeping a named instance around.
- **`Relu`, `Sigmoid`, `Tanh`, `LeakyRelu`, and `Gelu` are purely element-wise** and work on any tensor shape/rank, always preserving it. `Softmax` is the one exception, requiring a 4D input as noted above.

---

# Loss Functions Documentation

Loss functions quantify the difference between predictions and ground truth.

## Mean Squared Error (MSE)

```cpp
MSE mse;
Tensor loss = mse(predictions, targets);
```

**Definition:** `L = (1/N) * Σ(predictions_i - targets_i)²`

**Use Cases:** Regression, reconstruction tasks

**API:**
```cpp
Tensor operator()(const Tensor& predictions, const Tensor& targets);
```

## Cross-Entropy Loss

```cpp
CrossEntropy ce;
Tensor loss = ce(logits, targets);  // targets: one-hot encoded
```

**Definition:** `L = -(1/N) * Σ targets_i * log(softmax(logits_i))`

**Use Cases:** Multi-class classification

**Note:** Softmax is fused internally for numerical stability. Pass logits, not probabilities.

**API:**
```cpp
Tensor operator()(const Tensor& predictions, const Tensor& ground_truth);
```

## Binary Cross-Entropy (BCE)

```cpp
BinaryCrossEntropy bce;
Tensor loss = bce(logits, targets);  // targets: 0 or 1
```

**Definition:** `L = -(1/N) * Σ [targets_i * log(sigmoid(logits_i)) + (1 - targets_i) * log(1 - sigmoid(logits_i))]`

**Use Cases:** Binary classification, multi-label classification

**Note:** Sigmoid is fused internally for numerical stability. Pass logits, not probabilities.

**API:**
```cpp
Tensor operator()(const Tensor& predictions, const Tensor& ground_truth);
```
# Optimizers Documentation

This module provides the public-facing optimizers used to train models. Each optimizer owns the parameters it was constructed with, along with any internal state (momentum/moment buffers), and updates those parameters in place each time `update()` is called. Internally, `update()` dispatches to a device-specific backend (currently only CPU supported).

Two optimizers are currently available:

- **`Forge::SGD`** - stochastic gradient descent, with optional momentum
- **`Forge::Adam`** - adaptive moment estimation, with optional weight decay

## `Forge::SGD`

### Constructor

```cpp
explicit SGD(std::vector<Parameter> parameters, float lr = 0.01f, float momentum_coef = 0.0f);
```

| Argument | Meaning |
|---|---|
| `parameters` | The parameters this optimizer will update. Must all share the same device and dtype; duplicates are silently removed. |
| `lr` | Learning rate (step size). Default `0.01`. |
| `momentum_coef` | Momentum coefficient, must be in `[0, 1]`. `0.0` (default) = plain SGD. `> 0.0` = momentum SGD. |

If `momentum_coef > 0`, a zero-initialized velocity buffer is allocated internally for each parameter - no manual buffer setup needed.

### Methods

| Method | Meaning |
|---|---|
| `update()` | Applies one optimization step to all owned parameters, using their currently accumulated gradients. |
| `clear_grads()` | Zeroes the gradients of all owned parameters. Call after `update()`, before the next backward pass. |
| `setLearningRate(lr)` | Updates the learning rate. |
| `setMomentumCoef(momentum_coef)` | Updates the momentum coefficient (must be in `[0, 1]`). |
| `parameters()` | Read-only access to the owned parameters. |
| `learningRate()` | Returns the current learning rate. |
| `momentumCoef()` | Returns the current momentum coefficient. |

### Update rule

For each parameter `p` with gradient `g`:

- **Without momentum** (`momentum_coef == 0`):

  ```
  p = p - lr * g
  ```

- **With momentum** (`momentum_coef > 0`), using internal velocity buffer `V`:

  ```
  V = momentum_coef * V + g
  p = p - lr * V
  ```

  `V` accumulates an exponentially-weighted sum of past gradients, smoothing the descent direction and accelerating convergence along consistent gradient directions.

### Example

```cpp
Forge::SGD optimizer(model.parameters(), /*lr=*/0.01f, /*momentum_coef=*/0.9f);
Forge::MSE loss_fn;

for (auto& batch : batches) {
    optimizer.clear_grads();
    pred = model(batch)
    auto loss = loss_fn(ground_truth, pred);
    loss.backward();
    optimizer.update();
}
```

## `Forge::Adam`

### Constructor

```cpp
explicit Adam(const std::vector<Parameter>& parameters, float lr = 0.01f, float beta_1 = 0.9f,
    float beta_2 = 0.999f, float decay_factor = 0.01f);
```

| Argument | Meaning |
|---|---|
| `parameters` | The parameters this optimizer will update. Must all share the same device and dtype; duplicates are silently removed. |
| `lr` | Learning rate. Default `0.01`. |
| `beta_1` | Exponential decay rate for the first moment estimate, must be in `[0, 1]`. Default `0.9`. |
| `beta_2` | Exponential decay rate for the second moment estimate, must be in `[0, 1]`. Default `0.999`. |
| `decay_factor` | Weight-decay coefficient, applied only to parameters whose `need_decay` flag is set. Default `0.01`. |

Zero-initialized first- and second-moment buffers are allocated internally for each parameter. The step counter used for bias correction (`epoch`) is managed internally, starting at `1` and incrementing automatically on every `update()` call - no need to track it yourself.

### Methods

| Method | Meaning |
|---|---|
| `update()` | Applies one Adam step to all owned parameters, then advances the internal step counter. |
| `clear_grads()` | Zeroes the gradients of all owned parameters. Call after `update()`, before the next backward pass. |
| `reset()` | Resets the internal step counter back to `1` (e.g. when restarting training). Does not reset the moment buffers. |
| `setLearningRate(lr)` | Updates the learning rate. |
| `setBeta_1(beta_1)` | Updates beta_1 (must be in `[0, 1]`). |
| `setBeta_2(beta_2)` | Updates beta_2 (must be in `[0, 1]`). |
| `setDecayFactor(decay_rate)` | Updates the weight-decay coefficient (must be in `[0, 1]`). |
| `parameters()` | Read-only access to the owned parameters. |
| `firstMoment()` | Read-only access to the internal first-moment buffers (`M`). |
| `secondMoment()` | Read-only access to the internal second-moment buffers (`V`). |
| `beta_1()` / `beta_2()` | Current beta values. |
| `learningRate()` | Current learning rate. |
| `epoch()` | Current internal step counter. |
| `decayFactor()` | Current weight-decay coefficient. |

### Update rule

For each parameter `p` with gradient `g`, at step `t = epoch`:

```
M = beta_1 * M + (1 - beta_1) * g           # first moment
V = beta_2 * V + (1 - beta_2) * g^2          # second moment

M_hat = M / (1 - beta_1^t)                  # bias-corrected first moment
V_hat = V / (1 - beta_2^t)                  # bias-corrected second moment

p = p - lr * M_hat / (sqrt(V_hat) + e)       # e = 1e-8, fixed internally

if need_decay:
    p = p - lr * decay_factor * p           # weight decay, applied after the Adam step
```

### Example

```cpp
Forge::Adam optimizer(model.parameters(), /*lr=*/0.001f, /*beta_1=*/0.9f,
    /*beta_2=*/0.999f, /*decay_factor=*/0.01f);
Forge::MSE loss_fn;

for (auto& batch : dataloader) {
    optimizer.clear_grads();
    pred = model(batch)
    auto loss = loss_fn(ground_truth, pred);
    loss.backward();
    optimizer.update();
}
```

## Notes & gotchas

- **Construction validates inputs.** Constructing either optimizer with an empty parameter list, or with parameters spread across mixed devices/dtypes, throws `std::invalid_argument`. Duplicate parameters are detected and removed automatically, no need to de-duplicate yourself.
- **`momentum_coef`, `beta_1`, `beta_2`, and `decay_factor` are all range-checked** to `[0, 1]`, both at construction and whenever set via their setters.
- **Per-parameter weight decay:** decay is opt-in via each `Parameter`'s `need_decay` flag, not a global switch - set this when building your model if you want decay to skip certain parameters (e.g. biases/LayerNorm scales).
- **`Adam::epoch()` is managed for you** - it starts at `1` and increments on every `update()` call. Use `reset()` if you need to restart bias correction from scratch (e.g. after loading a fresh set of parameters into an existing optimizer).
- **`clear_grads()` is separate from `update()`** - remember to call it once per step (typically right before `loss.backward()`), or gradients will keep accumulating across steps.

  # LayerNorm

`Forge::LayerNorm` implements layer normalization over the last dimension of a 3D input tensor (`batch, seq_len, d_model`), with learnable per-feature scale (`gamma`) and shift (`beta`) parameters. Like the optimizers, the actual math runs through a device-specific backend (CPU, GPU, ...) resolved internally - callers just construct a `LayerNorm` and call it like a function.

---

## `Forge::LayerNorm`

### Constructor

```cpp
LayerNorm(std::size_t d_model, Dtype dtype, const Device& device, bool need_grads = true);
```

| Argument | Meaning |
|---|---|
| `d_model` | Size of the last dimension to normalize over (the feature dimension). |
| `dtype` | Data type for `gamma`, `beta`, and the computation. |
| `device` | Device the parameters live on (e.g. CPU, GPU). |
| `need_grads` | Whether `gamma`/`beta` are trainable (accumulate gradients). Default `true`. |

On construction, `gamma` is initialized to ones and `beta` to zeros, both with shape `[d_model]`.

### Call operator

```cpp
Tensor operator()(const Tensor& input);
```

Applies layer normalization to `input` and returns a new tensor of the same shape.

- `input` must be a 3D tensor shaped `(batch, seq_len, d_model)`, where the last dimension matches the `d_model` this `LayerNorm` was constructed with - otherwise it throws `std::invalid_argument`.
- `input` must reside on the same device as the `LayerNorm` instance - otherwise it throws `std::invalid_argument`.
- If `input`, `gamma`, or `beta` require gradients, the necessary autograd node is attached automatically - no manual backward wiring needed.

### Accessors

| Method | Meaning |
|---|---|
| `gamma()` | Mutable reference to the learnable scale parameter, shape `[d_model]`. |
| `beta()` | Mutable reference to the learnable shift parameter, shape `[d_model]`. |
| `d_model()` | Returns the configured feature dimension size. |
| `d_device()` | Returns the device this `LayerNorm` operates on. |
| `dtype()` | Returns the configured data type. |

### Computation

For each row `x` along the last dimension (i.e. each `(batch, seq_len)` position, a vector of length `d_model`):

```
mean = mean(x)                          # average over d_model
var  = mean((x - mean)^2)               # variance over d_model

x_norm = (x - mean) / sqrt(var + eps)   # eps = 1e-5, fixed internally

y = gamma * x_norm + beta               # gamma, beta broadcast over (batch, seq_len)
```

`gamma` and `beta` are applied element-wise per feature (broadcast across `batch` and `seq_len`), so every position in the sequence is rescaled/shifted the same way along the feature axis.

### Example

```cpp
Forge::LayerNorm ln(/*d_model=*/512, Dtype::Float32, Device::CPU);

Tensor x = ...; // shape (batch, seq_len, 512)
Tensor y = ln(x); // normalized output, same shape (batch, seq_len, 512)

y.backward(); // gradients flow back into x, ln.gamma(), and ln.beta() automatically

// gamma/beta are ordinary learnable parameters - hand them to an optimizer like any other:
Forge::Adam optimizer(ln.parameters(), /*lr=*/0.001f);
```

---

## Notes & gotchas

- **Input must be 3D.** `LayerNorm` expects `(batch, seq_len, d_model)` - reshape lower- or higher-rank tensors to this layout before calling it.
- **Normalization is over the last axis only.** Each `(batch, seq_len)` row is normalized independently across its `d_model` features; there is no cross-sequence or cross-batch normalization.
- **`eps` is fixed at `1e-5`** and is not currently configurable from the constructor.
- **`need_grads` only controls `gamma`/`beta`.** Even if you pass `need_grads = false`, the output tensor will still require gradients if the `input` you call it with requires gradients - gradient tracking is `need_grads OR input.need_grads()`.
- **Device/shape mismatches throw immediately** (`std::invalid_argument`) rather than silently broadcasting or casting - construct one `LayerNorm` per device/`d_model` combination you need.

# Self Attention

`Forge::SelfAttention` implements multi-head scaled dot-product self-attention with an optional causal mask, followed by a linear output projection. As with the other modules, the actual math runs through a device-specific backend resolved internally - you only ever interact with the `SelfAttention` object itself.

---

## `Forge::SelfAttention`

### Constructor

```cpp
SelfAttention(std::size_t d_model, std::size_t Q_K_dims, std::size_t V_dims, std::size_t heads, bool mask,
    Device device, Dtype dtype = Dtype::float32, Initializers initializer = Initializers::he_normal);
```

| Argument | Meaning |
|---|---|
| `d_model` | Input/output feature dimension. |
| `Q_K_dims` | Dimension of the query/key projection, per head. |
| `V_dims` | Dimension of the value projection, per head. |
| `heads` | Number of attention heads. |
| `mask` | Whether this instance uses causal masking. If `true`, you must call `createMask(seq_len)` before the first forward pass. |
| `device` | Device the weights live on. |
| `dtype` | Data type for weights and computation. Default `float32`. |
| `initializer` | Initialization scheme used for the query/key/value projection weights. Default `he_normal`. |

On construction, the query/key projection weights are allocated with shape `(heads, d_model, Q_K_dims)`, the value projection weights with shape `(heads, d_model, V_dims)`, and all three are initialized using `initializer`. An internal `Linear` layer (no bias) projects the concatenated multi-head output, `heads * V_dims`, back down to `d_model`.

### Call operator

```cpp
Tensor operator()(const Tensor& input) const;
```

Runs the forward pass and returns a tensor of shape `(batch, seq_len, d_model)`.

- `input` is expected as `(batch, seq_len, d_model)` (rank > 3 throws `std::invalid_argument`).
- If this instance was constructed with `mask = true`, you must have called `createMask(seq_len)` first - otherwise it throws `std::runtime_error`. The mask's size must also match the input's sequence length, or it throws `std::invalid_argument`.
- Gradients are wired up automatically for `input` and the query/key/value weights whenever any of them require gradients - no manual backward bookkeeping needed.

### `createMask`

```cpp
void createMask(std::size_t seq_len);
```

Builds a `(seq_len, seq_len)` causal mask and stores it internally. Only valid for instances constructed with `mask = true` (otherwise throws `std::runtime_error`). Call this once before the first forward pass, and again whenever `seq_len` changes.

### Accessors

| Method | Meaning |
|---|---|
| `query()` | Read-only access to the query projection weights, shape `(heads, d_model, Q_K_dims)`. |
| `key()` | Read-only access to the key projection weights, shape `(heads, d_model, Q_K_dims)`. |
| `value()` | Read-only access to the value projection weights, shape `(heads, d_model, V_dims)`. |
| `linear()` | Mutable access to the internal output-projection `Linear` layer (its own weights are also learnable - see the `Linear` module's docs). |
| `mask()` | Read-only access to the stored causal mask tensor. |
| `using_mask()` / `useMask()` | Whether this instance was constructed with masking enabled (both return the same flag). |
| `heads()` | Number of attention heads. |
| `d_model()` | Configured model dimension. |
| `device()` | Device the weights live on. |
| `dtype()` | Configured data type. |
| `dispatch_key()` | Internal device-routing key - not generally needed by callers. |

### Computation

Shapes: `input` is `(batch, seq_len, d_model)`; `query_W`/`key_W` are `(heads, d_model, Q_K_dims)`; `value_W` is `(heads, d_model, V_dims)`. `@` denotes matrix multiplication, batched over `batch` and `heads`.

For each head `h`:

```
Q = input @ query_W[h]                  # (batch, seq_len, Q_K_dims)
K = input @ key_W[h]                    # (batch, seq_len, Q_K_dims)
V = input @ value_W[h]                  # (batch, seq_len, V_dims)

scores = (Q @ K^T) / sqrt(Q_K_dims)     # (batch, seq_len, seq_len)

if using_mask:
    scores = scores + mask              # mask has -10000 at future positions (j > i)

attn = softmax(scores, axis=-1)         # (batch, seq_len, seq_len)
head_out = attn @ V                     # (batch, seq_len, V_dims)
```

The per-head outputs are concatenated along the last axis and projected back to `d_model`:

```
concat = concat_heads(head_out)         # (batch, seq_len, heads * V_dims)
output = concat @ W_out^T               # (batch, seq_len, d_model), W_out = linear() weights, no bias
```

### Example

```cpp
// Encoder-style self-attention, no mask
Forge::SelfAttention attn(/*d_model=*/512, /*Q_K_dims=*/64, /*V_dims=*/64, /*heads=*/8,
    /*mask=*/false, Device::CPU);

Tensor x = ...; // shape (batch, seq_len, 512)
Tensor y = attn(x); // shape (batch, seq_len, 512)
y.backward();
```

```cpp
// Decoder-style self-attention, causal mask
Forge::SelfAttention attn(/*d_model=*/512, /*Q_K_dims=*/64, /*V_dims=*/64, /*heads=*/8,
    /*mask=*/true, Device::CPU);

attn.createMask(seq_len); // required before the first forward pass

Tensor y = attn(x);
y.backward();

// query/key/value projection weights, plus the internal Linear's own weights,
// are all ordinary learnable parameters - register them with your optimizer
Forge::Adam optimizer(attn.parameters(), /*lr=*/0.0001f);
```

---

## Notes & gotchas

- **`createMask` must be called before forward when `mask = true`.** Forgetting it throws `std::runtime_error("create a mask prior to forward pass")`.
- **The mask is fixed at the sequence length you build it for.** If your sequence length changes between batches, call `createMask(new_seq_len)` again.
- **Heads up:** the input-vs-mask length check in `operator()` compares `input.shape()[0]` against the mask's sequence length. For a `(batch, seq_len, d_model)` input, `shape()[0]` is the batch size, not `seq_len` - it looks like it should be comparing against `shape()[1]` instead. As written, this check only behaves correctly when `batch_size == seq_len`, so don't rely on it to catch a real mismatch.
- **The output projection has no bias** - `linear()` is constructed with bias disabled.
- **Weight initialization applies only to Q/K/V.** The internal `Linear` layer initializes itself independently (see the `Linear` module's own docs for its default scheme).

## Embeddings

`Forge::Embedding` implements a learned token embedding lookup table -
mapping integer token IDs to dense `d_model`-dimensional vectors, the
first layer in a typical transformer.

### API

```cpp
Forge::Embedding embd(
    d_model,               // embedding dimension
    vocab_size,            // number of distinct tokens
    Dtype::float32,        // must be float32 or float64 - int32 is rejected at construction
    /*need_grads=*/true,
    Initializers::xavier_normal,
    Device::CPU
);

Tensor out = embd(seq_ids);  // seq_ids: 1-D int32 Tensor of token IDs -> (seq_len, d_model)
```

- `seq_ids` must be a 1-D `int32` tensor - any other dtype or rank throws
  `std::invalid_argument`.
- The embedding table itself (`m_embeddings`) must be `float32` (for now) -
  constructing with `Dtype::int32` throws immediately.
- Dispatches through `primitive_dispatcher()` to the correct backend (currently only CPU)
  implementation (`EmbeddingsImplAbstract`) based on the output tensor's
  dispatch key, keeping the op backend-agnostic at the call site.
- When `need_grads` is true, a gradient node (`EmbeddingsGradsAbstract`) is
  automatically attached to the autograd graph after the forward pass.
- `all_embeddings()` exposes the underlying weight tensor directly (e.g. for
  weight tying with an output projection).
- `parameters()` returns the embedding weight wrapped as a `Parameter`,
  named `embd.<n>.w`, for use with an optimizer.

  ## Tokenizer

 ## Tokenizer

`SimpleTokenizer` implements a from-scratch Byte Pair Encoding (BPE)
tokenizer with GPT-2-style pre-tokenization - the same word-splitting
regex GPT-2 uses (handling contractions, letter runs, digit runs, and
whitespace separately) before BPE merges are applied within each chunk.

### Training a vocabulary

```cpp
SimpleTokenizer tok(SimpleTokenizer::Type::BPE);
tok.on_file("corpus.txt", /*max_vocab=*/8000);
```

- Starts from a base vocabulary of 3 special tokens (`<pad>`, `<bos>`,
  `<eos>`) plus all 256 raw byte values, then iteratively merges the most
  frequent adjacent token pair in the corpus until `max_vocab` is reached
  or no pair occurs more than once.
- `max_vocab` must be at least 300 (`MIN_VOCAB_SIZE`) - lower throws
  `std::invalid_argument`.

### Loading a saved vocabulary

```cpp
tok.load("vocab.bin");
```

Vocabularies are stored in a simple binary format (`save`/`load`) - useful
for loading a pretrained vocabulary (e.g. reconstructed GPT-2 merge rules)
without retraining from a corpus.

### Encoding and decoding

```cpp
Forge::Tensor ids = tok.encode("he went to the");   // -> 1-D int32 Tensor of token IDs
StringVec tokens  = tok.decode(ids);           // -> token strings, one per ID
```

- `encode` returns a `Forge::Tensor` (int32), ready to feed directly into
  `Embedding::operator()`.
- `decode` throws `std::invalid_argument` if it encounters a token ID
  outside the loaded vocabulary.
- An optional `transformation` callback can be passed to `encode` to
  preprocess each pre-tokenized chunk (e.g. lowercasing) before BPE merges
  are applied.

## Reflection-Based Parameters & Model Load/Save (safetensors)

Forge uses [reflect-cpp](https://github.com/getml/reflect-cpp) to automatically
discover a model's trainable parameters from its struct layout, with no manual
registration required - any member exposing a `parameters()` method is picked
up automatically, including nested submodules.

### Defining a model

Any aggregate struct whose members expose `parameters() -> std::vector<Parameter>`
(e.g. `Linear`, `Embedding`, `LayerNorm`) is automatically reflectable:

```cpp
struct GPT2Block {
    Forge::SelfAttention attn;
    Forge::LayerNorm ln1;
    Forge::Linear    mlp_fc;
    Forge::LayerNorm ln2;
};
```

No base class, no manual parameter registration - `rfl::to_view` introspects
the struct's members directly.

### Extracting parameters for an optimizer

```cpp
auto params = Forge::extract_parameters(model);   // flat std::vector<Parameter>
Forge::Adam optimizer(params, /*lr=*/...);
```

Walks every member satisfying the `HasParameters` concept and collects their
`Parameter`s into one flat list - this is what feeds the optimizer without
you having to hand-list every layer's weights.

### Saving weights

```cpp
Forge::save(model, "weights.safetensors");
```

Writes an actual [safetensors](https://github.com/huggingface/safetensors)-format
file: an 8-byte header length, a JSON header describing each tensor's dtype,
shape, and byte offset (padded to an 8-byte boundary), followed by the raw
tensor data. Parameter names are built as `<member_name>.<param_name>`
(recursing into nested submodules via `get_state_dict`), matching the naming
convention each layer's own `parameters()` already establishes.

### Loading weights

```cpp
Forge::load(model, "weights.safetensors");
```

Reads the file, matches tensor names against the model's own state dict, and
copies loaded data directly into each parameter in place. Throws if the
number of tensors doesn't match the model's parameter count - a basic
architecture-mismatch guard. This is the mechanism used to load real GPT-2
weights directly into Forge's GPT-2 implementation.

`Forge::load_safetensors(filename)` is also available standalone, returning
a raw `std::map<std::string, Tensor>` without requiring a matching model
struct - useful for inspecting a checkpoint's contents directly.

