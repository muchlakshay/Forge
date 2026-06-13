# Forge: A Deep Learning Framework from Scratch

**Forge** is a from-scratch deep learning framework implemented entirely in **C++**. It provides a modular, performant foundation for building and training neural networks with a focus on systems-level optimization and educational clarity.

## Overview

Forge implements core deep learning abstractions and primitives, enabling you to build, train, and optimize neural network models efficiently. The framework is designed for performance and simplicity, making it suitable for both learning and production use cases.

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

  ---

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

---

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

---

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

### Slicing/Indexing

```cpp
Tensor batch = Tensor::Zeros({32, 128, 64});  // batch_size=32

// Get first sample from batch
Tensor first_sample = batch[0];  // Shape: (128, 64)

// Get tenth sample
Tensor tenth_sample = batch[9];  // Shape: (128, 64)
```

---

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

### More Complex Eigen Operations

```cpp
// 3D tensor: batch_size=4, sequence_length=8, embedding_dim=64
Tensor embeddings({4, 8, 64}, true);
embeddings = 0.0f;

auto emb_map = embeddings.as_eigen<float, 3>();

// Initialize embeddings with small random-like values
for (int b = 0; b < 4; b++) {
    for (int s = 0; s < 8; s++) {
        for (int d = 0; d < 64; d++) {
            emb_map(b, s, d) = (d + s + b) * 0.001f;
        }
    }
}

// Access specific batch sample
for (int s = 0; s < 8; s++) {
    for (int d = 0; d < 64; d++) {
        float val = emb_map(0, s, d);  // First batch item
        // Use val...
    }
}
```

---

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

---

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

---


