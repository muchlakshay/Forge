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

---

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

---

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

### Manual Weight Inspection and Modification

```cpp
Linear layer(10, 5);

// Access weights via Eigen
auto w_map = layer.weights().as_eigen<float, 2>();

// Print weight matrix
std::cout << "Weight matrix:\n";
for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 5; j++) {
        std::cout << w_map(i, j) << " ";
    }
    std::cout << "\n";
}

// Access bias via Eigen
auto b_map = layer.bias().as_eigen<float, 4>();
std::cout << "Bias values: ";
for (int d = 0; d < 5; d++) {
    std::cout << b_map(0, 0, 0, d) << " ";
}
std::cout << "\n";
```

---

## Building Neural Networks with Linear Layers

### Simple MLP (Multi-Layer Perceptron)

```cpp
#include "Forge.h"
using namespace Forge;

class SimpleNN {
public:
    Linear fc1;
    Linear fc2;
    Linear fc3;
    Relu relu;
    
    SimpleNN() 
        : fc1(784, 256, Initializers::he_normal),
          fc2(256, 128, Initializers::he_normal),
          fc3(128, 10, Initializers::xavier_normal) {}
    
    // Forward pass
    Tensor operator()(const Tensor& input) const {
        Tensor h1 = relu(fc1(input));     // (batch, 784) -> (batch, 256) -> ReLU
        Tensor h2 = relu(fc2(h1));        // (batch, 256) -> (batch, 128) -> ReLU
        Tensor out = fc3(h2);             // (batch, 128) -> (batch, 10)
        return out;
    }
    
    // Collect parameters for optimization
    std::vector<Parameter> parameters() {
        return {
            {&fc1.weights(), true},   // Decay enabled
            {&fc2.weights(), true},
            {&fc3.weights(), true},
            {&fc1.bias(), false},     // No decay for bias
            {&fc2.bias(), false},
            {&fc3.bias(), false}
        };
    }
};
```

### Training Loop with Linear Layers

```cpp
int main() {
    SimpleNN model;
    CrossEntropy loss_fn;
    Adam optimizer(model.parameters(), 0.001f);
    
    // Training data
    Tensor x_train({1000, 784});  // 1000 samples, 784 features
    Tensor y_train({1000, 10});   // 10 classes
    
    int epochs = 10;
    int batch_size = 32;
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        float epoch_loss = 0.0f;
        
        for (int b = 0; b < x_train.shape()[0]; b += batch_size) {
            // Get batch
            Tensor batch = x_train[b];
            Tensor targets = y_train[b];
            
            // Forward pass through linear layers
            Tensor logits = model(batch);
            
            // Compute loss
            Tensor loss = loss_fn(logits, targets);
            epoch_loss += *static_cast<float*>(loss.data());
            
            // Backward pass
            loss.backward();
            
            // Update weights (including linear layer weights/biases)
            optimizer.update();
            optimizer.clear_grads();
        }
        
        float avg_loss = epoch_loss / (x_train.shape()[0] / batch_size);
        std::cout << "Epoch " << epoch << " - Loss: " << avg_loss << "\n";
    }
    
    return 0;
}
```

---

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




