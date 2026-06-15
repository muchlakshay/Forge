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
# Activation Functions Documentation

Activation functions are non-linear transformations applied to the output of neural network layers. They introduce non-linearity into the model, enabling it to learn complex patterns. Forge supports 6 commonly-used activation functions.

---

## ReLU (Rectified Linear Unit)

**Mathematical Definition:**
```
ReLU(x) = max(0, x)
```

**Characteristics:**
- Simplest and most computationally efficient activation
- Zero gradient for negative inputs (dying ReLU problem)
- Output range: [0, ∞)

**Use Cases:**
- Hidden layers in CNNs and MLPs
- First choice for deep networks
- Default activation in modern architectures

### Usage Example

```cpp
#include "Forge.h"
using namespace Forge;

int main() {
    Relu relu;
    
    // Forward pass
    auto x = Tensor::Range(-10, 10, 1, true);  // Values from -10 to 10
    
    Tensor activated = relu(x);
    
    // ReLU zeros out all negative values
    // x: [-10, -9, ..., -1, 0, 1, ..., 9]
    // output: [0, 0, ..., 0, 0, 1, ..., 9]
    
    std::cout << "Output after ReLU:\n" << activated << "\n";
    
    // Backward pass (automatic via autograd)
    activated.backward();
    auto grads = x.grads();  // Gradient of ReLU
    
    return 0;
}
```

### In a Neural Network

```cpp
class ReLUNetwork {
public:
    Linear fc1, fc2, fc3;
    Relu relu;
    
    ReLUNetwork() 
        : fc1(784, 256), fc2(256, 128), fc3(128, 10) {}
    
    Tensor operator()(const Tensor& input) const {
        return fc3(relu(fc2(relu(fc1(input)))));
    }
};
```

---

## Sigmoid

**Mathematical Definition:**
```
Sigmoid(x) = 1 / (1 + e^(-x))
```

**Characteristics:**
- Output range: (0, 1) - perfect for probabilities
- Smooth gradient everywhere
- Prone to vanishing gradient problem
- Computationally more expensive than ReLU

**Use Cases:**
- Output layer for binary classification
- Probability outputs

### Usage Example

```cpp
#include "Forge.h"
using namespace Forge;

int main() {
    Sigmoid sigmoid;
    
    // Forward pass
    auto logits = Tensor::Range(-5, 5, 0.1f, true);
    
    Tensor probabilities = sigmoid(logits);
    // output ∈ (0, 1) - can be interpreted as probability
    
    std::cout << "Probabilities:\n" << probabilities << "\n";
    
    // Backward pass
    probabilities.backward();
    auto grads = logits.grads();
    
    return 0;
}
```

### Binary Classification Example

```cpp
class BinaryClassifier {
public:
    Linear fc1, fc2;
    Relu relu;
    Sigmoid sigmoid;
    
    BinaryClassifier() 
        : fc1(784, 128), fc2(128, 1) {}
    
    Tensor operator()(const Tensor& input) const {
        return sigmoid(fc2(relu(fc1(input))));  // Output: probability [0, 1]
    }
};

int main() {
    BinaryClassifier model;
    BinaryCrossEntropy loss_fn;
    
    Tensor x({64, 784});
    Tensor y_true({64, 1});  // Binary labels: 0 or 1
    
    Tensor predictions = model(x);  // Shape: (64, 1), values ∈ [0, 1]
    Tensor loss = loss_fn(predictions, y_true);
    
    loss.backward();
    // Update weights...
    
    return 0;
}
```

---

## Tanh (Hyperbolic Tangent)

**Mathematical Definition:**
```
Tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```

**Characteristics:**
- Output range: (-1, 1) - centered at 0
- Stronger gradient than Sigmoid
- Still suffers from vanishing gradient

**Use Cases:**
- When centered output is beneficial
- Alternative to Sigmoid

### Usage Example

```cpp
#include "Forge.h"
using namespace Forge;

int main() {
    Tanh tanh;
    
    // Forward pass
    auto x = Tensor::Range(-3, 3, 0.05f, true);
    
    Tensor activated = tanh(x);
    // output ∈ (-1, 1)
    
    std::cout << "Tanh output range: (-1, 1)\n";
    std::cout << "Sample outputs:\n" << activated << "\n";
    
    // Backward pass
    activated.backward();
    
    return 0;
}
```

---

## GELU (Gaussian Error Linear Unit, tanh approximation)

**Mathematical Definition:**
```
GELU(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
```

**Characteristics:**
- Smooth, differentiable activation
- Output range: approximately (-0.17, inf) for typical inputs
- State-of-the-art in transformer models
- More computationally expensive than ReLU but better performance

**Use Cases:**
- Transformer models (BERT, GPT)
- Modern NLP architectures
- When maximum accuracy is prioritized over speed

### Usage Example

```cpp
#include "Forge.h"
using namespace Forge;

int main() {
    Gelu gelu;
    
    // Forward pass
    Tensor x({32, 768});  // Typical transformer hidden size

    Tensor activated = gelu(x);
    
    std::cout << "GELU output:\n" << activated << "\n";
    
    // Backward pass
    activated.backward();
    
    return 0;
}
```

### Transformer Architecture

```cpp
class TransformerBlock {
public:
    Linear fc1, fc2;
    Gelu gelu;
    
    // Feed-forward network with GELU
    Tensor forward(const Tensor& x) const {
        Tensor hidden = gelu(fc1(x));  // Expand to 4x hidden dim
        return fc2(hidden);             // Project back
    }
};
```

---

## Softmax

**Mathematical Definition:**
```
Softmax(x_i) = e^(x_i) / Σ(e^(x_j))
```

**Characteristics:**
- Converts logits to probability distribution
- Output sums to 1.0
- Used exclusively at output layer
- Numerically stable implementations required

**Use Cases:**
- Multi-class classification output layer
- Attention weights normalization
- Probability distributions

### Usage Example

```cpp
#include "Forge.h"
using namespace Forge;

int main() {
    Softmax softmax;
    
    // Forward pass
    Tensor logits({32, 10});  // 10 classes
    
    Tensor probabilities = softmax(logits);
    // Each row sums to 1.0 and represents class probabilities
    
    std::cout << "Class probabilities:\n" << probabilities << "\n";
    
    // Verify probabilities sum to 1
    auto prob_map = probabilities.as_eigen<float, 2>();
    float row_sum = 0.0f;
    for (int j = 0; j < 10; j++) {
        row_sum += prob_map(0, j);
    }
    std::cout << "First row sum: " << row_sum << " (should be ≈ 1.0)\n";
    
    return 0;
}
```

### Multi-Class Classification

```cpp
class MultiClassifier {
public:
    Linear fc1, fc2, fc3;
    Relu relu;
    Softmax softmax;
    
    MultiClassifier() 
        : fc1(784, 256), fc2(256, 128), fc3(128, 10) {}
    
    Tensor operator()(const Tensor& input) const {
        Tensor hidden1 = relu(fc1(input));
        Tensor hidden2 = relu(fc2(hidden1));
        Tensor logits = fc3(hidden2);
        return softmax(logits);  // Convert to probabilities
    }
};

int main() {
    MultiClassifier model;
    CrossEntropy loss_fn;  // Often fused with softmax
    
    Tensor x({64, 784});
    Tensor y_true({64, 10});  // One-hot encoded labels
    
    Tensor predictions = model(x);  // Probabilities, shape: (64, 10)
    Tensor loss = loss_fn(predictions, y_true);
    
    loss.backward();
    
    return 0;
}
```

---

## Leaky ReLU

**Mathematical Definition:**
```
LeakyReLU(x) = max(a*x, x)  where α is typically 0.01
```

**Characteristics:**
- Allows small gradient for negative inputs (fixes dying ReLU)
- Smoother gradients than ReLU
- Minimal computational overhead vs ReLU
- Rarely dramatically better than ReLU, but safer

**Use Cases:**
- Alternative when ReLU neurons die
- When gradient flow in negative region is desired

### Usage Example

```cpp
#include "Forge.h"
using namespace Forge;

int main() {
    LeakyRelu leaky_relu;  // Default α = 0.01
    
    // Forward pass
    Tensor x({32, 64});
    
    Tensor activated = leaky_relu(x);
    // Negative values are scaled by α instead of being zeroed
    
    std::cout << "Leaky ReLU output:\n" << activated << "\n";
    
    // Backward pass
    activated.backward();
    
    return 0;
}
```

### GAN Generator Example

```cpp
class GANGenerator {
public:
    Linear fc1, fc2, fc3, fc4;
    LeakyRelu leaky_relu;
    
    GANGenerator() 
        : fc1(100, 256), fc2(256, 512), fc3(512, 1024), fc4(1024, 784) {}
    
    Tensor operator()(const Tensor& noise) const {
        Tensor h1 = leaky_relu(fc1(noise));
        Tensor h2 = leaky_relu(fc2(h1));
        Tensor h3 = leaky_relu(fc3(h2));
        Tensor output = fc4(h3);  // Raw image data
        return output;
    }
};
```

---




