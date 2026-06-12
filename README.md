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

