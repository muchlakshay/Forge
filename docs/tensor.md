# Tensor
`Forge::Tensor` is one of the most important core class that gives an abstraction over multidimensional data 

### Tensor Metadata

```c++
class Forge::Tensor {
    //tensor residence (currently only Device::CPU)
    Device m_device{};
    
    //tensor data type (currently only fp32 and int32)
    Dtype m_dtype{};
    
    //Storage object held by the *this
    std::shared_ptr<StorageAbstract> m_storage{};
    
    //ofc shapes, strides and size
    std::vector<std::size_t> m_shape{}, m_strides{};
    std::size_t m_size{};   
    
    //flag for if tensor requires grads
    bool m_need_grads{};
    
    //dispatch key for the dispatcher to route device dependent kernels
    DispatchKey m_dispatch_key{};
    
    //autodiff node
    mutable std::shared_ptr<NodeAbstract> m_node{};
    
    //tensor gradients
    std::shared_ptr<Tensor> m_grads{};
```

- Forge implements a radix-tree based CPU memory allocator that `Tensor` uses internally for memory allocations. It holds
up the memory cache until its freed explicitly using `cpu_memory_pool().hard_clear_cache()`, `cpu_memory_pool()`
returns a static `MemoryPoolCPU` object used by Forge. it invalidates the in-use memory.
- `Tensor` holds the device on which the tensor resides on (currently only CPU supported), the dtype of tensor (only fp32 
  and int32 supported currently), the storage object that manages storage for tensor (allocations, deallocations and copying), 
  its shape, strides, size, if it needs gradients, a dispatch key for the dispatcher to route device specific kernels, 
  an autodiff node in case the tensor is a result of an operation and the parents needs gradients and tensor's own gradients.

### Constructors

```c++
Tensor(const std::vector<std::size_t>& shape, Dtype dtype = Dtype::float32, bool need_grads = true,
       Device device = Device::CPU);
       
//move and copy ctors
Tensor(const Tensor& another);
Tensor(Tensor&& another) = default;

//move and copy assignment operator
Tensor& operator=(const Tensor& another);
Tensor& operator=(Tensor&& another) = default;
```

### Factory Functions

these are static data members of `Tensor` class

```c++
Forge::Tensor::Constant(const std::vector<std::size_t>& shape, const Scalar& constant, bool need_grads=true,
        Dtype dtype=Dtype::float32 ,Device device=Device::CPU);
```
`Tensor::Constant` makes tensor filled with a constant value
```c++
Forge::Tensor::Ones(const std::vector<std::size_t>& shape, bool need_grads=true,
    Dtype dtype=Dtype::float32 ,Device device=Device::CPU);
```
`Tensor::Ones` returns a tensor filled with ones
```c++
Forge::Tensor::Zeros(const std::vector<std::size_t>& shape, bool need_grads=true,
    Dtype dtype=Dtype::float32 ,Device device=Device::CPU);
```
`Tensor::Zeros` returns a tensor filled with zeros
```c++
Forge::Tensor::FromHostPtr(T* host_ptr, const std::vector<std::size_t>& shape, bool need_grads=true);
```
`Tensor::FromHostPtr` constructs a view over an preallocated host/cpu memory
```c++
Forge::Tensor::Range(T start, T end, T step=1, bool need_grads=true, Device device=Device::CPU);
```
`Tensor::Range` returns a 1D tensor with values ranging from "start" to "end" with "step"
```c++
Forge::Tensor::Random(const std::vector<std::size_t>& shape, Dtype=Dtype::float32,
        Initializers initializer=Initializers::xavier_normal, Device device=Device::CPU, bool need_grads=true);
```
`Tensor::Random` returns a tensor filled with random values, the four initializers supported are
`Initializers::xavier_normal` (default),
`Initializers::xavier_uniform`, `Initializers::he_normal` and `Initializers::he_uniform`

### Tensor Manipulation

Reshaping:

```c++
Tensor::reshape(const std::vector<std::size_t> &shape)
```
example:
```c++
//flat tensor
Tensor flat {Tensor::Range(0, 24)};  // [0, 1, 2, ..., 23]
//reshape to (2, 3, 4)
Tensor reshaped {flat.reshape(2, 3, 4)};
//elements remain the same, just different view
```
Cloning and Copying:
```c++
Tensor::clone()
Tensor::copy(const Tensor& another)
```
`Tensor::clone()` returns a deep copy while `Tensor::copy()` copies the metadata of `another` while sharing the same storage object

example:
 ```c++
Tensor original {Tensor::Ones({3, 3}, true)};
//deep clone - independent copy
Tensor independent_copy {original.clone()};

//copy data from one tensor to another
Tensor destination({3, 3});
destination.copy(original);
 ```

indexing:
```c++
Tensor batch {Tensor::Zeros({32, 128, 64})};  // batch_size=32

// Get first sample from batch
Tensor first_sample {batch[0]};  // Shape: (128, 64)

// Get tenth sample
Tensor tenth_sample {batch[9]};  // Shape: (128, 64)
```

### Accessing the underlying data as `Eigen::TensorMap`

```c++
Eigen::TensorMap<Eigen::Tensor<T, Rank, Eigen::RowMajor>> Forge::Tensor::as_eigen()
```

### autodiff
every operation in Forge make as autodiff node if any of the parent tensor need grads

example with `Linear` layer:

```c++
Linear l {{10, 32}};
Tensor x {Tensor::Ones({32, 10}, true)};

//forward pass
Tensor opt {l(x)};

//backpass
opt.backward() //backward() has an optional keep_graph=true parameter, set it to false if u want to destroy the graph after backpass

std::cout<<x.gradients();

x.clear_grads() //to set grads back to 0
```

### Other Member Functions

| Member Function  | Usage                                                       |
|------------------|-------------------------------------------------------------|
| `need_grads()`   | returns whether the tensor needs the gradients              |
| `shape()`        | returns the tensor's dimensions                             |
| `strides()`      | returns the tensor's memory strides                         |
| `dtype()`        | returns the tensor's data type                              |
| `device()`       | returns the device on which the tensor resides              |
| `data()`         | returns a void pointer to the tensor's underlying data      |
| `size()`         | returns the total number of elements                        |
| `dispatch_key()` | returns the tensor's kernel dispatch key                    |
| `storage()`      | returns the tensor's underlying storage object              |
| `node()`         | returns the tensor's autodiff node object                   |
| `grads()`        | returns the tensor's gradient storage pointer               |
| `gradients()`    | returns the gradient tensor directly                        |
| `backward()`     | runs backpropagation through the tensor's computation graph |
| `clear_grads()`  | resets the tensor's gradients to zero                       |