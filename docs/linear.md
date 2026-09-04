# Linear Layer

`Forge::Linear` class implements the dense fully connected linear layer, its one of the most fundamental transformation
in Deep Learning

linear transformation:
`y = I @ W.T + b`

here `I` (input) should have shape upto 4 rank `(batch_x, batch_y, batch_z, input_size)`, `W`(weights) is of shape `(output_size, input_size)`
and `B` (bias) has shape `(output_size)`

### Linear class metadata 

```c++
//the weight tensor
Tensor m_weights{};

//bias tensor
Tensor m_bias{};

//dispatch key for dispatcher to route to device specific kernels
DispatchKey m_dispatch_key;

//input size and output size
std::size_t m_input_size, m_output_size;

//if bias is being used
bool m_using_bias;

//dtype and device of linear layer
Dtype m_dtype;
Device m_device;

//if weights or bias (if any) need grads
bool m_need_grads{};

//instance tracker tracks how many instances of linear layer has been made
inline static InstanceTracker m_tracker;
int m_cnt {};
```
### Constructor

```c++
 Linear(std::size_t input_size, std::size_t output_size, bool need_grads=true, bool bias=true,
        Initializers initializer=Initializers::xavier_normal,
        Dtype dtype=Dtype::float32, Device device=Device::CPU);
```

here supported initializers are `Initializers::xavier_normal` (default),
`Initializers::xavier_uniform`, `Initializers::he_normal` and `Initializers::he_uniform`

examples:
```c++
#include "Forge.h"
using namespace Forge;

// Create a linear layer: 128 input features -> 64 output features
// Uses Xavier Normal initialization by default
Linear linear_1{128, 64};

// With custom data type
Linear linear_2{256, 128, Initializers::he_normal, Dtype::float32};

// Without bias term
Linear linear_no_bias{100, 50, Initializers::xavier_normal, Dtype::float32, Device::CPU, false};
```

### Forward pass

```c++
Tensor Linear::operator()(const Tensor& input);
```

example:
```c++
//create layer and input
Linear fc{784, 128};
Tensor input(Tensor::Random{32, 784});  // batch_size=32, features=784

//forward pass using operator()
Tensor output = fc(input);  //returns shape: (32, 128)

std::cout << "Input shape: ";
for (auto s : input.shape()) std::cout << s << " ";
std::cout << "\nOutput shape: ";
for (auto s : output.shape()) std::cout << s << " ";
std::cout << "\n";
```

### Backward Pass
```c++
Linear fc{784, 128};
Tensor input(Tensor::Random{32, 784});  // batch_size=32, features=784

//forward pass using operator()
Tensor output = fc(input);  //returns shape: (32, 128)
output.backward() //backpass
std::cout<<fc.weights().grads();
fc.weights().clear_grads(); //clear grads back to 0
```

### properties
| Member Function  | Usage                                                      |
|------------------|------------------------------------------------------------|
| `operator()`     | runs the linear layer on the input tensor                  |
| `parameters()`   | returns the layer's weights and bias as named parameters   |
| `weights()`      | returns the layer's weights                                |
| `bias()`         | returns the layer's bias                                   |
| `input_size()`   | returns the input feature size                             |
| `output_size()`  | returns the output feature size                            |
| `dispatch_key()` | returns the layer's kernels dispatch key                   |
| `device()`       | returns the device on which the layer's parameters resides |
| `dtype()`        | returns the layer's data type                              |

