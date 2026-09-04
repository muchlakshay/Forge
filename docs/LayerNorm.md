# LayerNorm

this class, `Forge::LayerNorm` applies layer normalization to the last dimension of a input tensor.
it has learnable `gamma` and `beta` parameters to scale and shift the normalized values.
you can simply create a `LayerNorm` and call it directly on a input `Tensor`.

**Steps**
```
m = mean(x)                          # mean over last dim
var  = mean((x - mean)^2)               # variance over last dim
x_norm = (x - mean) / sqrt(var + eps)   # eps = 1e-5, fixed internally
y = gamma * x_norm + beta               # gamma, beta broadcast over the first three dims
```

### Constructor
```c++
LayerNorm(const std::size_t d_model, bool need_grads=true, Dtype dtype=Dtype::float32, const Device& device=Device::CPU);
```

here `d_model` is the size of last dim of the tensor

`need_grads` flag, if the LayerNorm's parameters needs gardients

`dtype` for the dtype of the parameters (Default - `Dtype::float32`)

`device` is the device on which the parameters reside on (currently only `Device::CPU`)

### Member  Functions

| Member Functions | Usage                                                                     |
|------------------|---------------------------------------------------------------------------|
| `gamma()`        | returns the reference to the learnable scale parameter, shape `[d_model]` |
| `beta()`         | returns the reference to the learnable shift parameter, shape `[d_model]` |
| `d_model()`      | returns the configured feature dimension size                             |
| `dtype()`        | returns the data type of parameters                                       |
| `parameters()`   | returns a `std::vector<Parameter>` of LayerNorm's parameters              |
| `d_device()`     | eturns the device this `LayerNorm` reside                                 |



**Example**
```c++
Forge::LayerNorm ln(/*d_model=*/64);

Tensor x {Tensor::Random({10, 64})}; //random input tensor
Tensor y {ln(x)}; // normalized output over last dim

y.backward(); //backpass
```

