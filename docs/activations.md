# Activations
this doc contains the activation functions used throughout the models. Each function is a lightweight, stateless functor class
that can be constructed and called directly with a `Forge::Tensor`. When gradients are required, they are handled automatically during the `.backward()` pass, so you don't need to manually call any gradient functions.

Forge supports RELU, Sigmoid, LeakyRELU, GELU, Tanh and softmax

### Forge::Relu

`relu(x) = max(0, x)`

example:
```c++
Forge::Relu relu{};

Tensor x {Tensor::Random({10, 10})}; 
Tensor y {relu(x)}; // same shape as x
y.backward();
```
### Forge::Sigmoid

`sigmoid(x) = 1/(1+exp(-x))`

example:
```c++
Forge::Sigmoid sigmoid{};

Tensor x {Tensor::Random({10, 10})}; 
Tensor y {sigmoid(x)}; // same shape as x
y.backward();
```
### Forge::Tanh

`tanh(x) = (exp(x)-exp(-x))/(exp(x)+exp(-x))`

example:
```c++
Forge::Tanh tanh{};

Tensor x {Tensor::Random({10, 10})}; 
Tensor y {tanh(x)}; // same shape as x
y.backward();
```

### Forge::LeakyRelu

`leaky_relu(x)=max(a*x, x) [a=0.01 genrally, fixed in Forge]`

example:
```c++
Forge::LeakyRelu lr{};

Tensor x {Tensor::Random({10, 10})}; 
Tensor y {lr(x)}; // same shape as x
y.backward();
```

### Forge::Gelu
Forge uses the tanh approximation for GELU

`gelu(x) =  0.5 * x * (1 + tanh(k * (x + c * x^3))) here [k=sqrt(2 / pi) = 0.7978845608028654 and c = 0.044715]
`

example:
```c++
Forge::Gelu gelu{};

Tensor x {Tensor::Random({10, 10})}; 
Tensor y {gelu(x)}; // same shape as x
y.backward();
```

### Forge::Softmax

this class computes softmax along the last axis

`softmax(x) = exp(x_i-max(x)) / sigma_j(exp(x_j - max(x)))`

example:
```c++
Forge::Softmax softmax{};

Tensor x {Tensor::Random({10, 10})}; 
Tensor y {softmax(x)}; // same shape as x
y.backward();
```

