# Loss Functions
Loss functions measure how far the model's predictions are from the expected ground truth.

Forge supports Mean Squared Error, Cross Entropy Loss (log softmax fused), Binary Corss Entropy (sigmoid fused)

### Mean Squared Error (MSE)

`L = 1/N * sigma(y_j-x_j)^2`

example:
```c++
Forge::MSE mse{};
std::cout<<"Loss: "<<mse(pred, ground_truth)
```

`operator()` overload 
```c++
Tensor operator()(const Tensor& predictions, const Tensor& targets);
```

### Cross Entropy Loss

it fuses log softmax internally, no need to apply softmax prior to it

`L = -(1/N) * sigma( targets_i * log(softmax(logits_i)) )`

example:
```c++
Forge::CrossEntropy ce{};
std::cout<<"Loss: "<<ce(logits, ground_truth)
```

`operator()` overload
```c++
Tensor operator()(const Tensor& predictions, const Tensor& targets);
```

 ### Binary Cross Entropy

it fuses sigmoid internally, no need to apply sigmoid prior to it.

`L = -(1/N) * sigma (targets_i * log(sigmoid(logits_i)) + (1 - targets_i) * log(1 - sigmoid(logits_i)))`

example:
```c++
Forge::BinaryCrossEntropy bce{};
std::cout<<"Loss: "<<bce(logits, ground_truth)
```

`operator()` overload
```c++
Tensor operator()(const Tensor& predictions, const Tensor& targets);
```

