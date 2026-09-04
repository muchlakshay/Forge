# Optimizers
this doc contains the optimizers used to train models. Each optimizer keeps track of the parameters it was given along with any internal state, such as momentum/moment buffers. Calling `update()`
applies the optimizer step and updates the parameters in place. The actual update is handled by the device-specific backend,
which currently supports CPU only.

two optimizers that are available:

**`Forge::SGD`** - stochastic gradient descent with optional momentum

**`Forge::Adam`** - adaptive moment estimation with optional weight decay

### Forge::SGD

**update rule:**

without momentum (`momentum_coef=0.0f`)

`
p = p - lr * g
`

with momentum (`momentum_coef>0.0f`)

`V = momentum_coef * V + g`

`p = p - lr * V`

**constructor:**
```c++
SGD::SGD(std::vector<Parameter> parameters, float lr=0.01f, float momentum_coef=0.0f);
```

here `parameters` are the list of all the parameters to optimize, if it contains duplicates they will be removed
automatically.

`lr` is the learning rate set to 0.01f by default

`momentum_coef` is the momentum coefficient (0.0f default), it must be in range [0, 1], `momentum_coef==0` then it will act like vanilla SGD
and if  `momentumc_coef>0.0f` it will act as momentum SGD

the momentum buffers are automatically handled internally

| Member Function                  | Usage                                                                                |
|----------------------------------|--------------------------------------------------------------------------------------|
| `update()`                       | performs an optimization step using the gradients currently stored in the parameters |
| `setLearningRate(lr)`            | changes the optimizer's learning rate                                                |
| `clear_grads()`                  | resets all parameter gradients to zero before the next backward pass                 |
| `parameters()`                   | provides read-only access to the parameters managed by the optimizer                 |
| `setMomentumCoef(momentum_coef)` | changes the momentum coefficient. The value must be between `0` and `1`              |
| `learningRate()`                 | returns the optimizer's current learning rate                                        |
| `momentumCoef()`                 | returns the current momentum coefficient                                             |

**Example**

```c++
Forge::SGD optimizer(model.parameters(), /*lr=*/0.01f, /*momentum_coef=*/0.9f);
Forge::MSE loss_fn{};

for (auto& batch : batches) {
    optimizer.clear_grads();
    pred {model(batch)} 
    
    auto loss {loss_fn(pred, ground_truth)}; 
    loss.backward();
    
    optimizer.update(); //update step
    optimizer.clear_grads() //zero the grads
}
```

### Forge::Adam

**update rule:**

```
M = beta_1 * M + (1 - beta_1) * g           # 1st moment
V = beta_2 * V + (1 - beta_2) * g^2          # 2nd moment

M_hat = M / (1 - beta_1^t)                  # bias-corrected 1st moment
V_hat = V / (1 - beta_2^t)                  # bias-corrected 2nd moment

p = p - lr * M_hat / (sqrt(V_hat) + e)       # e = 1e-8, fixed internally

if need_decay:
p = p - lr * decay_factor * p   # applying weight decay
```

**constructor**
```c++
 Adam(const std::vector<Parameter>& parameters, float lr=0.01f, float beta_1=0.9f, float beta_2=0.999f, float decay_factor=0.01f);
```
here `parameters` are the list of all the parameters to optimize, if it contains duplicates they will be removed
automatically.

`lr` is the learning rate

`beta_1` is the exponential decay rate for 1st moment, must be in range [0, 1] (default 0.9f)

`beta_2` is the exponential decay rate for 2nd moment, must be in range [0, 1] (default 0.999f)

`decay_factor` is the weight decay factor, its applied to only the parameters with `need_decay=true` (default 0,01f)

for each parameter, the optimizer creates zero-filled 1st and 2nd moment buffers internally. The `epoch` counter used for
bias correction is also handled automatically internally, starting at `1` and increasing with every `update()` call.

| Member Function              | Usage                                                              |
|------------------------------|--------------------------------------------------------------------|
| `update()`                   | performs one Adam optimization step and advances the internal step counter |
| `clear_grads()`              | resets all parameter gradients to zero before the next backward pass |
| `reset()`                    | Sets the internal step counter back to `1` without clearing the moment buffers |
| `setLearningRate(lr)`        | changes the current learning rate                                  |
| `setBeta_1(beta_1)`          | changes the `beta_1` value. Must be between `0` and `1`            |
| `setBeta_2(beta_2)`          | changes the `beta_2` value. Must be between `0` and `1`            |
| `setDecayFactor(decay_rate)` | changes the weight-decay coefficient. Must be between `0` and `1`  |
| `parameters()`               | provides read-only access to the parameters managed by the optimizer |
| `firstMoment()`              | provides read-only access to the first-moment buffers (`M`)        |
| `secondMoment()`             | provides read-only access to the second-moment buffers (`V`)       |
| `learningRate()`             | returns the current learning rate                                  |
| `beta_1()` / `beta_2()`      | returns the current beta values                                    |
| `decayFactor()`              | returns the current weight-decay coefficient                       |
| `epoch()`                    | returns the current internal step counter                          |

**Example**
```c++
Forge::Adam optimizer(model.parameters(), /*lr=*/0.001f, /*beta_1=*/0.9f,
    /*beta_2=*/0.999f, /*decay_factor=*/0.01f);
    
Forge::MSE loss_fn{};

for (auto& batch : dataloader) {
    pred {model(batch)};
    auto loss {loss_fn(ground_truth, pred)};
    loss.backward();
    
    optimizer.update(); //update
    optimizer.clear_grads(); //zero grads
}
```

