# Self Attention

`Forge::SelfAttention` provides multi-head self-attention using scaled dot-product attention, with support for an optional
causal mask to restrict which tokens can attend to others. the attention output is then passed through a linear projection.

**steps**
```
Q = I@Q_W    # (batch, heads, seq_len, Q_K_dims)
K = I@K_W    # (batch, heads, seq_len, Q_K_dims)
V = I@V_W    # (batch, heads, seq_len, V_dims)

sc = (Q@K_T)/sqrt(d_model)   # (batch, heads, seq_len, seq_len)

if (using_mask) sc+=mask

attn_scores =  softmax(sc) @ V       # (batch, heads, seq_len, V_dims)

opt = linear_proj(attn_scores) # (batch, seq_len, d_model)
```

### Constructor

```c++
SelfAttention(std::size_t d_model, std::size_t Q_K_dims, std::size_t V_dims, std::size_t heads, bool mask,
        bool need_grads = true, bool qkv_bias=false, bool proj_bias=false, Device device=Device::CPU, Dtype=Dtype::float32,
        Initializers initializer=Initializers::he_normal);
```
here `d_model` is  the size of the input and output feature dimension

`Q_K_dims` is the size of the query and key projections for each head

`V_dims` is the size of the value projections for each head

`heads` are the number of attention heads used

`mask` enables causal masking when set to `true`. In this case, `createMask(seq_len)` must be called before the first forward pass

`device` specifies where the model weights reside.

`dtype` is the data type used for the weights and computations (Defaults to `Dtype::float32`)

`initializer` determines how the query, key, and value projection weights are initialized (Defaults to `he_normal`), see the 
`Tensor` documentation for all the supported initializers

`qkv_bias` determines if Q, K and V have bias or not

`proj_bias` determines if the internal linear projection has bias or not

when the instance is created, the query and key weights are allocated with the shape `(heads, d_model, Q_K_dims)`,
while the value weights use `(heads, d_model, V_dims)`. all of these weights are initialized with the selected `initializer`.
the outputs from all attention heads are then combined and passed through an internal `Linear` layer (with optional bias),
which maps `heads * V_dims` back to `d_model`.

### Member Functions

| Member Function  | Usage                                                                            |
|------------------|----------------------------------------------------------------------------------|
| `query()`        | returns the query projection weights                                             |
| `key()`          | returns the key projection weights                                               |
| `value()`        | returns the value projection weights                                             |
| `useMask()`      | returns whether causal masking is enabled                                        |
| `dispatch_key()` | returns the kernel dispatch key of the instance                                  |
| `device()`       | returns the device on which the parameters reside                                |
| `dtype()`        | returns the data type used for the parameters                                    |
| `heads()`        | returns the number of attention heads                                            |
| `d_model()`      | returns the input/output feature dimension                                       |
| `need_grads()`   | indicates whether gradients are required for the parameters                      |
| `qkv_bias()`     | indicates whether the query, key, and value projections use bias terms           |
| `proj_bias()`    | indicates whether the output projection uses a bias term                         |
| `query_bias()`   | returns the query projection bias                                                |
| `key_bias()`     | returns the key projection bias                                                  |
| `value_bias()`   | returns the value projection bias                                                |
| `parameters()`   | returns a `std::vector<Parameter>` containing all the parameters of the instance |

### Example

**without mask**

```c++
Forge::SelfAttention attn{/*d_model=*/512, /*Q_K_dims=*/64, /*V_dims=*/64, /*heads=*/8,
    /*mask=*/false};

Tensor x {Tensor::Random({10, 512})};
Tensor y {attn(x)}; // shape (batch, seq_len, 512)
y.backward(); //backpass
```

**with mask**

```c++
Forge::SelfAttention attn{/*d_model=*/512, /*Q_K_dims=*/64, /*V_dims=*/64, /*heads=*/8,
    /*mask=*/true};

attn.createMask(seq_len); // required before the first forward pass

Tensor y {attn(x)};
y.backward(); //backpass
```

