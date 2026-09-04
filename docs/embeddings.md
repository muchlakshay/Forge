# Embeddings

`Forge::Embedding` provides a learned embedding table that converts integer token IDs into vectors of size `d_model`.
it is typically used as the first layer of a transformer to turn token IDs into continuous representations

### Constructor

```c++
 Embedding(std::size_t d_model, std::size_t vocab_size, Dtype dtype=Dtype::float32,
        bool need_grads=true, Initializers initializer = Initializers::xavier_normal, Device device=Device::CPU);
```
here `d_model` is the size of each embedding vector

`vocab_size` is the number of tokens in the vocabulary

`dtype` is data type used for the embedding weights (defaults to `float32`)

`need_grads` flag for whether embedding weight needs grads for training (Defaults to `true`)

`initializer` is the initializer used to initialize the embedding weights (Defaults to `xavier_normal`).
see the Tensor documentation for all the supported initializers

`device` is the device where the embedding weights are stored

### Example

```c++
Forge::Embedding embd(
    d_model,               // embedding dimension
    vocab_size,            // number of distinct tokens
);

Tensor out {embd(seq_ids)};  // seq_ids are 1-D int32 Tensor of token IDs -> (seq_len, d_model)
```

### Member Functions
| Member Function | Usage                                    |
|-----------------|------------------------------------------|
| `d_model()`     | returns the size of the embedding vector |
| `vocab_size()`  | returns the vocabulary size              |


