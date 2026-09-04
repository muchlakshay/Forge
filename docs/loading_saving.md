# Reflection-Based Parameters & Model Load/Save (safetensors)

Forge uses reflect-cpp to extract a model’s trainable parameters from its struct layout. you do not have to register each layer or parameter.
when a member has a parameters() method Forge can detect it even if primitives are nested within other ones

### defining a model

any struct that contains members exposing `parameters()` which returns `std::vector<Parameter>` can be used with Forge’s reflection system. for example:

```c++
struct GPT2Block {
    Forge::SelfAttention attn;
    Forge::LayerNorm ln1;
    Forge::Linear mlp_fc;
    Forge::LayerNorm ln2;
};
```

declare all of ur data members of the model that has parameters in a separate aggregate struct, then make an instance of that
struct in the main model definition.

that is essentially all that is required.

### Extracting parameters for an optimizer

once the model is defined, retrieving all of its parameters is simply using:
```c++
std::vector<Forge::Parameter> Forge::extract_parameters(T& model)
```

just pass in the model to it

**example**
```c++

auto params {Forge::extract_parameters(model)};
Forge::Adam optimizer(params, /*...*/); //using the extracted parameters
```

`extract_parameters()` recursively traverses the model. gathers the parameters from every member that satisfies the
`HasParameters` concept. the outcome is a flat `std::vector<Parameter>` which can be passed directly to an optimizer.

therefore you do not need to collect the weights and biases of every layer

### saving the parameters

models can be saved directly in the safetensors format using:

```c++
 void save(T& data_members, const std::string& filename);
```
just pass in the aggregate containing the data members of the models (the primitives), and the safetensors file to save to.

**example**
```c++
Forge::save(model, "weights.safetensors");
```

Forge writes the standard safetensors structure: an eight‑byte header length, a JSON header that contains the tensor dtype, shape and byte offsets and then the actual tensor data.
parameter names are generated from the model’s structure. for primitives Forge continues to walk down the hierarchy creating
names such as `<primitive_name>.<instance_number>.<parameter_tag>`, for example `ln.0.w` for the gamma of first instance of LayerNorm.

### loading in the weights

loading functions in the same way by using:

```c++
void load(T& data_members, const std::string& filename);
```

**example**

```c++
Forge::load(model, "weights.safetensors");
```

Forge reads the checkpoint and matches the model's parameters internal names against loaded state dictionary and copies the
data into the existing parameters. it also verifies that the number of tensors in the checkpoint matches the number of parameters,
in the model, which helps catch architecture mismatches.

this is also how Forge loads GPT‑2 weights into its GPT‑2 implementation.

if you do not want to load weights into a model you can use:

```c++
std::map<std::string, Tensor> load_safetensors(const std::string& filename);
```

**example**
```cpp
auto loaded_state_dict {Forge::load_safetensors("weights.safetensors")};
```

this returns a `std::map<std::string, Tensor>` that contains the tensors from the checkpoint. it is handy when you only want to inspect a safetensors file without having a matching model structure.