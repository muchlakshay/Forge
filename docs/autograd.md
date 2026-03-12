# Autograd Engine

The autograd subsystem is Forge's implementation of reverse-mode automatic differentiation — the algorithm that makes neural network training possible. It lives in `core/autograd/` and consists of three files:

- `node_abstract.h` — The abstract interface for computation graph nodes
- `node.h` — The concrete, templated node implementation
- `attach_node.h` — Utility for wiring nodes into the computation graph

This is the piece that turns Forge from a tensor math library into something that can train models.

---

## How Automatic Differentiation Works (Brief Background)

If you're already familiar with autograd, skip this section. If not, here's the core idea.

When you compute `y = f(g(x))`, the derivative `dy/dx` is `f'(g(x)) * g'(x)` — the chain rule. In a neural network, you have hundreds of these composed functions (layers), and you need gradients with respect to all the parameters.

**Forward-mode AD** computes derivatives alongside the forward computation. It's efficient when you have few inputs and many outputs. Neural networks are the opposite — many parameters (inputs) and one loss value (output).

**Reverse-mode AD** (backpropagation) records the computation in a graph during the forward pass, then walks the graph backward to compute all gradients in one pass. It's efficient when you have many inputs and few outputs — exactly the neural network case.

Forge implements reverse-mode AD.

---

## The Computation Graph

During the forward pass, every operation that involves gradient-tracked tensors creates a node in the computation graph. The node records:
- What operation was performed
- Which tensors were inputs (parents)
- Which tensor is the output (child)
- How to compute the gradient for this specific operation

The graph is a DAG (directed acyclic graph) — each node points to its parents, and the edges represent data flow. When you call `backward()` on the output tensor, it walks this graph in reverse, computing gradients at each node.

```
Forward:   x ──► [matmul] ──► y ──► [relu] ──► z ──► [loss] ──► L

Backward:  x ◄── [matmul_bwd] ◄── y ◄── [relu_bwd] ◄── z ◄── [loss_bwd] ◄── L
           (dx)                    (dy)                  (dz)                  (dL=1)
```

---

## Node Abstraction

The abstract base class is deliberately minimal:

```cpp
class NodeAbstract {
public:
    virtual void backward() = 0;
    virtual ~NodeAbstract() = default;
};
```

Just one method: `backward()`. That's all the tensor needs to know about its computation graph node — "compute gradients and propagate them to parents."

I kept this minimal for the same reason the storage abstraction is minimal: the tensor shouldn't know the details. It holds a `shared_ptr<NodeAbstract>` and calls `backward()` when needed. The concrete node type handles everything else.

---

## The Concrete Node

This is where the interesting design lives. The `Node` class is a template with three parameters:

```cpp
template <typename OpClass, std::size_t Parents, ValidEnum OpEnum>
class Node final : public NodeAbstract
```

### Template Parameters

**`OpClass`**: The gradient computation class. This is a functor (like `MatMulBackward`, `ReLUBackward`, etc.) that knows how to compute gradients for a specific operation. It must have a `compute_grads()` method.

**`Parents`**: The number of parent tensors, known at compile time. A binary operation like addition has `Parents = 2`. A unary operation like ReLU has `Parents = 1`. This is a compile-time constant because it determines the size of the parent array — no dynamic allocation for storing parent references.

**`OpEnum`**: The operation enum type (must satisfy the `ValidEnum` concept, meaning it has a `::Count` member). This is used for looking up the gradient kernel in the dispatcher.

### Why Template Everything?

I could have made the node store a `std::vector<Tensor>` for parents and a `std::function` for the gradient computation. That would be simpler. But:

1. **Fixed-size parent array**: `std::array<Tensor, Parents>` means no heap allocation for parent storage. For operations with 1-3 parents (which is nearly all of them), this avoids a vector allocation per node.

2. **Type-safe gradient function**: The `OpClass` parameter means the compiler knows the exact gradient function type. This enables inlining and eliminates `std::function` overhead (which typically involves a heap allocation and virtual dispatch).

3. **Compile-time validation**: If you try to create a node with the wrong number of parents for an operation, it fails at compile time, not runtime.

The tradeoff is more complex template signatures, but the generated code is tighter.

### Internal State

```cpp
std::array<Tensor, Parents> m_parents;     // Input tensors
Tensor m_child;                            // Output tensor
DispatchKey m_dispatch_key;                // For kernel lookup
const Dispatcher<OpEnum>& m_dispatcher;    // Reference to dispatcher
OpEnum m_op;                               // Which operation this is
Kernel* m_gradFn;                          // Pointer to gradient kernel
```

The node stores everything needed for the backward pass:
- The parent tensors (to compute gradients with respect to them)
- The child tensor (to access the output gradient, dL/d(child))
- A reference to the dispatcher and the operation enum (to look up the gradient kernel)
- A direct pointer to the gradient kernel (cached at construction time for fast access)

The gradient kernel pointer (`m_gradFn`) is resolved once during node construction via the dispatcher. This avoids repeated dispatcher lookups during the backward pass.

### The Backward Method

```cpp
void backward() override {
    backward_impl(std::make_index_sequence<Parents>{});
}

template <std::size_t... I>
void backward_impl(std::index_sequence<I...>) {
    static_cast<OpClass*>(m_gradFn)->compute_grads(m_child, m_parents[I]...);
    ((m_parents[I].node() ? m_parents[I].node()->backward() : void()), ...);
}
```

This is compact but there's a lot happening:

1. **Index sequence trick**: `std::make_index_sequence<Parents>{}` generates a compile-time sequence `{0, 1, ..., Parents-1}`. This is used to unpack the parent array as individual arguments to `compute_grads()`.

2. **Gradient computation**: `static_cast<OpClass*>(m_gradFn)->compute_grads(m_child, m_parents[I]...)` calls the gradient function, passing the child tensor and all parent tensors. The static cast is safe because the kernel was registered with the correct type.

3. **Recursive propagation**: The fold expression `((m_parents[I].node() ? m_parents[I].node()->backward() : void()), ...)` calls `backward()` on each parent's node (if it has one). This is how the chain rule propagates through the graph.

The `void()` cast in the ternary is a C++ trick to make both branches have the same type for the fold expression to work.

### Why Not Use std::function?

`std::function<void(Tensor&, Tensor&...)>` would be the "obvious" C++ choice for storing a callback. But:
- `std::function` typically heap-allocates for non-trivial callables
- It adds virtual dispatch overhead
- It erases the type, so the compiler can't inline

By storing the gradient function as a `Kernel*` and static-casting to the known `OpClass`, I get direct function calls with potential inlining, and zero heap allocation for the function object itself.

---

## Attaching Nodes to Tensors

The `attach_node` function is the bridge between the forward pass and the computation graph:

```cpp
template <typename OpClass, std::size_t Parents, ValidEnum OpEnum, typename... ParentTensors>
void attach_node(Tensor& tensor, Device device, const Dispatcher<OpEnum>& dispatcher,
                 OpEnum op, ParentTensors&&... parent_tensors)
```

When an operation creates an output tensor, it calls `attach_node` to:
1. Create a new `Node` object with the operation class, parents, and dispatcher
2. Assign it to the output tensor's `m_node` field

This is called during the forward pass, only for operations involving gradient-tracked tensors. If none of the inputs require gradients, no node is created — the forward pass runs without any overhead.

The function takes parent tensors as a parameter pack (`ParentTensors&&...`), which allows it to handle operations with any number of inputs. The `Parents` template parameter ensures the count matches at compile time.

---

## The Gradient Flow

Here's the full lifecycle:

### 1. Forward Pass (Graph Building)
```
x (requires_grad=true) ──► operation ──► y
                                │
                                └── attach_node(y, ..., x)
                                    Creates Node<OpBackward, 1, ...>
                                    y.m_node = new_node
```

### 2. Loss Computation
```
y ──► loss_fn ──► loss (scalar)
                    │
                    └── loss.m_grads = 1.0  (seed gradient)
```

### 3. Backward Pass (Gradient Propagation)
```
loss.backward()
  └── loss.m_node->backward()
        └── OpBackward::compute_grads(loss, y)
              └── y.m_grads = dL/dy
        └── y.m_node->backward()
              └── OpBackward::compute_grads(y, x)
                    └── x.m_grads = dL/dx
```

Each `backward()` call:
1. Computes the local gradient (dL/d(parent) given dL/d(child))
2. Stores it in the parent's gradient tensor
3. Recursively calls backward on the parent's node

---

## Design Decisions Summary

| Decision | Rationale |
|----------|-----------|
| Minimal `NodeAbstract` interface | Tensor only needs `backward()` — everything else is internal to the node |
| Compile-time parent count | Fixed-size array avoids heap allocation per node |
| Template-based gradient function | Eliminates `std::function` overhead, enables inlining |
| Index sequence for parameter unpacking | Turns array storage into function arguments at compile time |
| Cached gradient kernel pointer | Avoids repeated dispatcher lookups during backward pass |
| `attach_node` as free function | Keeps node creation logic separate from both tensor and operation code |
| Recursive backward propagation | Natural implementation of the chain rule on a DAG |

---

## Current Limitations

The autograd engine has the infrastructure in place but is missing the actual gradient kernels:

- **No math operation gradients**: Since math operations aren't implemented yet, there are no `MatMulBackward`, `AddBackward`, etc. classes. The `Node` template is ready to accept them — it's just waiting for the operations.
- **No gradient accumulation**: Currently gradients are overwritten. For operations where a tensor is used multiple times, gradients need to be accumulated (summed). This will need to be added.
- **No graph cleanup**: After backward, the computation graph nodes remain in memory. A mechanism to clear or detach the graph after use would reduce memory pressure during training.
- **Single-threaded**: The backward pass is sequential. For large models with independent subgraphs, parallel backward computation could help.

These are all planned improvements — the core architecture supports them without structural changes.

---

## File Reference

| File | Lines | Purpose |
|------|-------|---------|
| `node_abstract.h` | ~11 | Pure virtual node interface with `backward()` |
| `node.h` | ~38 | Templated node with compile-time parent count and typed gradient function |
| `attach_node.h` | ~17 | Free function to create and attach nodes during forward pass |
