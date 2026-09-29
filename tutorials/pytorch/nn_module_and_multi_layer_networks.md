# `nn.Module` and Multi-Layer Networks

**The mental model:** a module is a persistent tree of registered state and a callable description of data flow. With gradient recording enabled, each call creates a fresh autograd graph. Keeping those two lifetimes separate explains parameter registration, repeated calls, `state_dict()`, and why gradients accumulate.

**Reading route:** start with [parameters, buffers, and state](#parameters-buffers-and-state), then [what the graph saves](#the-module-doesnt-save-forward-results--the-graph-does). The sections on `Sequential`, hooks, and loss modules answer specific interface questions once that distinction is clear.

| At this point | What exists |
|---|---|
| After `__init__` | Registered parameters, buffers, and child modules |
| After `model(x)` | Those same objects plus this call's graph when gradients are recorded |
| After `loss.backward()` | Gradients accumulated on leaf parameters |
| After `optimizer.step()` | Updated parameter values for the next call |

______________________________________________________________________

## What `nn.Module` Is

`nn.Module` is the base class every layer and every model subclasses. You subclass it, fill `__init__` with layers, and write `forward()`. The backward pass is defined automatically by autograd.

**SWE analogy:** an abstract base class with inversion-of-control and lifecycle hooks — closest to a React `Component` or a Spring `@Component`. Like those, it:

- auto-discovers registered fields (parameters instead of props/beans)
- has mode switches: `train()` / `eval()` change behavior of Dropout and BatchNorm
- moves everything to device: `model.to("cuda")`
- serializes state: `state_dict()` / `load_state_dict()`

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()   # MUST call first
        # assign layers here

    def forward(self, x):
        # data flow — autograd builds the graph as this runs
```

______________________________________________________________________

## `__setattr__` Interception

`nn.Module` overrides Python's `__setattr__`. Every `self.x = y` in `__init__` is intercepted and classified by type:

- `nn.Parameter` → registered in `self._parameters`
- `nn.Module` (a sub-module) → registered in `self._modules`
- Anything else → stored as an ordinary attribute, **invisible to the framework**

The "magic": assigning `self.fc1 = nn.Linear(8, 16)` automatically registers that layer's weights and biases as parameters. No explicit registration call needed.

**The silent bug:**

```python
class Buggy(nn.Module):
    def __init__(self):
        super().__init__()
        self.good = nn.Parameter(torch.randn(3))        # registered ✓
        self.bad  = torch.randn(3, requires_grad=True)  # plain tensor ✗

print([n for n, _ in Buggy().named_parameters()])  # ['good']  ← 'bad' missing
```

If `bad` is a plain tensor:

1. **It never trains** — `model.parameters()` omits it, optimizer never updates it
1. **It stays on CPU** — `model.to("cuda")` won't move it → cryptic device mismatch crash in `forward()`
1. **It isn't saved** — `state_dict()` omits it, reload silently loses it

All three fail silently at definition time.

______________________________________________________________________

## Parameters, Buffers, and State

**Parameters** are tensors the model *learns*. Wrap in `nn.Parameter` (`requires_grad=True` by default). Sub-module parameters (e.g. `nn.Linear`'s `weight` and `bias`) are auto-registered when you assign the sub-module.

**Buffers** are persistent state the model *doesn't learn* — must be saved and moved to GPU, but not trained. Register with `self.register_buffer("name", tensor)`.

**Tensor state** = parameters + persistent buffers. `state_dict()` saves those tensors; the model class and its configuration must also be available to reconstruct the model.

```
state  (everything in state_dict — saved & device-moved)
├── parameters   → nn.Parameter, requires_grad=True, optimizer trains these
└── buffers      → register_buffer(), not trained but persistent
```

BatchNorm is the canonical example: `weight`/`bias` are parameters (learned), `running_mean`/`running_var` are buffers (updated during forward passes, not by the optimizer).

```python
bn = nn.BatchNorm1d(4)
print([n for n, _ in bn.named_parameters()])  # ['weight', 'bias']
print([n for n, _ in bn.named_buffers()])      # ['running_mean', 'running_var', ...]
print(list(bn.state_dict().keys()))            # all of the above
```

**Decision rule:**

- Should the optimizer learn it? → `nn.Parameter`
- Persistent, must save/move to GPU, but *not* learned? → `register_buffer`
- Throwaway intermediate in `forward`? → local variable, don't store on `self`

______________________________________________________________________

## A Module Is a Tree Node, Not a Layer

A layer is a Module, but so is the whole model. `nn.Linear`, `nn.Sequential`, and a ResNet are the same kind of object, and they nest through `_modules`. Each node owns its own `_parameters`, `_buffers`, and child `_modules`.

`.parameters()`, `.to(device)`, and `state_dict()` are just recursive walks over that tree. The registry is the whole point of the class.

______________________________________________________________________

## The Module Doesn't Save Forward Results — the Graph Does

Backward needs intermediate values from forward, such as the input to a matmul. A natural guess is that the module stores them. It doesn't. **Autograd attaches them to the output tensor.** Each op's output carries a `grad_fn`, and that backward node holds only what its gradient formula needs:

| Op | Backward node | Saves |
|---|---|---|
| `nn.Linear` | `AddmmBackward0` | both operands (`mat1` = input, `mat2` = weight) |
| `a @ b` | `MmBackward0` | both operands |
| `relu` | `ReluBackward0` | its output (`result`) |
| `a + b` | `AddBackward0` | no tensors |

```python
import torch, torch.nn as nn

lin = nn.Linear(3, 2)
x = torch.randn(1, 3)
y1, y2 = lin(x), lin(x)          # two independent graphs; lin recalls neither

print(y1.grad_fn)                                        # AddmmBackward0
print(y1.grad_fn is y2.grad_fn)                          # False
print(y1.grad_fn._saved_mat1.data_ptr() == x.data_ptr()) # True: graph holds the input

y1.sum().backward()
print(lin.weight.grad.shape)     # torch.Size([2, 3]): grads land on the parameter
```

Three consequences show that this state lives outside the module:

1. **Calling one module twice builds two independent graphs.** That's what makes RNN unrolling and weight tying work.
1. **`retain_graph=True` exists** because `backward()` frees the *graph's* saved tensors. A second `backward()` through the same graph raises `Trying to backward through the graph a second time`. The module is untouched.
1. **`del loss` frees activation memory while the model stays alive**, as long as nothing else holds the graph (a kept `logits` tensor does).

Some state does persist on the module: gradients. `backward()` accumulates into `param.grad`, which is why you call `optimizer.zero_grad()`.

**The split:**

```
Module  → persistent state: parameters, buffers, .grad   (optimizer reads this)
Graph   → per-call state:   saved activations            (backward() reads this)
```

This is why activation memory grows with batch size and depth, and parameter memory doesn't.

Sources: [nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html), [nn.Parameter](https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html), [Autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html).

______________________________________________________________________

## Why You Need Nonlinearity

A single `nn.Linear` computes `y = xW^T + b` for row-shaped batches — an affine map. Writing one example as a column vector, two layers without an activation give:

W₂(W₁x + b₁) + b₂ = (W₂W₁)x + (W₂b₁ + b₂)

It collapses to *one* linear layer. **Depth buys nothing without nonlinearity.**

**The fix:** activation functions between layers:

h = σ(W₁x + b₁), then y = W₂h + b₂.

The `σ` bends the space — the composition can't be flattened. Stacking nonlinear warps lets you carve arbitrarily complex decision boundaries.

______________________________________________________________________

## The Two Idioms

### `nn.Sequential` — straight-line data flow

```python
self.net = nn.Sequential(
    nn.Linear(in_dim, hidden),
    nn.ReLU(),
    nn.Linear(hidden, out_dim),
)
# forward: return self.net(x)
```

PyTorch calls each layer's output as the next layer's input. You never write `forward()` — Sequential owns it. Think of it as a Unix pipe: `input | layer1 | layer2 | output`.

### Explicit `forward()` — you control the data flow

```python
def forward(self, x):
    h = torch.relu(self.fc1(x))
    return self.fc2(h)
```

Identical in result to the Sequential above — but *you're writing the wiring*. This matters the moment your graph isn't a straight line.

### Why Transformers Can't Use Sequential

The residual connection `x + sublayer(x)` requires **two references to `x`** — the original, and the transformed version. Sequential has already thrown `x` away by the time `sublayer(x)` is done.

```python
def forward(self, x):
    return x + self.attention(x)  # Sequential cannot express this
```

Sequential only knows: *"pass output of step N as input to step N+1."* It can't say *"also remember step N's input and add it back later."*

### Rule of thumb

- Toy MLP, simple classifier → `Sequential` is fine
- Skip connections, multiple inputs/outputs, branching → explicit `forward()`
- Transformers → always explicit `forward()`

Key insight: **the training loop is identical in both cases**. The idiom only affects model definition.

______________________________________________________________________

### Side-by-side

**Shared setup:**

```python
import torch
import torch.nn as nn

X = torch.randn(100, 8)
y = torch.randint(0, 2, (100,)).float()
criterion = nn.BCEWithLogitsLoss()
```

**Idiom A — `nn.Sequential`:**

```python
model = nn.Sequential(
    nn.Linear(8, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(100):
    logits = model(X).squeeze()
    loss = criterion(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**Idiom B — Explicit `forward()`:**

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

model = MLP()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(100):
    logits = model(X).squeeze()
    loss = criterion(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**What's actually the same:**

| | Sequential | Explicit |
|---|---|---|
| `model(X)` call | ✅ | ✅ |
| `loss.backward()` | ✅ | ✅ |
| `optimizer.step()` | ✅ | ✅ |
| `model.parameters()` | ✅ | ✅ |

**The only real difference:** in Idiom B you can write skip connections:

```python
def forward(self, x):
    h = torch.relu(self.fc1(x))
    return x[:, :1] + self.fc2(h)  # impossible in Sequential
```

______________________________________________________________________

## `model(x)` vs `model.forward(x)` — Never Call `forward` Directly

`self.net(x)` goes through `__call__`, which wraps `forward` with hook machinery and other module call behavior:

```python
# Pseudo-internals of nn.Module.__call__
def __call__(self, *args):
    for hook in self._forward_pre_hooks: args = hook(self, args) or args
    out = self.forward(*args)
    for hook in self._forward_hooks: out = hook(self, args, out) or out
    return out
```

Calling `model.forward(x)` directly bypasses module hooks. For example, a forward hook registered for shape inspection will fire for `model(x)` but not for `model.forward(x)`.

```python
with torch.no_grad():
    out = model(x)         # no graph built
    direct = model.forward(x) # also no graph: no_grad is a thread-local context
```

`no_grad` is a context setting, so it applies to either call. The practical reason to use `model(x)` is that it respects module call machinery, including hooks and framework wrappers that depend on `__call__`.

**One line:** always use `model(x)`, never `model.forward(x)`.

______________________________________________________________________

## Why Loss Functions Are `nn.Module`

**Short answer:** consistency and composability.

```python
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(pred, yb)   # calls loss_fn.forward(pred, yb) under the hood
```

Some loss functions have **persistent state**, such as fixed class weights:

```python
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0]))
loss_fn.to(device)   # moves the weight tensor to GPU — works because it's a Module
```

A plain function can't do `.to(device)`. And because it's a Module, you can **embed it inside another model:**

```python
class ModelWithLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(8, 2)             # two class logits
        self.criterion = nn.CrossEntropyLoss()  # nested — fully tracked

    def forward(self, x, y):
        return self.criterion(self.net(x), y)
```

This pattern is common in HuggingFace and multi-GPU code (see [training_loop.md](../training/training_loop.md#why-embed-loss-inside-the-model)).

**One line:** `nn.Module` is PyTorch's universal interface for "anything with a forward pass" — loss functions qualify, so they inherit `.to(device)`, composability, and hook support for free.

______________________________________________________________________

## Shape Discipline

Every early bug will be a shape mismatch. Track dimensions explicitly:

```
x:    (B, in_dim)
W1:   (hidden, in_dim)   → fc1(x): (B, hidden)
ReLU: (B, hidden)        → shape unchanged (elementwise)
W2:   (out_dim, hidden)  → fc2:    (B, out_dim)
```

The batch dim `B` rides along untouched — layers operate on the last dim. Transformer tensors flow the same way, just with more dims: `(B, seq_len, d_model)`.

## Common confusions

- A plain tensor with `requires_grad=True` can receive a gradient but is not automatically a registered parameter. An optimizer built from `model.parameters()` will miss it.
- A buffer moves and saves with the module, but the optimizer does not learn it. BatchNorm running statistics are the standard example.
- `model.eval()` changes mode-sensitive modules such as Dropout and BatchNorm; it does not itself turn off autograd. Use `torch.no_grad()` or inference mode when gradients are unnecessary.

## Check your understanding

1. Why can one `nn.Linear` be called twice and still produce two independent forward graphs?
1. What happens if `self.weight` is a plain `torch.Tensor(requires_grad=True)` rather than `nn.Parameter`?
1. Why do two linear layers with no activation collapse into one affine transformation?

<details markdown="1"><summary>Answers</summary>

1. The module stores weights, while each call's output graph stores that call's operation history and saved values.
1. It is not registered in `model.parameters()` or `state_dict()`, and `model.to(device)` will not move it.
1. Composing affine maps gives another affine map: weights multiply and biases combine.

</details>

**One-minute recap:** modules own persistent registered state; graph nodes hold per-call backward information; nonlinear activations make stacked layers more expressive than one affine layer.
