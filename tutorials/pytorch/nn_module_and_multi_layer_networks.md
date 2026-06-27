# `nn.Module` and Multi-Layer Networks

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

**State** = parameters + buffers. `state_dict()` returns everything needed to fully reconstruct the model.

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

## Why You Need Nonlinearity

A single `nn.Linear` computes `y = xW^T + b` — purely linear. Stack two with nothing between:

$$W_2(W_1 x) = (W_2 W_1)x = W\_{\\text{combined}},x$$

It collapses to *one* linear layer. **Depth buys nothing without nonlinearity.**

**The fix:** activation functions between layers:

$$h = \\sigma(W_1 x + b_1), \\qquad y = W_2 h + b_2$$

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

## Shape Discipline

Every early bug will be a shape mismatch. Track dimensions explicitly:

```
x:    (B, in_dim)
W1:   (hidden, in_dim)   → fc1(x): (B, hidden)
ReLU: (B, hidden)        → shape unchanged (elementwise)
W2:   (out_dim, hidden)  → fc2:    (B, out_dim)
```

The batch dim `B` rides along untouched — layers operate on the last dim. Transformer tensors flow the same way, just with more dims: `(B, seq_len, d_model)`.
