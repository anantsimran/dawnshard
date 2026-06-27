# Inspecting Your Model — Best Practices

Most PyTorch bugs are silent shape/dtype/device mismatches that either crash 50 layers later or — worse — *broadcast successfully into wrong results*.

______________________________________________________________________

## Shape Checking

**Assert shapes explicitly.** Broadcasting will happily turn a `(32,1)` vs `(32,)` mismatch into a `(32,32)` tensor with no error.

```python
def assert_shape(t: torch.Tensor, expected: tuple, name: str = "tensor") -> torch.Tensor:
    assert t.dim() == len(expected), \
        f"{name}: rank {t.dim()} != {len(expected)} {tuple(t.shape)}"
    for i, (a, e) in enumerate(zip(t.shape, expected)):
        assert e == -1 or a == e, \
            f"{name}: dim {i} is {a}, expected {e} {tuple(t.shape)}"
    return t   # returns tensor so calls can be chained
```

Usage: `assert_shape(logits, (batch, -1, vocab), "logits")`. Use `-1` as a wildcard dim.

**Name dimensions in comments** on every line that reshapes: `# x: (B, T, C)`. Reshape/permute bugs are the #1 time sink.

**Write a shape smoke-test before training.** Run one forward+backward on a random batch — catches 90% of wiring bugs in under a second:

```python
x = torch.randn(4, 8)
out = model(x)
assert_shape(out, (4, 1), "out")
loss = criterion(out.squeeze(), torch.zeros(4))
loss.backward()
assert all(p.grad is not None for p in model.parameters())
```

**Verify parameter registration** after building a model to catch unregistered tensors early:

```python
def assert_param_names(model: nn.Module, expected: set[str]) -> None:
    actual = {name for name, _ in model.named_parameters()}
    assert actual == expected, \
        f"param mismatch: missing={expected - actual}, extra={actual - expected}"
```

______________________________________________________________________

## `jaxtyping` + `beartype` — Runtime Shape Contracts

Shape annotations as type hints, enforced at runtime. `jaxtyping` provides the vocabulary; `beartype` enforces it. Together they give typed contracts: wrong shape in → immediate, readable exception at the call site.

```bash
pip install jaxtyping beartype
```

```python
from jaxtyping import Float, jaxtyped
from beartype import beartype
from torch import Tensor

@jaxtyped(typechecker=beartype)
def linear(x: Float[Tensor, "batch in_dim"],
           w: Float[Tensor, "in_dim out_dim"]) -> Float[Tensor, "batch out_dim"]:
    return x @ w

x = torch.randn(32, 128)
w = torch.randn(128, 10)
linear(x, w)           # ✓

bad = torch.randn(64, 99)
linear(bad, w)         # ✗ raises immediately: in_dim is 99 here but 128 in w
```

The shared name `in_dim` means `x.shape[1]` must equal `w.shape[0]`. Same-name enforcement catches cross-argument relationship bugs that a standalone `assert` can't express cleanly.

Apply to `forward()` during development:

```python
class TinyNet(nn.Module):
    @jaxtyped(typechecker=beartype)
    def forward(self, x: Float[Tensor, "batch 2"]) -> Float[Tensor, "batch 1"]:
        ...
```

______________________________________________________________________

## Graph Visualization

### Option 1 — `torchviz`: the actual backward graph

```bash
pip install torchviz   # also: brew install graphviz / apt-get install graphviz
```

```python
from torchviz import make_dot
y = model(torch.randn(1, 2))
dot = make_dot(y, params=dict(model.named_parameters()))
dot.render("graph", format="png")
```

Use `show_attrs=True, show_saved=True` to see what autograd saves for the backward pass. Those saved tensors drive GPU memory during training — useful when debugging OOMs.

### Option 2 — `torchview`: module-level view with shapes

Better when you want to see your architecture rather than raw `grad_fn` nodes. `device='meta'` traces shapes with zero memory allocation.

```python
from torchview import draw_graph
g = draw_graph(model, input_size=(1, 2), device='meta')
g.visual_graph   # renders inline in a notebook
```

Both options need the `graphviz` system binary on your PATH — that's the #1 reason these silently error.

### Option 3 — zero-dependency: walk `grad_fn` yourself

```python
def print_graph(t, depth=0, seen=None):
    fn = getattr(t, "grad_fn", t)
    if fn is None:
        return
    seen = set() if seen is None else seen
    if id(fn) in seen:
        return
    seen.add(id(fn))
    print("  " * depth + type(fn).__name__)
    for next_fn, _ in getattr(fn, "next_functions", ()):
        if next_fn is not None:
            print_graph(next_fn, depth + 1, seen)
```

```python
y = model(torch.randn(1, 2))
print_graph(y)
# AddmmBackward0
#   AccumulateGrad        ← leaf parameter (.grad lands here)
#   ReluBackward0
#     AddmmBackward0
#       ...
```

`AccumulateGrad` nodes are leaf parameters (where `.grad` lands). `*Backward0` nodes are operations. This is the same structure `torchviz` renders, just as text.

**Recommendation:** option 3 to understand graph structure, `torchview` for day-to-day architecture + shape debugging, `torchviz` only when you need backward/saved-tensor detail.
