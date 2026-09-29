# Gradients and Gradient Descent

**The question:** how can a single loss tell each weight which way to move? Every forward operation records a local connection. Backward follows those connections in reverse, multiplying local derivatives and adding contributions wherever paths meet.

For a quick revision, read the [chain rule trace](#chain-rule-trace) first; then use the sections on leaves and differentiability to explain where gradients appear and when they stop carrying useful information.

______________________________________________________________________

## The Computation Graph

When you create a tensor with `requires_grad=True`, PyTorch tracks all operations on it. Each resulting tensor gets a `grad_fn` attribute referencing the operation that created it. These link together into an acyclic graph encoding the full computation history.

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2        # y.grad_fn = PowBackward0
y.backward()      # walk the graph backward; fill .grad on leaves
x.grad            # → tensor([4.])
```

Calling `.backward()` walks the graph in reverse and fills each **leaf** tensor's `.grad`.

______________________________________________________________________

## Why `.grad` Lands on the Leaf, Not the Output

`.backward()` computes the derivative *of the output* with respect to the *leaves*. The output is the numerator; the leaves are the denominators. The result gets stored on the **leaf** — the thing you differentiate against.

In real training: you call `loss.backward()`. `loss` is the output (numerator). Model parameters are the leaves (denominators). Those parameters need to know how to change — so that's where the gradient lands.

```python
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)
y = a * b
y.backward()
print(a.grad)   # 3.0  — dy/da = b
print(b.grad)   # 2.0  — dy/db = a
# y.grad is None — output nodes don't store gradients by default
```

______________________________________________________________________

## Why Centered Inputs Stabilize Gradients

Take one neuron: `z = w·x + b`. The gradient of the loss with respect to weight `w` is:

```
∂L/∂w = ∂L/∂z · x
                 ↑
         the input value itself appears here
```

**The input directly scales the gradient.**

**Intuition:** for one example with positive inputs, every incoming weight gradient has the sign of `∂L/∂z`. Across examples, the sum can still have different signs, but strongly correlated input features can make weight updates poorly conditioned and cause a zig-zag path:

```
Uncentered inputs (all positive):     Centered inputs (mix of ±):
weights forced to move together        weights move independently
→ zig-zag path to minimum              → direct path

    \  /\                                   \
     \/  \                                   \
          \                                   \
           → slow                              → fast
```

Centering inputs around 0 allows positive and negative contributions to each weight gradient. It often reduces correlations and makes descent better conditioned; it does not guarantee a straight path or independent updates.

**Second effect — scale:** if `x` is huge, `∂L/∂w` is huge → giant weight jumps → instability. Dividing by std keeps gradient magnitudes in a sane range. This is the same reasoning behind BatchNorm. (LeCun et al., *Efficient BackProp*, 1998.)

______________________________________________________________________

## Chain Rule Trace

Autograd is the chain rule, automated. For `L = f(g(h(w)))`:

∂L/∂w = (∂L/∂f) · (∂f/∂g) · (∂g/∂h) · (∂h/∂w)

Each factor is **local** — it only depends on one operation. Autograd computes each independently, then multiplies them walking backward. The loss starts the relay by handing back `1.0` (since `∂L/∂L = 1`).

**Concrete trace** — `L = (w·x − y)²` with `x=3, y=10, w=2`:

```
Forward:
a = w * x      = 6
b = a - y      = -4
L = b ** 2     = 16

Backward:
∂L/∂L = 1
∂L/∂b = 2b          = -8
∂L/∂a = ∂L/∂b · 1   = -8
∂L/∂w = ∂L/∂a · x   = -24
```

`w.grad` → `-24`. Swap in cross-entropy and only the final node's local rule changes.

______________________________________________________________________

## Leaf vs Non-Leaf, `retain_grad`, and `no_grad`

**Leaf node:** a tensor you created directly, not produced by an operation. Model parameters are leaves — they sit at the edges of the computation graph.

**Non-leaf node:** any tensor that is the *result* of an operation. Intermediate activations and the loss itself are non-leaf.

```python
w = torch.tensor([1.0], requires_grad=True)  # leaf
x = torch.tensor([2.0])                       # leaf (requires_grad=False)
z = w * x                                      # non-leaf — result of multiply
loss = z.sum()                                 # non-leaf
```

`loss.backward()` walks the graph backward and **accumulates into `.grad` of leaf tensors only:**

```python
loss.backward()
print(w.grad)   # populated  — w is a leaf
print(z.grad)   # None       — z is non-leaf, grad discarded by default
```

Non-leaf grads are computed transiently during backprop (to propagate the chain rule) then discarded — memory savings. Add `z.retain_grad()` before `backward()` if you actually need a non-leaf's grad.

**`torch.no_grad()`:** disables graph recording inside its context. Results have `requires_grad=False`:

```python
with torch.no_grad():
    z = w * x
    print(z.requires_grad)   # False — no graph built
```

Two reasons to use at eval time: (1) no graph = less memory, (2) no graph-building overhead = faster. You're not training, so you never need the backward pass.

______________________________________________________________________

## Why Any Differentiable Loss Works

The loss isn't special to autograd — it's just the **final node** in the graph, the one that outputs a scalar. Autograd walks backward from any scalar through whatever operations produced it.

The real constraint: every operation from input to loss must have a known local derivative. PyTorch ships a derivative rule for each primitive op (`+`, `*`, `matmul`, `relu`, `exp`, `log`, ...). As long as your loss is built out of these, autograd handles it.

______________________________________________________________________

## Differentiability — Three Cases

### Case 1: Differentiable everywhere

`x²`, `exp`, `sin`, `matmul` — smooth, single-valued derivative at every point. No issue.

### Case 2: Differentiable almost everywhere (kinks)

`ReLU`, `abs` — undefined at exactly one point.

| Input | ReLU derivative |
|---|---:|
| x > 0 | 1 |
| x < 0 | 0 |
| x = 0 | undefined mathematically; PyTorch chooses 0 |

PyTorch resolves the tie by convention (returns `0` for `relu'(0)`). This is a **subgradient** — legitimate for convex kinks. In practice it never matters since you almost never land exactly on `x=0` in floating point.

```python
x = torch.tensor([0.0], requires_grad=True)
torch.relu(x).backward()
x.grad   # tensor([0.])  ← chosen subgradient
```

### Case 3: Zero-gradient / non-differentiable — breaks training

`argmax`, `round`, `floor`, sampling.

These do not provide a useful path for ordinary backpropagation. `floor` and `round` have zero derivatives almost everywhere in autograd. `argmax` returns integer indices with no gradient path, so trying to backpropagate through the indices raises rather than silently returning zeros. Sampling a discrete choice similarly needs a special estimator if its choice affects the loss.

```python
x = torch.tensor([2.7], requires_grad=True)
y = torch.floor(x)
y.backward()
x.grad   # tensor([0.])  ← defined but useless
```

**Why this matters for Transformers:** picking a token with `argmax(logits)` does not carry a gradient to the logits. Cross-entropy instead operates on the full logit vector, so gradient flows back into the scores. Token selection is usually deferred to generation or inference.

When you need to backprop through a discrete choice: **straight-through estimator** (pretend the non-differentiable op was identity on the backward pass) and **Gumbel-softmax** (smooth, temperature-controlled approximation) exist for this.

## Common confusions

- A gradient is a **slope at the current values**, not an update. The optimizer uses it to choose an update.
- Gradients **accumulate** in leaf `.grad` fields. A second backward pass adds to an existing gradient unless it is cleared.
- `torch.no_grad()` stops recording new operations; it does not change the `requires_grad` setting of an existing parameter.

## Check your understanding

1. In `L = (w·x − y)²` with `w=2`, `x=3`, `y=10`, why is `∂L/∂w = −24`?
1. After `loss.backward()`, which tensor normally stores a parameter's derivative?
1. Why can cross-entropy train a classifier while `argmax(logits)` cannot carry its training signal?

<details markdown="1"><summary>Answers</summary>

1. The error is `−4`; squaring contributes `2(−4)=−8`, and multiplying by `x=3` gives `−24`.
1. The leaf parameter's `.grad` field.
1. Cross-entropy uses the continuous scores; `argmax` produces a discrete index with no gradient path.

</details>

**One-minute recap:** a forward pass builds a graph; backward applies local chain rules; leaf gradients accumulate; the optimizer makes the actual update.
