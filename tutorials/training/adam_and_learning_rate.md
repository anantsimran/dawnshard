# Adam and the Learning Rate

**Revision question:** Why can a learning rate that works for SGD fail for Adam, and why does Adam still need a learning-rate schedule?

Think of the gradient as a **local direction**, the learning rate as the **overall step scale**, and Adam's moments as a **per-coordinate adjustment** based on recent gradients. Adam changes how the gradient becomes an update; it does not choose a safe global scale for every model or training phase.

**Reading map:** [The one-dimensional loss](#the-1-d-picture) makes large and small steps visible. [Momentum](#momentum-remembering-the-direction) and [Adam's update](#adam-line-by-line) explain how history changes an update. [AdamW](#adam-vs-adamw-where-weight-decay-goes), [learning-rate selection](#picking-a-learning-rate), and [warmup and decay](#warmup-and-decay) are reference sections for diagnosing a run. See [optimizer and scheduler](optimizer_and_scheduler.md) for where these values live in PyTorch.

Every output shown below was produced by running the code with `torch==2.12.1`.

______________________________________________________________________

## What the Learning Rate Controls

The update rule for plain gradient descent is one line:

```
w ← w − lr · ∂L/∂w
```

The gradient says *which way* is downhill. `lr` says *how far* to walk. The gradient is only valid locally, like a slope measured under your feet, so a step that's too long lands somewhere the slope said nothing about.

### The 1-D picture

Take the simplest loss with a single minimum, a parabola: `L(w) = ½·a·w²`. Its gradient is `a·w`, so one step is:

```
w ← w − lr·a·w  =  (1 − lr·a) · w
```

Every step multiplies `w` by the same factor `(1 − lr·a)`. The whole story is in that factor. With curvature `a = 4`, starting from `w = 1`:

| `lr` | factor `1 − 4·lr` | first 4 steps | What happens |
|---|---|---|---|
| 0.1 | 0.6 | 0.6, 0.36, 0.216, 0.130 | smooth convergence |
| 0.4 | −0.6 | −0.6, 0.36, −0.216, 0.130 | overshoots, flips sign, still converges |
| 0.6 | −1.4 | −1.4, 1.96, −2.744, 3.842 | overshoots further each time: **diverges** |

So there is a hard ceiling: **gradient descent on this parabola converges when `0 < lr < 2/a`**. Steeper curvature means a lower ceiling.

### Why one global `lr` is a compromise

A real network has millions of directions, each with its own curvature. The **steepest** direction sets the ceiling (`lr < 2/a_max`), and the **flattest** direction then crawls, because its factor `1 − lr·a_min` is barely below 1. The ratio `a_max / a_min` (the *condition number*) is how badly one shared `lr` fits everyone.

That mismatch is the problem both momentum and Adam attack.

### Symptoms

| What the loss curve does | Likely cause |
|---|---|
| `nan` or `inf` within the first few hundred steps | `lr` may be too high; also check invalid operations and masking |
| Drops, then jumps up and oscillates at a high value | `lr` too high |
| Drops fast, then plateaus above where a smaller `lr` ends up | `lr` too high to settle; decay it (see [Schedules](#warmup-and-decay)) |
| Decreases in a slow, straight, boring line | `lr` too low |
| Train loss fine, val loss rising | usually overfitting, not the learning rate |

______________________________________________________________________

## Momentum: Remembering the Direction

SGD with momentum keeps a running sum of past gradients and steps along that instead of the raw gradient. PyTorch's form:

```
buf ← μ·buf + grad
w   ← w − lr · buf
```

- In a direction where the gradient keeps the same sign, `buf` grows to `grad / (1 − μ)`. With `μ = 0.9` that's a **10× longer step**, for free.
- In a direction where the gradient flips sign every step (the oscillating, steep direction), the terms cancel and `buf` stays small.

In this constant-gradient picture, momentum speeds up the persistent direction and cancels alternating directions. The steady-state buffer scales as `1/(1 − μ)`: changing `momentum` from 0.9 to 0.99 makes that scale 10× larger. Real gradients change over time, but the `lr` that worked before may no longer be safe.

______________________________________________________________________

## Adam, Line by Line

Adam (Kingma & Ba, 2015) keeps two running averages **per parameter** and uses them to give each parameter its own step size. PyTorch defaults: `lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0`.

```
t ← t + 1
m ← β1·m + (1 − β1)·grad          # 1st moment: smoothed direction
v ← β2·v + (1 − β2)·grad²         # 2nd moment: smoothed squared magnitude
m̂ ← m / (1 − β1^t)                # bias correction
v̂ ← v / (1 − β2^t)                # bias correction
w ← w − lr · m̂ / (√v̂ + ε)
```

These live in `optimizer.state[param]` under the keys `step`, `exp_avg` (`m`), and `exp_avg_sq` (`v`).

### The key ratio: `m̂ / √v̂`

Read the last line as `w ← w − lr · (a normalized direction)`. That direction is often near `−1` to `1`, but it is not strictly bounded there.

- **Consistent gradient** (the same value `g` every step): `m̂ → g` and `√v̂ → |g|`, so the ratio is `±1`. The parameter moves by exactly `lr`.
- **Alternating gradients** (the sign keeps flipping, mean ≈ 0): `m̂` tends toward 0 but `√v̂` stays at the typical magnitude, so the systematic update is small. Individual noisy steps can still move the parameter.

**Follow two gradients:** start with zero moments and `lr=0.01`. A first gradient of `+2` gives bias-corrected `m̂=2`, `√v̂=2`, and an update of about `−0.01`. If the next gradient is `−2`, the first moment nearly cancels (`m̂≈−0.105`) while `√v̂` remains `2`; the update is only about `+0.00053`. Adam remembers the disagreement between those directions.

So **Adam's step is roughly `lr` times the gradient's persistent direction relative to its recent magnitude.** That has two consequences you'll use constantly.

**1. `lr` scales a normalized update.** With SGD, the step is `lr · grad`, so the right `lr` depends strongly on gradient magnitude. With Adam, a consistent gradient often gives a coordinate update near `lr`, regardless of that gradient's size. Transient updates can exceed `lr`, so it is a scale, not a cap.

**2. Adam is approximately insensitive to scaling the loss** when `ε` is negligible, weight decay is absent, and numeric precision is adequate. Multiply the loss by 100: `grad`, `m`, and `√v` all scale by 100, and the ratio stays nearly unchanged. On a small linear regression, 50 steps:

| | loss × 1 | loss × 100 |
|---|---|---|
| Adam, `lr=1e-2` | final weights match to 4e-8 | ← same |
| SGD, `lr=1e-2` | converging (first weights `≈ [0.42, −1.07]`) | diverged (`≈ 3.8e8`) |

A sudden large gradient after a long run of tiny ones illustrates why `lr` is not a hard cap: the ratio can reach about `(1 − β1)/√(1 − β2)`, or `3.2` with default betas.

### Why bias correction exists

`m` and `v` start at zero, so early on they are averages of "a few real gradients plus a lot of zeros." Without correction, at step 1:

```
m = 0.1 · g          v = 0.001 · g²
m / √v = 0.1·g / (0.0316·|g|) ≈ 3.16 · sign(g)
```

The very first step would be **3.16× `lr`** on every parameter, exactly when the weights are random and the gradients least trustworthy. Dividing by `1 − β^t` undoes the zero-initialization: at `t = 1`, `m̂ = g` and `v̂ = g²`, so the first step is exactly `lr` in each coordinate. In PyTorch:

```python
w = torch.nn.Parameter(torch.tensor([1.0, -2.0, 3.0]))
opt = torch.optim.Adam(params=[w], lr=0.01)
w.grad = torch.tensor([0.5, -300.0, 1e-3])   # wildly different magnitudes
opt.step()
# every coordinate moves approximately 0.01, against the sign of its gradient
```

The correction fades as `β^t → 0`, but slowly for `β2 = 0.999`: `1 − 0.999^1000 ≈ 0.63`, so it's still doing real work after a thousand steps.

### `ε`

`ε` stops the division from blowing up when `√v̂ ≈ 0`. For a parameter whose gradients are much smaller than `1e-8`, `ε` dominates the denominator and Adam turns into plain momentum-SGD with step `lr · m̂ / ε`. The default is fine for almost everything. The original Transformer used `1e-9`.

### The betas

| Knob | Default | Averaging window ≈ `1/(1−β)` | Raise it when | Lower it when |
|---|---|---|---|---|
| `β1` | 0.9 | ~10 steps | gradients are very noisy (small batches) | you need fast reaction to changing gradients |
| `β2` | 0.999 | ~1000 steps | rarely | training is unstable with loss spikes; the original Transformer used 0.98 |

Lowering `β2` makes `v̂` track recent gradient sizes faster. A sudden gradient spike then gets normalized away sooner instead of producing that worst-case `3·lr` step.

______________________________________________________________________

## Adam vs AdamW: Where Weight Decay Goes

Weight decay pulls every weight toward zero a little each step, which discourages the model from relying on a few huge weights. There are two ways to add it, and in Adam they are **not** the same.

**`Adam(weight_decay=λ)`: L2 penalty, coupled** with the default `decoupled_weight_decay=False`. PyTorch adds `λ·w` to the gradient *before* the moments:

```
grad ← grad + λ·w
... then the normal Adam update
```

The decay term now goes through the `m̂ / √v̂` normalization. Two bad things follow:

- Weights with large gradients (large `√v̂`) get their decay divided away, so they're barely regularized. Weights with tiny gradients get decayed hard.
- A decay term that is tiny compared with `w` still gets normalized up to a full `lr`-sized step.

**`AdamW(weight_decay=λ)`: decoupled** (Loshchilov & Hutter, 2019). The decay is applied straight to the weight, outside the normalization:

```
w ← w · (1 − lr·λ)
... then the normal Adam update, using the raw grad
```

The difference is stark on a single weight `w = 2.0` with an explicit zero gradient tensor, `lr = 0.1`, `λ = 0.01`, after one step:

| Optimizer | `w` after one step | Why |
|---|---|---|
| `Adam(weight_decay=0.01)` | **1.900** | `grad = 0.02`, normalized to a full `lr` step |
| `AdamW(weight_decay=0.01)` | **1.998** | `2.0 · (1 − 0.1·0.01)` |

**Use `AdamW` whenever you want weight decay.** Its PyTorch default is `weight_decay=1e-2`, while `Adam`'s is `0`. `Adam(..., decoupled_weight_decay=True)` is the same algorithm as `AdamW`.

Common practice is to **not decay biases and normalization weights** (LayerNorm's `γ`/`β`). Their role differs from weight matrices, and pulling LayerNorm's scale toward zero can fight the normalization. Use two param groups:

```python
decay, no_decay = [], []
for param in model.parameters():
    (no_decay if param.ndim < 2 else decay).append(param)

optimizer = torch.optim.AdamW(
    params=[
        {"params": decay, "weight_decay": 0.01},
        {"params": no_decay, "weight_decay": 0.0},
    ],
    lr=3e-4,
)
```

`param.ndim < 2` catches biases and LayerNorm parameters (all 1-D) and keeps embedding and linear weight matrices in the decayed group.

______________________________________________________________________

## Picking a Learning Rate

### Starting points

These are well-trodden starting points, not answers. Always check the loss curve.

| Setting | Starting `lr` | Notes |
|---|---|---|
| Small MLP / CNN, Adam | `1e-3` | The default. What `mnist.py` uses |
| Transformer from scratch, AdamW | `1e-4` to `5e-4`, with warmup | Bigger models need smaller values |
| Fine-tuning a pretrained model | `1e-5` to `5e-5` | You're nudging good weights, not searching |
| SGD + momentum 0.9 (vision) | `0.1` at batch size 256 | Scale with batch size (below) |

For reference, the original Transformer (Vaswani et al., 2017, §5.3) used Adam with `β2 = 0.98`, `ε = 1e-9`, and a schedule ([the Noam schedule](optimizer_and_scheduler.md#the-noam-schedule)) that peaks at about `7e-4` after 4000 warmup steps for `d_model = 512`.

### The LR range test

Instead of guessing, measure (Smith, 2017). Start from a tiny `lr`, multiply it by a constant factor after every batch, and record the loss. Plot loss against `lr` on a log x-axis.

```python
import math

import torch
from torch import nn
from torch.utils.data import DataLoader


def lr_range_test(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    lr_min: float = 1e-7,
    lr_max: float = 1.0,
    num_steps: int = 200,
) -> tuple[list[float], list[float]]:
    """Sweep the learning rate exponentially and record the loss at each step.

    Trains the model in place, so pass a throwaway copy
    (``copy.deepcopy(model)``). The loader must yield at least
    ``num_steps`` batches.

    Returns:
        (lrs, losses), one entry per step; stops early once the loss blows up.
    """
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr_min)
    growth = (lr_max / lr_min) ** (1 / (num_steps - 1))
    lrs, losses = [], []
    for step, (x, y) in zip(range(num_steps), loader):
        lr = lr_min * growth**step
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        lrs.append(lr)
        losses.append(loss.item())
        if not math.isfinite(loss.item()) or loss.item() > 4 * min(losses):
            break
    return lrs, losses
```

Reading the plot:

```
loss
 |---------.
 |          \
 |           \           /
 |            \         /
 |             '--...--'
 +-------------------------- lr (log)
   flat    steep    min   explodes
```

- Flat on the left: `lr` too small to change anything.
- The loss falls fastest in the middle.
- It bottoms out, then explodes.

**Pick roughly 10× below the `lr` at the minimum**, or the point of steepest descent. The minimum itself is already on the edge of instability. Loss is noisy batch to batch, so smooth the curve (for example, an exponential moving average) before reading it.

### Batch size moves the learning rate

A bigger batch gives a less noisy gradient, so you can afford a bigger step. These are heuristics, and both break down at very large batches:

- **SGD: linear scaling.** Batch ×k → `lr` ×k, plus warmup (Goyal et al., 2017). This is how they trained ImageNet at batch size 8192.
- **Adam: square-root scaling.** Batch ×k → `lr` ×√k (Malladi et al., 2022). Adam already normalizes by gradient magnitude, so it gets less benefit from the reduced noise.

If you change the batch size and keep the `lr`, you have changed the experiment.

______________________________________________________________________

## Warmup and Decay

[optimizer_and_scheduler.md](optimizer_and_scheduler.md#the-scheduler) covers *how* schedulers edit `param_groups` and which one to pick. Here is *why* Adam in particular wants one.

**Warmup.** In the first steps, `v̂` is estimated from only a handful of gradients, so the per-parameter step sizes are themselves noisy. Bias correction fixes the *average* size, not the *variance*. Large, badly aimed steps at random initialization can push training into a bad region it never recovers from (Liu et al., 2020, the RAdam paper). Ramping `lr` up over the first few hundred to few thousand steps gives `v̂` time to settle. Transformers need this most. See [transformer/norm.md](../transformer/norm.md#q-why-is-the-number-of-layers-what-breaks-post-norm).

**Decay.** Late in training you're near a minimum. A large `lr` keeps bouncing around it (the plateau row in [Symptoms](#symptoms)). Shrinking `lr` lets the weights settle into the bottom of the basin.

For how to build these (warmup + cosine with `SequentialLR`, the Noam schedule with `LambdaLR`, and which cadence to step them at), see [Warmup and Decay](optimizer_and_scheduler.md#warmup-and-decay) in the scheduler guide.

______________________________________________________________________

## Inspecting a Live Optimizer

```python
optimizer.param_groups[0]["lr"]                   # the current base lr (what a scheduler edits)
state = optimizer.state[model.fc.weight]          # buffers populated after the first step()
state["step"], state["exp_avg"], state["exp_avg_sq"]
```

A useful health check is the **update-to-weight ratio**: how much each weight matrix changes per step relative to its size. A rough heuristic from CS231n is around `1e-3`. Far below that suggests `lr` is too low; far above suggests it's too high.

```python
before = {n: p.detach().clone() for n, p in model.named_parameters()}
optimizer.step()
for name, param in model.named_parameters():
    ratio = (param.detach() - before[name]).norm() / before[name].norm()
    print(f"{name:30s} {ratio.item():.1e}")
```

Log this for a few steps at the start of training, not every step. The clone doubles parameter memory.

______________________________________________________________________

## Cheat-Sheet

| Question | Answer |
|---|---|
| What does `lr` mean for Adam? | The scale of a normalized update; it is not a strict cap |
| First thing to try | `AdamW(lr=1e-3)` for small models; `3e-4` + warmup for transformers |
| Loss is `nan` | Check invalid operations and masking, then try a lower `lr` or warmup |
| Want weight decay | `AdamW`, not `Adam(weight_decay=...)`; skip biases and norms |
| Changed batch size | Recheck `lr`; √k for Adam and k for SGD are starting heuristics |
| Unsure about `lr` | Run the LR range test, pick ~10× below the minimum |
| Loss plateaus early | Add decay (cosine), or lower `lr` late in training |
| No fixed end to training | Noam: linear warmup, then `lr ∝ 1/√step`; step it per batch |

______________________________________________________________________

## Common Confusions

| Easy to mix up | What to remember |
|---|---|
| Gradient magnitude versus Adam update size | The first Adam step with a consistent nonzero gradient is near `lr` per coordinate, regardless of raw gradient scale. Later steps depend on moment history. |
| Bias correction versus warmup | Bias correction removes the zero-initialization bias of the moments; warmup limits the global learning rate while early estimates are still noisy. |
| L2 decay versus AdamW decay | Coupled L2 enters Adam's moments; AdamW shrinks weights separately from gradient normalization. |
| A plateau versus overfitting | A high training loss that will not settle may call for decay; rising validation loss while training improves usually points to generalization. |

## Check Your Recall

1. Why does doubling a consistent gradient double an SGD step but often barely change an Adam step?
1. At the first Adam update, what does bias correction remove—and what does it leave uncertain?
1. Why does decoupled weight decay behave differently from adding `λ·w` to Adam's gradient?

<details markdown="1">
<summary>Answers</summary>

1. SGD multiplies the raw gradient by `lr`; Adam divides its smoothed gradient by the square root of smoothed squared gradients, so the common scale mostly cancels when `ε` is negligible.
1. It removes the systematic shrinkage caused by initializing `m` and `v` at zero. One gradient is still a noisy estimate of what future gradients will look like.
1. Adding `λ·w` changes Adam's moment estimates and is normalized with the task gradient. AdamW applies a separate weight shrinkage proportional to `lr·λ`.

</details>

**One-minute recap:** For a fixed loss curvature, too large an `lr` overshoots and too small an `lr` crawls. Adam uses gradient history to adjust each coordinate, but its global `lr` still controls overall step scale. Bias correction, warmup, decay, and AdamW solve different problems; choose them by the behavior you need to explain.

______________________________________________________________________

## Sources

- Kingma & Ba, *"Adam: A Method for Stochastic Optimization"* (ICLR 2015). The algorithm, bias correction, and step-size bound. arXiv:1412.6980
- Loshchilov & Hutter, *"Decoupled Weight Decay Regularization"* (ICLR 2019). AdamW. arXiv:1711.05101
- Smith, *"Cyclical Learning Rates for Training Neural Networks"* (WACV 2017). The LR range test. arXiv:1506.01186
- Goyal et al., *"Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour"* (2017). Linear scaling and warmup. arXiv:1706.02677
- Liu et al., *"On the Variance of the Adaptive Learning Rate and Beyond"* (ICLR 2020). Why adaptive optimizers need warmup (RAdam). arXiv:1908.03265
- Malladi et al., *"On the SDEs and Scaling Rules for Adaptive Gradient Algorithms"* (NeurIPS 2022). Square-root scaling for Adam. arXiv:2205.10287
- Vaswani et al., *"Attention Is All You Need"* (2017), §5.3. Transformer optimizer settings and the Noam schedule. arXiv:1706.03762
- CS231n notes, *"Neural Networks Part 3: Learning and Evaluation"*. Update-to-weight ratio heuristic. cs231n.github.io/neural-networks-3/
- PyTorch source: `torch/optim/adam.py` (update rule, `decoupled_weight_decay`), `torch/optim/adamw.py` (`weight_decay=1e-2` default)
