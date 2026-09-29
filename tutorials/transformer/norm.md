# Pre-Norm vs. Post-Norm

**Revision question:** When a transformer block adds a new result to its residual
stream, does it normalize the running stream itself or only the copy read by the
next sublayer? That one placement choice changes the path gradients can take.

**Reading route:** Start with [one token's normalization](#what-layer-normalization-does),
then [trace two sublayers](#pre-norm-vs-post-norm-where-layernorm-goes), and finish
with the gradient-path question. The depth and FFN questions, the
[edge-set diagrams](#the-edge-set-for-two-sublayers), and the interpretability
discussion add detail when you need it. Use the [recall check](#recall-check)
after a break.

## What Layer Normalization Does

Think of one token's features as a row. LayerNorm compares each feature with the
other features **in that row**. For `x = [1, 2, 3]`, the mean is `2`, the variance
is `2/3`, and the centered, rescaled row is approximately
`[-1.225, 0, 1.225]` (ignoring `ε`). The row `[10, 20, 30]` gives the same
normalized values: its overall scale is gone, while its feature pattern remains.

Given an input vector `x ∈ ℝ^d` — one token's activations — the full operation is:

```text
LN(x) = γ ⊙ [(x − μ) / √(σ² + ε)] + β
```

where `μ = (1/d) Σ_i x_i` is the mean, `σ² = (1/d) Σ_i (x_i − μ)²` is the variance,
`γ` and `β` are learned per-dimension scale and shift parameters, `ε` is a small
constant for numerical stability, and `⊙` is element-wise multiplication.

Two stages, not one:

1. **Center and rescale.** Subtract the mean, divide by the standard deviation. The
   result has approximately zero mean and unit variance (exactly so when `ε = 0`
   and variance is nonzero) — no learned parameters involved yet.
1. **Learned affine transform.** `γ` and `β` shift and rescale the normalized
   vector per-dimension, letting the model undo the normalization in whichever
   directions it isn't helpful.

Layer normalization computes `μ` and `σ` **within one token's vector**,
independently of every other position and every other example in the batch —
the key axis difference from BatchNorm, which uses statistics across examples
for each feature instead. One consequence: apart from the small `ε` term,
multiplying the entire
input vector by a positive scalar leaves `LN(x)` unchanged before the affine
transform. A sublayer reading only the normalized vector has no way to recover
that overall scale — it's thrown away by construction.

**Vocabulary**, for the token × feature picture below:

- **Row** = one token. Its whole "meaning vector" at this point in the network.
- **Column** = one feature (channel / dimension) — `d_model` of them, e.g. 4096.
- **Feature** = a single column. "Feature 2 of token `cat`" is one cell.

```
                feat_1   feat_2   feat_3   feat_4   ...
token "the"   [  0.8     -1.2      3.5      0.1     ... ]
token "black" [ -2.1      0.3      0.0      4.7     ... ]
token "cat"   [  1.0      4.0     -0.6      0.2     ... ]
```

The row for `"cat"` is that token's whole vector, `[1.0, 4.0, -0.6, 0.2, ...]`. The
`feat_2` column read down every row is the same feature, compared across tokens.
`LN` normalizes **along a row** — across `feat_1 … feat_{d_model}` for one token —
never down a column.

### Which axes: `normalized_shape`

```python
ln = nn.LayerNorm(normalized_shape=d_model)
out = ln(x)  # x: (B, L, d_model) → out: (B, L, d_model)
```

`normalized_shape` names the **trailing** dimensions that supply `μ` and `σ²`, and
it is also the shape of `γ` and `β`. Passing `d_model` gives `B × L` independent
normalizations, one per row of the picture above. Pooling over more axes would
break that:

- **Adding `L`** (`normalized_shape=(L, d_model)`) mixes tokens. One token's output
  would depend on every other token in the sequence, padding included, and on the
  sequence length. In a causal decoder, future tokens would leak into earlier
  positions through `μ` and `σ`.
- **Pooling across the batch** is BatchNorm's choice. One example's output would
  depend on which other examples share its batch, and training (batch statistics)
  would behave differently from inference (running averages).

### Why it's called layer norm

If it normalizes per token, why the name? The name says which axis the statistics
pool over, in contrast with BatchNorm:

- **BatchNorm** (Ioffe & Szegedy, 2015): fix one hidden unit, pool over the
  **batch**. Each feature gets a mean and variance computed across examples.
- **LayerNorm** (Ba, Kiros & Hinton, 2016): fix one example, pool over every hidden
  unit in the **layer**. The statistics come from that layer's activations.

The paper targeted MLPs and RNNs, where a layer's output for one input is a single
vector, so "normalize over the layer" meant "normalize that vector." A transformer
adds a sequence axis: every position gets its own `d_model` vector from the layer,
and LayerNorm treats each position the way the paper treated one input. "Per token"
is what "per layer, per example" becomes once there is a sequence axis.

The name can mislead in the other direction too. In the paper's convolutional
setting, "the layer" meant all of `(C, H, W)`, so it pooled over spatial positions
as well as channels. Read `normalized_shape`, not the name.

______________________________________________________________________

## What Normalization Really Removes

After `x̂ = (x − μ) / σ`, computed across the `d_model` features of a single token,
that token's vector has mean 0 and variance 1 — so its length is pinned:
`‖x̂‖ = √d_model` when `ε` is ignored and variance is nonzero. The vectors lie
on a sphere within the mean-zero subspace. Direction is free; magnitude is not.

So it isn't quite right to say "feature 1 can't be large." A feature far above
the token's mean remains far above the mean after normalizing: subtracting one
mean and dividing by one standard deviation preserves the ordering and ratios of
*pairwise differences*. It does not preserve ratios of raw feature values, because
centering changes those. What's destroyed is the **token's original mean and
overall scale**, not its pattern of relative differences. That matters because
whatever reads `x̂` next is a matmul feeding an activation or a softmax, and both
of those care about absolute magnitude, not just direction.

`γ` and `β` let the model choose a shared scale and baseline for each feature.
They reshape and shift the normalized vectors, but they **cannot recover each
token's original mean or scale**: those were discarded before the affine step.
The resulting feature scales are learned by gradient descent rather than inherited
from the preceding layer.

______________________________________________________________________

## The Mental Model for `γ` and `β`

`γ_i` answers "how much should feature dimension `i` be amplified?" — that's a
property of the **feature**, not the token. `β_i` is the paired answer for "what
should the baseline value of feature `i` be?" Both are indexed by feature only:
shape `[d_model]`, no batch or sequence axis. The same `γ_i`, `β_i` apply to every
token, in every sequence, in every batch.

**Yes, they're learned.** `γ` and `β` are `nn.Parameter`, updated by the optimizer
exactly like any weight matrix. Writing `y = γ ⊙ x̂ + β` and `g = ∂L/∂y`, the
gradients are:

```
∂L/∂β_i = Σ_{b,l}  g[b, l, i]
∂L/∂γ_i = Σ_{b,l}  g[b, l, i] · x̂[b, l, i]
```

That sum over every `(b, l)` is the signature of a shared parameter: `γ_i` was
used at all `B × L` token positions, so it accumulates gradient contributions from
every one of them. This is exactly why the shapes work out — the upstream
gradient is `[B, L, d]`, and you reduce away the axes the parameter was broadcast
over (`B` and `L`) to land back at `[d]`, matching `γ`'s and `β`'s own shape.

Separately, gradient also flows backward through `x̂` into `x` — and *that* path is
not elementwise, because `μ` and `σ` are themselves functions of every element of
the token. Changing `x_j` changes `μ` and `σ`, which changes `x̂_i` for every `i`,
not just `i = j`. That's why `LayerNorm`'s backward pass into `x` has cross terms
between features, even though the forward pass and the `γ`/`β` gradients above are
purely elementwise or reduction-only.

______________________________________________________________________

## Pre-Norm vs. Post-Norm: Where LayerNorm Goes

Every transformer block wraps two sublayers — self-attention and a feedforward
network — in a residual connection plus a `LayerNorm`. Where exactly the `LayerNorm`
goes is not a cosmetic choice. It changes how gradients flow during training and
what the residual stream even *means* for interpretability.

`r_l` is the residual stream entering block `l`. `Sublayer` stands for either
self-attention or the FFN — the same analysis applies to both.

**Post-norm** (the original Transformer):

```text
r_{l+1} = LN(r_l + Sublayer(r_l))
```

**Pre-norm** (GPT-2 and most modern architectures):

```text
r_{l+1} = r_l + Sublayer(LN(r_l))
```

**Trace two sublayers, attention `A` then feedforward `F`:**

| Step | Pre-norm: normalize the branch being read | Post-norm: normalize the new stream |
|---|---|---|
| Attention | `a = r₀ + A(LN₁(r₀))` | `a = LN₁(r₀ + A(r₀))` |
| Feedforward | `r₂ = a + F(LN₂(a))` | `r₂ = LN₂(a + F(a))` |
| Skip path through both | `r₀ → + → + → r₂` | `r₀ → + → LN₁ → + → LN₂ → r₂` |

Both variants have residual additions. Only pre-norm has a path through the stack
that bypasses every internal `LayerNorm`. A pre-norm stack usually applies a final
normalization before its output head, so that final norm still sees the result.

______________________________________________________________________

## Cheat-Sheet

| | Post-Norm | Pre-Norm |
|---|---|---|
| Recurrence | `r_{l+1} = LN(r_l + Sublayer(r_l))` | `r_{l+1} = r_l + Sublayer(LN(r_l))` |
| Gradient path to `r_0` | Every layer's backward step is multiplied by `LN`'s Jacobian | Every layer contributes an identity term — a direct path back to the input |
| LR warmup | Often important for stable training near initialization | Less tied to norm placement, though still commonly used |
| Residual stream magnitude | Renormalized every sublayer | Can grow with depth; a final `LN` is common before the output head |
| Is the stream a sum of per-layer contributions? | Each addition is immediately renormalized | Yes, before the final `LN` — this is `hook_resid_pre` in TransformerLens |
| Used by | Vaswani et al. 2017 (original Transformer) | GPT-2 and most descendants |

______________________________________________________________________

## Q: What does "depth" mean here?

The next three questions supply vocabulary about depth and feedforward layers.
You can skip to the graph and gradient sections if those terms are already
familiar.

**Depth** is the number of stacked residual blocks (layers) in the model — the `L`
in "a 12-layer / 96-layer transformer." It has nothing to do with model width
(`d_model`) or sequence length.

> In this tutorial `L` counts layers. In the code, `L` in jaxtyping shapes is
> sequence length (`app/src/training/transformer/constants.py`). Same letter, unrelated
> quantity.

Each transformer layer has two sublayers, attention and the FFN, and each sublayer
gets its own residual connection and its own `LN`. So an `L`-layer model has `2L`
places where norm placement matters. The recurrences in this tutorial step once per
sublayer, which means a real `L`-layer model applies them `2L` times. The factor of
2 changes no argument below, so the math just writes `L`.

______________________________________________________________________

## Q: How wide is the FFN sublayer?

Width is a separate axis from depth. In *"Attention Is All You Need"*, the model
dimension was `d_model = 512` and the FFN's inner dimension was `d_ff = 2048`:

```
FFN(x) = W_2 · ReLU(W_1 · x + b_1) + b_2
W_1: d_model → d_ff      (512 → 2048)
W_2: d_ff → d_model      (2048 → 512)
```

This fourfold expansion, `d_ff = 4 × d_model`, is a common rule of thumb, though
variations exist. Expanding lets the FFN project each token into a
higher-dimensional space, where complex patterns may be easier to learn, before
projecting back down to `d_model`. The projection back is what lets the output be
added to the residual stream.

______________________________________________________________________

## Q: ReLU or GELU inside the FFN?

The nonlinearity between `W_1` and `W_2` is the only thing that stops the FFN from
collapsing into a single linear map. The original Transformer used ReLU. GPT-1,
GPT-2, and BERT switched to GELU.

**ReLU** gates each unit by the sign of its input:

```
ReLU(x) = max(0, x)
ReLU'(x) = 1 if x > 0, else 0
```

It's cheap and gives exact zeros, but negative inputs get exactly zero local
gradient. If a row of `W_1` has a negative pre-activation for every token, that
unit outputs 0 and receives no gradient through those inputs. It can stay inactive
unless its inputs change enough through updates elsewhere. This is a **dead
neuron**.

**GELU** (Gaussian Error Linear Unit) replaces the hard gate with a soft one:

```
GELU(x) = x · Φ(x)        Φ = standard normal CDF
```

ReLU multiplies `x` by 0 or 1. GELU multiplies `x` by `Φ(x)`, the probability that a
standard normal sample is below `x`. That factor slides smoothly from 0 to 1. For
large `|x|` the two functions agree. The difference is near zero:

| `x` | −2 | −1 | −0.5 | 0 | 0.5 | 1 | 2 |
|---|---|---|---|---|---|---|---|
| `ReLU(x)` | 0 | 0 | 0 | 0 | 0.5 | 1 | 2 |
| `GELU(x)` | −0.046 | −0.159 | −0.154 | 0 | 0.346 | 0.841 | 1.954 |
| `GELU'(x)` | −0.085 | −0.083 | 0.133 | 0.5 | 0.867 | 1.083 | 1.085 |

(Values computed with `torch.nn.functional.gelu`.)

Three consequences:

- **Smooth.** There's no kink at 0. The derivative there is 0.5.
- **Negative inputs still get gradient.** At `x = −2` the gradient is small but not
  zero, so a unit pushed negative can still be pulled back. It doesn't die the way
  a ReLU unit does.
- **Not monotonic.** GELU dips to a minimum of about −0.17 at `x ≈ −0.75` before
  rising back toward 0. Slightly negative inputs give slightly negative outputs, and
  very negative inputs give roughly zero.

The adoption of GELU in GPT and BERT is an empirical design choice, not a theorem
that it always trains better than ReLU. Smooth gradients and fewer completely
inactive units are common motivations.

In PyTorch:

```python
h = F.relu(input=W_1(x))
h = F.gelu(input=W_1(x))  # exact: x * Φ(x)
h = F.gelu(input=W_1(x), approximate="tanh")  # tanh approximation
```

The `tanh` approximation, `0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))`, is
within 5e-4 of exact GELU on `[−3, 1]`. The older `x · sigmoid(1.702 · x)` shortcut
is looser, at about 0.02. Many recent LLMs have moved on from both ReLU and GELU to
gated FFNs such as SwiGLU (Shazeer 2020).

______________________________________________________________________

## Q: The op sequence is the same either way — aren't the two graphs identical?

This is the deeper graph explanation behind the two-row trace above. Follow the
skip edge through the diagrams; counting `N`, `S`, and `+` alone loses it.

Write out the flat sequence of operations for a stack of `2L` sublayers, with `N`
for the norm, `S` for the sublayer body, and `+` for the residual add:

```
pre-norm (with its final norm):   N S +  N S +  N S +  …  N S +  N
post-norm:                          S + N  S + N  S + N  …  S + N
```

Same symbols, same order, same counts. The two sequences differ by exactly one
leading `N` — not "roughly," but identically up to that single boundary term. So
the question is a fair one, and the answer is that the flat reading is lossy.

### `+` is a merge, not a unary op

Writing the ops in a line treats `+` as one more link in a chain. It isn't. It is
the one node with **two** incoming edges, and the linear text silently drops the
second one.

```
pre-norm                          post-norm

x ──────┬──────────► +            x ──────┬──────► + ──► N ──►
        │            ▲                    │        ▲
        └─► N ─► S ──┘                    └─► S ───┘

  branch point is BEFORE N          branch point is AFTER N
  → N sits on the sublayer branch   → N sits on the trunk
```

In pre-norm the skip connection taps `x` *upstream* of the norm, so the norm sits
on a side branch that dead-ends into the add. In post-norm the tap comes off the
norm's output, so the norm sits on the trunk and everything downstream — including
the next block's skip connection — inherits it.

### The edge set, for two sublayers

Label the adds `A₁, A₂`, the norms `N₁, N₂`, the sublayers `S₁, S₂`:

```
pre-norm

x₀ ──┬──────────► A₁ ──┬──────────► A₂ ──► N_final ──►
     │            ▲     │           ▲
     └─ N₁ ─ S₁ ──┘     └─ N₂ ─ S₂ ─┘

post-norm

x₀ ──┬──────► A₁ ──► N₁ ──┬──────► A₂ ──► N₂ ──►
     │        ▲            │       ▲
     └─ S₁ ───┘            └─ S₂ ──┘
```

Same node types, same counts (modulo the one extra final norm). Different edges.
Pre-norm has a direct `A₁ → A₂` edge; post-norm has none, and every transit between
consecutive adds is forced through an `N`.

That makes the two DAGs non-isomorphic. One useful invariant is: *does a path from
the input to the last residual addition touch zero internal norm nodes?*

| | Post-norm | Pre-norm |
|---|---|---|
| Direct edge `Aᵢ → Aᵢ₊₁` | No — forced through an `N` | Yes |
| Norms on the identity path | `2L` | 0, at any depth |
| A norm-free path to the last add | None exists | `x₀ → A₁ → A₂ → … → A_2L` |

No relabeling of nodes repairs that. The difference lives in the edge set, not the
node list — which is why counting ops finds nothing and why the gap *widens* with
depth rather than washing out. The two are identical only at `L = 0`.

### A witness you can run

Force every sublayer to output exactly zero (`S ≡ 0`) and the two stacks collapse
to different expressions:

```
pre-norm:   N_final(x₀)           one norm, whatever L is
post-norm:  N ∘ N ∘ … ∘ N (x₀)    2L norms
```

Symbolically those are very different. Numerically there is a catch worth knowing:
with freshly initialized parameters (`γ = 1`, `β = 0`), `LayerNorm` is idempotent —
`N(x)` already has zero mean and unit variance, so `N(N(x)) = N(x)` up to the `ε`
term. At init the two collapse to the *same* vector. Give the norms non-default
affine parameters, as training does, and they separate at once:

```
fresh init (γ = 1, β = 0):   max|pre − post| = 0.0000
random γ, β:                 max|pre − post| = 2.0439
```

(`d_model = 8`, 6 sublayers, `x₀ ~ 10 · N(0, 1)`, `γ ~ N(1, 0.3²)`, `β ~ N(0, 0.3²)`.)

So the zeroed-sublayer test witnesses that the *graphs* differ; it is not evidence
that the two models differ at step 0. The structural argument above holds either
way, and the gradient consequence in the next section is what bites during
training.

### Where the distinction hides in the code

`nn.Sequential` trains you to read a network as a list of layers. A residual
network needs an explicit branch and addition, which is why this block writes
`forward` directly. In `EncoderBlock.forward`, the skip edge is the bare
`batch` on the left of the `+`:

```python
attn_in_normalized = self.attn_in_norm(batch)
attn, weights = self.attn(batch=attn_in_normalized, pad_mask=pad_mask)
batch = batch + self.dropout(attn)
```

The normalized value is bound to `attn_in_normalized` and never written back to `batch`, so
the `batch` on the right of the `+` is still the *unnormalized* value that entered
the block. A hypothetical post-norm variant moves normalization after the add:

```python
attn, weights = self.attn(batch=batch, pad_mask=pad_mask)
batch = post_attn_norm(batch + self.dropout(attn))
```

Which value the name `batch` refers to at the moment of the add is the entire
pre-norm/post-norm distinction, and it is invisible in any flattened listing of the
layers.

______________________________________________________________________

## Q: Mechanically, why does post-norm hurt gradient flow?

Write the local Jacobian of each recurrence with respect to its input `r_l`.

**Post-norm:**

```
r_{l+1} = LN(r_l + Sublayer(r_l))
∂r_{l+1}/∂r_l = J_LN · (I + J_Sublayer)
```

Backprop through `L` stacked layers multiplies `L` of these together:

```
∂r_L/∂r_0 = Π_l  J_LN,l · (I + J_Sublayer,l)
```

`LayerNorm`'s Jacobian is not the identity — it rescales by the input's own
standard deviation and removes the mean, so its singular values depend on the
local statistics of whatever it's fed. Every one of the `L` factors in that
product includes a non-identity `J_LN` term. Multiplying many non-identity
Jacobians together is exactly the setup that causes vanishing/exploding gradients
in any deep network — the transformer isn't special here, it's just deep.

**Pre-norm:**

```
r_{l+1} = r_l + Sublayer(LN(r_l))
∂r_{l+1}/∂r_l = I + J_Sublayer · J_LN
```

```
∂r_L/∂r_0 = Π_l  (I + J_Sublayer,l · J_LN,l)
```

Expand that product. Picking the `I` term at every single layer is one term in
the expansion with coefficient 1 no matter how deep the stack is: there is a
path from the last addition back to the input that never multiplies by a
sublayer or an internal `LayerNorm` Jacobian. That's the algebraic content of
"the residual connection gives gradients a highway." Other terms can still add
to or cancel this term, and a final output norm has its own Jacobian, so this
path does not guarantee a particular gradient norm. Post-norm has no such
internal identity path, because `LN` sits outside each `+`.

This is the same identity-mapping argument He et al. made for ResNets. Xiong et al.
(2020) worked out the transformer version, with explicit bounds covered in the next
section.

______________________________________________________________________

## Q: Why is the number of layers what breaks post-norm?

Both variants have a residual connection. What differs is whether the residual
stream is a clean identity path, and that depends on where the `LN` sits:

- **Pre-norm:** `x_{l+1} = x_l + F(LN(x_l))`. The Jacobian is
  `I + J_F · J_LN`, so the product through `L` layers includes an identity
  path. If the residual stream grows, `J_LN` often scales down the branch
  contribution; this can make training less sensitive to depth. The full
  product need not stay close to identity, and its actual gradient scale depends
  on the parameters, data, and final output path.
- **Post-norm:** `x_{l+1} = LN(x_l + F(x_l))`. The `LN` sits on the main path, so
  the backward pass is multiplied by the `LN` Jacobian (roughly a `1/σ` rescaling)
  at each of the `2L` sublayers. Those factors compound, so different layers get
  gradients of different scale, and the gap grows with `L`.

Xiong et al. (2020) make this precise at initialization. The gradient norm on the
last layer's parameters is bounded by:

```
post-norm:  O(d · √(ln d))        independent of L
pre-norm:   O(d · √(ln d / L))    shrinks as L grows
```

These are initialization-time bounds under the assumptions of that analysis,
not guarantees for every architecture throughout training. They explain why,
in Xiong et al.'s measurements, post-norm's per-layer gradient norm rises toward
the output layers while pre-norm's is more even across layers.

**Why warmup helps.** A large step size at step 0 can hit that big gradient near the
output and make training diverge. The original Transformer's answer was a
learning-rate **warmup schedule**: start the LR near zero and ramp it up over the
first few thousand steps, so the early updates stay small until the network is past
that fragile starting point. Skip warmup with post-norm and training often diverges
in the first few hundred steps.

**How deep before it breaks.** These are historical rules of thumb for
conventional post-norm training, not architectural limits:

| Depth | Post-norm |
|---|---|
| 6 encoder + 6 decoder (Vaswani et al. 2017) | Fine, with warmup |
| ~12–18 layers | Gets finicky |
| Past ~20–30 layers | Often needs extra care (Admin/DeepNet-style initialization, careful residual scaling) |

Liu et al. (2020) trace this to post-norm's output shift, which gets larger with
depth. Pre-norm, by contrast, trains 30+ layer encoders and today's 80–120 layer
LLMs. Those LLMs still usually use a warmup schedule, which also helps with
other early-training issues such as Adam's noisy early statistics. But with
pre-norm, warmup is no longer the only thing preventing divergence.

______________________________________________________________________

## Q: If pre-norm's stream is a raw, un-renormalized sum, why doesn't it blow up?

It does grow. Nothing in the pre-norm recurrence bounds `‖r_l‖` — each block adds
`Sublayer(LN(r_l))` on top of whatever was already there, and the addition is
never rescaled back down. Empirically, the residual-stream norm in pre-norm models
tends to grow across depth rather than staying flat.

Two things keep this from being a problem:

- **Every sublayer's own input is still normalized.** `LN(r_l)` is what
  self-attention and the FFN actually see, so a sublayer's forward pass operates
  on a unit-scale input regardless of how large `r_l` itself has gotten.
  `LayerNorm` here is a *read* operation — it doesn't touch what gets written back
  to the stream.
- **One more `LN` sits between the last block and the output head.** GPT-2-style
  stacks add a final `LN` (often called `ln_f`) after the last block, before the
  unembedding matrix. Without it, the ever-growing, unbounded-scale residual
  stream would be fed directly into the logit projection.

So pre-norm doesn't remove normalization — it moves every `LN` to be a *read*
at the top of a sublayer, plus one final read before the output, rather than a
*write* on the stream itself at every layer.

______________________________________________________________________

## Q: Why does this matter for interpretability specifically?

Because of what "the residual stream" is allowed to mean. Unroll the pre-norm
recurrence across all `L` layers:

```
r_L = r_0 + Sublayer_0(LN(r_0)) + Sublayer_1(LN(r_1)) + ... + Sublayer_{L-1}(LN(r_{L-1}))
```

That's an exact sum **at the residual stream**: the embedding and each sublayer
output are added without normalizing the running total. Attention outputs can be
decomposed further into head contributions. This supports direct logit
attribution (project a term onto the unembedding), activation patching (swap a
contribution and measure the downstream effect), and activation steering (add a
vector to the stream). The later sublayers depend on the changed stream, and the
final norm is nonlinear, so changing one term does **not** imply an exactly
additive change in final logits.
This is why MI code — TransformerLens included — reads the stream at
`hook_resid_pre`, the point right before a block's internal `LN`, since that value
equals the running sum of every contribution so far, untouched.

Unroll the post-norm recurrence and the sum breaks:

```
r_1 = LN(r_0 + Sublayer_0(r_0))
r_2 = LN(r_1 + Sublayer_1(r_1))
    = LN(LN(r_0 + Sublayer_0(r_0)) + Sublayer_1(r_1))
    ...
```

Every addition is immediately passed through a nonlinear `LN` before the next
term gets added. `Sublayer_0`'s contribution to `r_2` isn't the raw vector it
produced — it's whatever that vector became after being folded into `r_1`'s `LN`
and then partially re-expressed through `Sublayer_1`. There is no checkpoint where
"the stream" equals a clean sum of raw per-layer outputs. Attribution and patching
can still be applied to post-norm models, but they need to account for the
intervening normalization. This is one reason pre-norm residual streams are
especially convenient for interpretability work.

______________________________________________________________________

## Where this fits in our code

`app/src/training/transformer/modules.py` has `MultiHeadAttentionLayer`, which is only the
`Sublayer` term — it takes a batch and returns the attention output, with no
residual add and no block-level `LN` wrapping it. `EncoderBlock` in the same file
currently uses pre-norm for both sublayers. The two possible placements are:

```python
# Post-norm
x = ln(x + self_attn(x))
x = ln(x + ffn(x))

# Pre-norm
x = x + self_attn(ln(x))
x = x + ffn(ln(x))
```

For a decoder-only causal language model (see
[attention_masks.md](attention_masks.md)), pre-norm plus a final `LN` before the
output projection is the GPT-2 convention. That final norm is a separate design
choice outside the `EncoderBlock` shown here.

______________________________________________________________________

## Common confusions

| If you remember… | Check this distinction |
|---|---|
| “LayerNorm normalizes tokens together.” | It computes statistics across **features within each token**. Its learned `γ` and `β` are shared across tokens. |
| “Layer norm pools over the whole layer, so over every token.” | “Layer” is the contrast with “batch”: one example's layer activations. In a transformer that is one token's `d_model` vector. |
| “Pre-norm skips normalization.” | The sublayer reads `LN(r_l)`, and the stack commonly has a final `LN`; the running residual sum itself bypasses the internal norms. |
| “The identity path makes gradients constant.” | It supplies one direct path through the internal stack; other paths and the final norm still affect the total gradient. |
| “An additive stream means an additive change in logits.” | Contributions add at the stream, while later sublayers and the output normalization can respond nonlinearly. |

## Recall check

1. For one token with features `[1, 2, 3]`, which axis supplies the mean and variance? Would multiplying that whole row by 10 change its normalized pattern much?
1. Trace the skip path across two pre-norm sublayers. Which operations does it bypass that a post-norm skip path cannot?
1. Why can a pre-norm stack have a growing residual stream without giving each sublayer a growing-scale input?

<details markdown="1">
<summary>Answers</summary>

1. The feature axis of that token. Ignoring `ε`, positive scaling gives the same normalized pattern; the learned affine parameters then act on it.
1. `r₀ → first add → second add → r₂`. It bypasses the two internal `LayerNorm` operations and both sublayer branches. Post-norm puts a `LayerNorm` after each add.
1. Each sublayer reads a normalized copy, `LN(r_l)`. The raw residual stream is carried through the additions, and a final normalization commonly sits before the output head.

</details>

**One-minute recap:** LayerNorm standardizes each token's features. Pre-norm uses
that standardized vector to *read* the residual stream, then adds a new result to
the raw stream. Post-norm normalizes the *new sum*. That placement determines
whether the internal skip path is an identity path and whether the running stream
remains an exact sum of sublayer outputs.

______________________________________________________________________

## References

- Ba, Kiros & Hinton, *"Layer Normalization"* (2016). Introduces `LN`, normalizing
  within a single example's features rather than across the batch. arXiv:1607.06450
- Vaswani et al., *"Attention Is All You Need"* (2017). Original Transformer,
  post-norm. arXiv:1706.03762
- Xiong et al., *"On Layer Normalization in the Transformer Architecture"* (2020).
  Formal analysis of gradient norm at initialization for pre-norm vs. post-norm,
  and why post-norm needs warmup. arXiv:2002.04745
- Hendrycks & Gimpel, *"Gaussian Error Linear Units (GELUs)"* (2016). Defines
  `GELU(x) = x · Φ(x)` and its `tanh` and sigmoid approximations. arXiv:1606.08415
- Shazeer, *"GLU Variants Improve Transformer"* (2020). Gated FFN activations,
  including SwiGLU. arXiv:2002.05202
- Liu et al., *"Understanding the Difficulty of Training Transformers"* (Admin,
  EMNLP 2020). Post-norm's output shift gets larger with depth, and an
  initialization that tames it. arXiv:2004.08249
- Nguyen & Salazar, *"Transformers without Tears: Improving the Normalization of
  Self-Attention"* (IWSLT 2019). Empirical results showing pre-norm trains deeper
  models. arXiv:1910.05895
- He et al., *"Identity Mappings in Deep Residual Networks"* (2016). The
  identity-path argument for why residual connections stabilize gradients, applied
  here to `LayerNorm` placement instead of conv blocks. Tests variants that place
  BN/ReLU at different points around the add — the same edge-placement question,
  pre-dating transformers. arXiv:1603.05027
- Veit, Wilber & Belongie, *"Residual Networks Behave Like Ensembles of Relatively
  Shallow Networks"* (NeurIPS 2016). An `n`-block residual net is a DAG with `2ⁿ`
  input→output paths, not a chain of `n` — why reading a residual stack as a flat
  list of ops loses the structure. arXiv:1605.06431
- Wang et al., *"Learning Deep Transformer Models for Machine Translation"* (ACL
  2019). 30-layer pre-norm encoders that diverge under post-norm. arXiv:1906.01787
- Radford et al., *"Language Models are Unsupervised Multitask Learners"* (GPT-2,
  2019). Moved `LayerNorm` to the input of each sublayer and added a final `LN`
  after the last block.
- Elhage et al., *"A Mathematical Framework for Transformer Circuits"* (Anthropic,
  2021). The residual stream as a shared communication channel that every layer
  reads from and writes to additively. transformer-circuits.pub
- TransformerLens docs (Neel Nanda). `hook_resid_pre` / `hook_resid_post` naming
  and the standard MI convention of reading the pre-norm residual stream.
