# Transformer Compute and Memory

**Revision question:** Given a model size, a token budget, and a set of GPUs, how
long might training take, and what must fit in each GPU's memory? Keep **total
work** (FLOPs), **work per second** (FLOP/s), and **bytes per GPU** separate.

**Reading route:** [Count the work](#count-the-training-work), then follow a
[one-billion-parameter memory budget](#a-worked-training-budget). Use
[activation memory](#activations-grow-with-the-microbatch) and
[sharding](#what-zero-shards) when the first estimate does not fit. Finish with
the [recall check](#recall-check). For the operations inside a block, see
[attention masks](attention_masks.md) and [pre-norm](norm.md).

These are planning estimates for a **dense transformer**. The arithmetic assumes
that matrix multiplications dominate training and that the model trains all of its
parameters. Attention kernels, context length, precision, optimizer, and parallel
layout change real costs. All GB figures below are decimal (`10⁹` bytes).

## Count the Training Work

| Symbol | Meaning | Unit |
|---|---|---|
| `P` | trainable parameters | parameters |
| `D` | tokens processed during training, including repeated passes | tokens |
| `C` | total training work | FLOPs |
| `N` | GPUs doing the work | GPUs |
| `r` | **sustained training throughput** per GPU | FLOP/s per GPU |
| `T` | elapsed training time | seconds |

For one token, a dense weight matrix needs roughly two floating-point operations
per weight in the forward pass: one multiply and one addition. Backpropagation
through that matrix computes both an input gradient and a weight gradient, adding
roughly twice the forward work. Across `D` tokens:

```text
forward   ≈ 2 P D FLOPs
backward  ≈ 4 P D FLOPs
training  ≈ 6 P D FLOPs
T         ≈ 6 P D / (N r) seconds
```

This is a useful first estimate, not an exact count. It omits some elementwise
operations, optimizer work, and attention's sequence-length-dependent cost. Long
contexts can make attention more significant. `r` must be a *measured* or
plausible sustained training rate for this workload; a GPU's advertised peak is
not the denominator to use for a wall-clock estimate.

### Work through one run

Suppose `P = 1 billion`, `D = 20 billion`, `N = 8`, and a representative run
sustains `r = 50 trillion FLOP/s` on each GPU:

```text
C ≈ 6 × 10⁹ × 20 × 10⁹ = 1.2 × 10²⁰ FLOPs
N r = 8 × 50 × 10¹² = 4 × 10¹⁴ FLOP/s
T ≈ 1.2 × 10²⁰ / (4 × 10¹⁴) = 300,000 s ≈ 3.5 days
```

The units cancel: `FLOPs / (FLOP/s) = seconds`. A **petaFLOP-day** is
`10¹⁵ FLOP/s × 86,400 s = 8.64 × 10¹⁹ FLOPs`; this run uses about `1.39`
petaFLOP-days. A **GPU-hour** instead counts devices times elapsed hours and says
nothing about their throughput unless the GPU type and achieved rate are known.

### Parameters and tokens are different choices

At fixed compute, `P × D` is approximately fixed: doubling `P` leaves roughly
half as many training tokens. The 2022 Chinchilla study found an approximate
`D ≈ 20P` balance for its compute-optimal setting. That is a result under its
training objective and budget, not a rule that every model should stop after
`20P` tokens. A smaller model trained longer may cost more compute to reach a
given loss, yet be cheaper to serve repeatedly because each generated token
reads fewer weights. Decide the inference target and training budget together.

The [source article's GPT-NeoX scaling plot](https://blog.eleuther.ai/transformer-math/#engineering-takeaways-for-compute-costs)
shows measured samples per second falling below ideal linear scaling as V100
GPU count rises. That is a historical measurement of one setup, so measure `r`
for the current model and cluster.

## Inference Memory: Weights and the KV Cache

**Weight memory** is `P × bytes per stored parameter`. The exponent determines
range and the mantissa determines precision, but only the *stored* format sets
weight bytes:

| Stored weights | Exponent bits | Mantissa bits | Bytes per parameter | `7B` parameters |
|---|---:|---:|---:|---:|
| FP32 | 8 | 23 | 4 | 28 GB |
| BF16 | 8 | 7 | 2 | 14 GB |
| FP16 | 5 | 10 | 2 | 14 GB |
| INT8 | — | — | about 1, plus scale metadata | about 7 GB plus metadata |

Every floating-point row also has one sign bit. TF32 is a **matrix-multiply
compute mode** with 8 exponent and 10 mantissa bits; it does not turn an FP32
weight tensor into a 19-bit storage format. FP32 weights still take 4 bytes each.
Quantized weights need scales and sometimes other metadata, so their exact memory
depends on the quantization method. The
[source's precision diagram](https://blog.eleuther.ai/transformer-math/#model-weights)
shows these bit layouts side by side.

For autoregressive inference, attention also keeps past keys and values. With
ordinary multi-head attention, an approximate KV-cache budget is:

```text
KV bytes ≈ 2 × layers × batch × cached_tokens × KV_width × bytes_per_value
```

The leading `2` counts keys and values. `KV_width = d_model` for ordinary
multi-head attention; grouped-query or multi-query attention uses fewer KV heads
and a smaller width. For `32` layers, batch `1`, `4096` cached tokens,
`KV_width = 4096`, and two-byte values, the cache alone is about `2.15 GB`.
A 7B BF16 model therefore needs more than its `14 GB` of weights, even before
temporary buffers and allocator overhead. The source article's `1.2 × weight memory` rule of thumb can miss a large cache at long context or high batch size.

## A Worked Training Budget

Training keeps more than weights. Use this **specific** mixed-precision Adam
layout: two-byte working weights and gradients, an FP32 master weight copy, and
FP32 Adam first and second moments. Before activations and temporary buffers:

| Resident item | Bytes per parameter | For `P = 1B` |
|---|---:|---:|
| Working weights | 2 | 2 GB |
| Gradients | 2 | 2 GB |
| FP32 master weights | 4 | 4 GB |
| Adam first moment | 4 | 4 GB |
| Adam second moment | 4 | 4 GB |
| **Total parameter state** | **16** | **16 GB** |

That `16P` figure belongs to this layout. Native autocast may instead keep FP32
weights and gradients with two FP32 moments (`4 + 4 + 8 = 16` bytes per
parameter), without a persistent low-precision weight copy. Optimizers with
different state, quantized moments, or frozen parameters change the total.

The full device estimate starts with four buckets:

```text
training memory ≈ weights + gradients + optimizer state + activations
                  + temporary/communication buffers
```

The first three buckets scale mostly with `P`. Activations depend on the
**microbatch on one GPU**, sequence length, hidden width, and layers. A model
that fits at batch size one may run out of memory when the microbatch grows,
even though `P` stays fixed.

### Activations grow with the microbatch

Backpropagation needs intermediate forward results. Keeping all of them saves
compute but uses memory. **Activation checkpointing** saves selected boundaries
and recomputes discarded intermediates during backward. This trades extra
forward work for a smaller activation bucket.

One published estimate for FP16 activations, with no sequence parallelism, uses
`s` = sequence length, `b` = microbatch per GPU, `h` = hidden width, `L` = layers,
`a` = attention heads, and `t` = tensor-parallel degree:

```text
no recomputation         ≈ s b h L (10 + 24/t + 5 a s/(h t)) bytes
selective recomputation  ≈ s b h L (10 + 24/t) bytes
full recomputation       ≈ 2 s b h L bytes
```

The `s²` contribution in the first line comes from attention intermediates;
selective recomputation removes it in this model. These constants describe the
implementation analyzed by *Reducing Activation Recomputation in Large
Transformer Models*, so use them to reason about scaling rather than as an
allocator-exact promise for every attention kernel. Full recomputation can add
up to roughly one extra forward pass: the `6PD` training estimate can approach
`8PD` before other overheads.

For a roughly 1B-parameter shape with `h = 2048`, `L = 18`, `a = 16`,
`s = 2048`, `b = 2`, and `t = 1`, `s b h L = 150,994,944`. The formulas give:

| Activation policy | Approximate activation bytes per GPU | With the `16 GB` replicated parameter state |
|---|---:|---:|
| No recomputation | 17.2 GB | 33.2 GB |
| Selective recomputation | 5.13 GB | 21.1 GB |
| Full recomputation | 0.30 GB | 16.3 GB |

Even the `21.1 GB` selective estimate leaves less than 3 GB on a 24 GB GPU
for temporary buffers and allocator overhead. The
[source's activation chart](https://blog.eleuther.ai/transformer-math/#activations-and-batch-size)
illustrates the same tradeoff for larger models; its red line marks an 80 GB
GPU limit.

## What ZeRO Shards

Ordinary data parallelism gives each GPU a complete copy of the weights,
gradients, and optimizer state. Let `d` be the number of **data-parallel
replicas**, `W` the working-weight bytes, `G` the gradient bytes, `O` the
optimizer-state bytes, and `A` each GPU's activations. ZeRO progressively
partitions parameter state among those `d` ranks:

| Layout | Approximate resident bytes per GPU | What is partitioned |
|---|---|---|
| Replicated | `W + G + O + A` | nothing |
| ZeRO-1 | `W + G + O/d + A` | optimizer state |
| ZeRO-2 | `W + (G + O)/d + A` | optimizer state, gradients |
| ZeRO-3 | `(W + G + O)/d + A + live buffers` | optimizer state, gradients, weights |

ZeRO-3 gathers weights needed for the current computation and frees or reuses
them later. The live gathered weights and communication buffers mean its peak
memory is higher than the partitioned-state term alone. Sharding also adds
communication; the cheapest memory layout is not automatically the fastest run.

The [source article's ZeRO illustration](https://blog.eleuther.ai/transformer-math/#sharded-optimizers),
credited there to the [ZeRO paper](https://arxiv.org/abs/1910.02054), shows
the same progression: first the optimizer state, then gradients, then
parameters are partitioned across data-parallel ranks. Activations remain a
separate budget.

Return to the `P = 1B` layout above, with `d = 8` and no model parallelism.
Here `W = 2 GB`, `G = 2 GB`, and `O = 12 GB` (master weights plus two Adam
moments):

| Layout | Parameter-state calculation | Approximate bytes per GPU before activations |
|---|---|---:|
| Replicated | `2 + 2 + 12` | 16 GB |
| ZeRO-1 | `2 + 2 + 12/8` | 5.5 GB |
| ZeRO-2 | `2 + (2 + 12)/8` | 3.75 GB |
| ZeRO-3 | `(2 + 2 + 12)/8` | 2 GB **plus live buffers** |

Add each GPU's activation and temporary-buffer budget to every row. The table
is a storage estimate, not a guaranteed peak-memory measurement.

## How Data, Tensor, and Pipeline Parallelism Combine

| Dimension | Splits | Main consequence |
|---|---|---|
| Data parallel (`d`) | training examples across complete model replicas | higher token throughput; ZeRO can shard state across these ranks |
| Tensor parallel (`t`) | individual layer computations across GPUs | smaller local weight partitions, with communication inside layers |
| Pipeline parallel (`p`) | groups of layers across GPUs | smaller local layer sets, with in-flight microbatch activations |

If these axes form a regular mesh, `N = d × t × p`; therefore **ZeRO's shard
count is `d = N/(t p)`**, not `N`. For example, 32 GPUs with `t = 4` and
`p = 4` have only `d = 2` full model replicas. For a local microbatch of `b`
examples per replica and `k` gradient-accumulation microbatches, the global
batch is `b × d × k`. Tensor and pipeline ranks working on the same example do
not multiply that batch size.

The [source's 32-GPU diagram](https://blog.eleuther.ai/transformer-math/#sharded-optimizers--3d-parallelism)
draws these as three directions: data parallelism supplies independent examples
to complete model replicas; tensor and pipeline parallelism partition each
replica's computation.

As a first approximation, tensor and pipeline parallelism can reduce local
weight storage toward `W/(t p)` when layers and matrices partition evenly.
Actual peaks depend on unbalanced layers, pipeline microbatches, gathered
parameters, activations, and communication buffers. Measure a representative
step before committing to a cluster size.

## Common Confusions

| Tempting shortcut | Check this instead |
|---|---|
| “Peak TFLOP/s tells me training time.” | Use sustained **training** FLOP/s measured for a similar workload and parallel layout. |
| “`D` is the number of unique tokens on disk.” | Count every token the model processes, including repeated epochs. |
| “TF32 means weights take fewer than four bytes.” | TF32 changes compute precision for FP32 matrix operations; FP32 storage still takes four bytes. |
| “A 7B BF16 model needs exactly 14 GB to serve.” | Add the KV cache, temporary buffers, and any quantization metadata. |
| “Training always costs 16 bytes per parameter.” | Name the weight, gradient, and optimizer dtypes and whether a master copy exists. |
| “Eight GPUs give eight ZeRO shards.” | Only the **data-parallel degree** counts; tensor and pipeline ranks split one replica. |

## Recall Check

1. A dense model has `2B` parameters and trains on `40B` tokens. About how many training FLOPs does `6PD` predict?
1. Why can a model's inference memory rise while its weight file remains unchanged?
1. In the worked `1B`-parameter layout, which three buckets make up the `12 GB` optimizer state?
1. With `N = 32`, `t = 4`, and `p = 2`, how many data-parallel ranks can share ZeRO state?

<details markdown="1">
<summary>Answers</summary>

1. `6 × 2 × 10⁹ × 40 × 10⁹ = 4.8 × 10²⁰` FLOPs. Divide by *sustained aggregate* FLOP/s to estimate seconds.
1. The KV cache grows with the number of cached tokens and the inference batch. Temporary buffers can grow too.
1. The FP32 master weights, Adam first moment, and Adam second moment: `4 + 4 + 4` bytes per parameter.
1. `d = 32/(4 × 2) = 4` data-parallel ranks.

</details>

**One-minute recap:** `6PD` estimates dense-transformer training work; dividing
by sustained aggregate FLOP/s estimates time. Weight bytes are only the start of
an inference budget, because the KV cache grows with context and batch. Training
adds gradients, optimizer state, and activations. Checkpointing spends compute to
reduce activations; ZeRO partitions parameter state across **data-parallel**
ranks. Always state the precision and parallel layout behind a memory number.

______________________________________________________________________

## References

- Anthony, Biderman & Schoelkopf, [*Transformer Math 101*](https://blog.eleuther.ai/transformer-math/)
  (EleutherAI, 2023). Source for the planning framework, historical hardware
  measurements, and linked figures. Its `1.2×` inference heuristic describes
  the article's setting.
- Hoffmann et al., [*Training Compute-Optimal Large Language Models*](https://arxiv.org/abs/2203.15556)
  (2022). The parameter/token tradeoff behind the approximate `D ≈ 20P` result.
- Korthikanti et al., [*Reducing Activation Recomputation in Large Transformer
  Models*](https://arxiv.org/abs/2205.05198) (2022). Activation-memory formulas
  and selective recomputation.
- Rajbhandari et al., [*ZeRO: Memory Optimizations Toward Training Trillion
  Parameter Models*](https://arxiv.org/abs/1910.02054) (2020). The three
  progressive state-partitioning stages.
