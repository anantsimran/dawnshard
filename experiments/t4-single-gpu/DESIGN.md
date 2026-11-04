# T4: single-GPU performance, design notes

This ticket adds a performance model
([perf.py](../../app/src/training/train/perf.py)), a chunked cross-entropy
([losses.py](../../app/src/training/train/losses.py)), a roofline plot
([roofline.py](../../app/src/training/viz/roofline.py)), gradient accumulation,
autocast and MFU as loop policy, activation checkpointing in the Qwen3 decoder, and
performance flags on its entry point. The ledger is in [RESULTS.md](RESULTS.md).

______________________________________________________________________

## Predictions from a shape, not a model

`perf.py` works from a `ModelShape`, a handful of integers any decoder config fills
in (`model_shape` does it for Qwen3). FLOPs per token, MFU, and memory are functions
of that shape, so a row can be predicted before the model is built, which is the
point of the ledger.

- **FLOPs = 6N + 12 x layers x heads x head_dim x seq.** N counts weights that take
  part in a matmul: a tied head counts once, an untied embedding lookup does not. A
  test checks the 6N part exactly against `torch.utils.flop_counter` on a real
  model; the counter has no formula for the CPU attention kernel, so the attention
  term is checked by hand.
- **Memory term by term:** weights, grads, Adam moments, activations (the tensors
  each layer saves for backward under a fused attention kernel), and logits (twice:
  forward and gradient). Static memory is also measured, from the tensors
  themselves, and the tests check that the two agree.

**Pros.** Every prediction is a few lines of arithmetic that can be read and
disputed. **Cons.** The activation count is a model of what a fused attention kernel
and autograd save; it ignores allocator slack, temporaries, and the bf16 weight
copies autocast caches. The ledger's "why it missed" column is where those show up.

**Alternative considered.** Measuring instead of predicting (run, read
`max_memory_allocated`). It gives the number but not the reason, and the ticket's
exercise is the reason.

## Accumulation and autocast are policy on `EpochSpec`

`accumulation_steps` and `autocast_dtype` are fields of `EpochSpec`; `train_step`
reads them. Accumulation zeroes on every k-th batch, scales each loss by 1/k, and
steps after the k-th, so the summed gradient is the mean over the group and the
global batch stays fixed while the micro-batch changes.

- **Pros.** The loop's API stays one `fit`; the change is two fields with defaults
  that reproduce the old behavior, so every existing caller is unchanged.
- **Cons.** A loader whose length is not a multiple of k leaves a partial group
  unstepped, by design rather than by a silent short step. DDP's `no_sync` during
  accumulation is not handled here; that is T5's.

**Alternative considered.** A custom `StepFn` per feature. It keeps `train_step`
smaller, but `fit` hard-wires `train_step`, so every feature would need its own `fit`.

## MFU in history only when asked

`fit` adds `targets_per_s` and `mfu` to an epoch record only when the spec sets
`flops_per_target` (and `peak_flops` for MFU). Runs that do not ask keep their
history format, so old and new history files still compare. The epoch number
includes warmup; `StepTimer`, a train probe, gives the steady state after 20 steps,
timed with a CUDA synchronize at each optimizer step.

## Chunked cross-entropy through `torch.utils.checkpoint`

`ChunkedCrossEntropy` takes hidden states and reads the head weight; each chunk's
logits are computed and reduced inside `checkpoint`, so they are freed after forward
and recomputed in backward. Equal to `F.cross_entropy` in loss and gradients within
1e-6 in float64.

- **Pros.** Twenty lines, no custom autograd function, works with a tied head and
  with `ignore_index`.
- **Cons.** One extra head matmul per chunk in backward. A fused kernel (Liger's
  fused linear cross-entropy) computes the gradient in the forward pass and avoids
  the recompute, but is a CUDA/Triton dependency.

The model opts in with `output="hidden"`; the default still returns logits, so
nothing else changes.

## Activation checkpointing in the model

`Qwen3(checkpointing="full")` wraps each layer in `checkpoint`; "selective" wraps
only the MLP, the largest per-layer term (four `intermediate_size`-wide tensors).
Both use `use_reentrant=False` and act only in training mode. A test checks that all
three modes give identical gradients.

## Roofline

`viz/roofline.py` times five kernels (MLP up-projection matmul, causal SDPA, RMSNorm,
the SwiGLU product, an AdamW step) with closed-form FLOPs and bytes, and plots
achieved FLOP/s against intensity under the compute and bandwidth roofs. Peak and
bandwidth are arguments, because the right numbers are the device's dense ones and
must be looked up and cited, not guessed.

## Not built

Per-phase step-time split (data, forward, backward, optimizer, idle) is read from a
`profiled_fit` trace, not computed. Traces of steps 100 to 105 specifically would
need a profiler schedule; `profiled_fit` traces whole epochs.

## Routes not to take

- **Optimizing before predicting.** Without a written prediction there is nothing to
  learn from the measurement; every ledger row starts with one.
- **Timing CUDA work without a synchronize.** Kernels are asynchronous; the clock
  measures launch, not execution.
- **MFU against a sparse or fp16-accumulate peak.** It halves the number; use the dense
  bf16 peak of the exact part, with its source.
- **Leaving out the attention FLOPs, or counting the embedding lookup as a matmul.**
  The first understates MFU by about 40% at 2048 tokens; the second overstates it.
  A tied LM head is a real matmul.
- **Expecting bf16 autocast to save static memory.** It keeps fp32 weights, grads, and
  Adam state; it saves activations and time.
- **Materializing full fp32 logits for a 150K vocabulary.** It is the largest single
  activation; compute the loss in chunks from hidden states.
- **Activation checkpointing as a default.** It costs about a third more compute and
  only pays if the freed memory buys a larger micro-batch.
- **Changing the global batch while tuning.** Every row must train the same thing;
  change the micro-batch and the accumulation together.
