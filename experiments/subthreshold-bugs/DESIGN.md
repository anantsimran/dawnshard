# Sub-threshold bugs: design notes

This study asks whether a parity test can catch a bug in a parallel training
implementation when the bug's error in any single step is smaller than the
floating-point noise of a correct implementation. It adds one library module,
[parity.py](../../app/src/training/parallel/parity.py), and one study,
[subthreshold.py](../../app/src/training/experiments/subthreshold.py), on top of the
T5 to T7 code: `NaiveDDP`, `BucketedDDP`, `Zero2AdamW`, and the hand tensor-parallel
plan. Numbers are in [RESULTS.md](RESULTS.md).

______________________________________________________________________

## The problem

Two parity checks are in common use, and each has a blind spot.

- **Loss parity** trains the candidate and a one-process reference from the same init
  and compares losses (T7's `tp_pp.py parity`). Reordered sums make the runs differ in
  the last bits, and training amplifies those bits: the T7 TP run starts 4.8e-7 apart
  and ends 2.6e-4 apart after 50 steps with no bug. A tolerance loose enough for that
  drift also passes small bugs.
- **Single-iteration differential testing** runs one step of each and compares tensors
  against a tolerance derived from round-off analysis (TTrace, arXiv 2506.09280). It
  removes the drift, but sees one step. Its authors name the gap: it can miss "bugs
  that cause deviations at the magnitude of numerical errors in each SGD iteration,"
  whose accumulated effect still matters, and they report finding no such bugs.

Reports of that kind of bug exist. DeepSpeed #6351 (open since 2024) is a ZeRO-2 run
whose convergence depends on `reduce_bucket_size` and that only fp32 gradient
accumulation fixed; DeepSpeed #8913 is an extra bf16 rounding between an fp32
reduction and fp32 accumulation; DeepSpeed #5307 and #6640 are updates that vanish
because weights are kept in bf16. The question is a test that separates such bugs from
noise without a hand-set tolerance.

## The check: restart every step, then test across steps

Each step, the candidate loads the reference's weights and Adam moments (sharded as
the candidate holds them), both take one step on the same batch, and the step records
the error in the update:

`error_k = (W_cand_after - W_cand_before) - (W_ref_after - W_ref_before)`

The restart means no error compounds: step k's error depends on step k alone, so
training's chaotic amplification never enters. The drift test then asks one question
of the K recorded errors: do they point the same way? It needs no tolerance.

- **The drift ratio** is `||Σ_k error_k||² / Σ_k ||error_k||²`. For errors that are
  independent and zero-mean, the cross terms cancel in expectation and the ratio is
  about 1: their sum grows like √K. A biased bug adds the same direction every step, so
  its sum grows like K and the ratio grows linearly.
- **The p-value** comes from flipping signs. Under the null hypothesis each error is
  symmetric about zero, so flipping the sign of any subset leaves the joint
  distribution unchanged. The observed ratio is compared with the ratios of 4,000
  random sign patterns, computed from the K x K Gram matrix of the errors, so the test
  never stores more than one copy of each error.

## The claims, and why they hold

### Claim 1: a weight-relative tolerance of bf16 size cannot see a bf16 storage bug

Let `u = 2^-8`, the unit roundoff of bf16. Round-to-nearest changes any value by at
most `u` times its magnitude, so a candidate whose weights are the bf16 rounding of the
reference's satisfies `||W_cand - W_ref|| <= u ||W_ref||`. The weight-relative error of
a per-tensor check is therefore at most 3.9e-3 on every tensor and every step,
whatever the bug does to training, and any tolerance at least that large, which is
what a tolerance derived for a bf16 recipe is, passes it.

The update is different. When weights are stored in bf16 and an update `Δ_i` is
smaller than half the spacing of bf16 numbers at `W_i`, round-to-nearest returns `W_i`
and the update is lost. For a weight of 0.02 the spacing is 2^-13, about 1.2e-4, so
every Adam update below 6.1e-5 vanishes: the update-relative error is 1 on those
coordinates while the weight-relative error is about 3e-3. A per-step check should
compare updates, not weights, and the study records both. A tolerance calibrated
below `u` sees the bug where the weights are large and change slowly (here the norm
gains, whose bug-free weight error is under 1.1e-5), so the claim is about tolerances
of bf16 size, not about every tolerance.

### Claim 2: the drift test has false-alarm probability at most α

Assume the per-step errors `e_1, ..., e_K` are independent and each is symmetric
about zero. For any fixed signs `s`, `(s_1 e_1, ..., s_K e_K)` then has the same
distribution as `(e_1, ..., e_K)`. Draw `B` sign patterns uniformly and independently.
The observed statistic and the `B` flipped statistics are exchangeable, so the rank of
the observed one among the `B + 1` is uniform (ties broken against rejection), and the
Monte Carlo p-value `(1 + #{flipped >= observed}) / (B + 1)` satisfies
`P(p <= α) <= α` for every `K`, every dimension, and every noise scale. This is the
standard randomization-test argument (Lehmann and Romano, *Testing Statistical
Hypotheses*; the `+1` follows Phipson and Smyth, 2010). The two assumptions are where
the claim can fail, and the bug-free candidates measure whether they do. Restart makes
each step's error a function of that step's batch and the reference state alone, which
is close to independence but not equal to it, since consecutive reference states are
close. Symmetry is what reordering noise is expected to have but is not guaranteed to
have (arXiv 2511.00025 finds matmul rounding noise structured).

### Claim 3: a bias far below the per-step noise is found in few steps

Model each error as `e_k = μ + ξ_k`: a fixed bias `μ` plus zero-mean noise with
`E||ξ_k||² = σ²` and `E<ξ_i, ξ_j>² = σ⁴ / d_eff` for `i ≠ j`, where `d_eff` is the
effective dimension of the noise (the parameter count, if the noise is isotropic). Let
`ρ = ||μ||² / (||μ||² + σ²)`, the share of each step's error energy the bias carries.

- The observed ratio has mean about `1 + (K - 1) ρ`.
- The sign-flipped ratios have mean 1 and standard deviation about
  `√2 · sqrt(ρ² + 1 / d_eff)`.

So the test rejects once `(K - 1) ρ` exceeds `z_α √2 sqrt(ρ² + 1/d_eff)`, with `z_α`
about 2.3 for α = 0.01. Two regimes follow:

- When `ρ >> 1 / sqrt(d_eff)`, a handful of steps suffices: `K > 1 + 3.3` for α = 0.01.
- When `ρ << 1 / sqrt(d_eff)`, the test needs `K ≈ 3.3 / (ρ sqrt(d_eff))` steps.

*Worked example.* The tiny model here has about 1e5 parameters. Suppose a bug's bias
carries 0.1% of each step's error energy (`ρ = 1e-3`) and the noise is isotropic. Then
`1 / sqrt(d_eff) ≈ 3.2e-3 > ρ`, and the test needs about `3.3 / (1e-3 · 316) ≈ 10`
steps. A per-step check comparing `||e_k||²` against the noise level would need `ρ`
above `z_α sqrt(2 / d_eff) ≈ 1e-2`, ten times larger, and it would also need `σ²` to
be known, which in practice means a tolerance calibrated per model and per recipe. The
drift test is self-normalizing: it compares the errors with each other.

This is an approximation (it treats the Gram entries as Gaussian), not a theorem; the
unit tests in [test_parity.py](../../app/tests/test_parity.py) check both regimes on
synthetic errors, and the study checks it on real ones.

## What the study plants

All bugs go into `BucketedDDP` with 2 ranks and 2 micro-batches per rank, under bf16
autocast with fp32 weights and AdamW: the mixed-precision recipe, where correct
implementations already disagree at bf16 scale each step.

| Bug | What it does | Expected bias | Analogue |
|---|---|---|---|
| `grad_bf16` | rounds the averaged gradient to bf16 before AdamW | none: rounding is unbiased | DeepSpeed #8913 |
| `microbatch_bf16` | accumulates micro-batch gradients in bf16 | none at 2 micro-batches | DeepSpeed #6351 |
| `bf16_master` | keeps weights in bf16 although the recipe says fp32 | updates shrink (stagnation) | DeepSpeed #5307, #6640 |
| `decay_on_norms` | applies weight decay to the norm gains the recipe excludes | gains shrink every step | a common config bug |
| `adam_step_offset` | runs Adam's step counter one ahead | bias correction too small | DeepSpeed #8910 |

One more candidate is a bug found, not planted: the hand TP plan as T7 ships it
(`tp`). Its QK-norm gains are replicated, but each rank applies them to its own heads
only, so each rank's gradient is a partial sum over half the heads, and nothing
all-reduces it. Those four tensors' gradients are 62% to 82% off; every other tensor
matches the dense model to 6e-7 in fp32. T7's DTensor plan has the same gap (the same
four errors to four digits), so T7's check that the two plans agree could not catch
it, and its 50-step loss parity passed it. TorchTitan's transformers backend declares
these gains as replicated DTensors on the TP mesh, so DTensor reduces their gradient
(`hf_sharding.py`: "Q/K norms (Qwen3) — weight replicated, activations stay
heads-sharded"). `tp_fixed` adds a tensor hook that all-reduces each gain's gradient,
and serves as the TP control.

The bug-free candidates are `NaiveDDP`, `BucketedDDP` with one bucket and with 0.01
MB buckets, `Zero2AdamW`, and `tp_fixed`. On the calibration seed they set the
per-step limits: the largest loss difference and whole-model update-relative error
any of them reached, and per tensor the largest weight- and update-relative error.
That is the tightest tolerance a per-step check could use without failing those runs;
the evaluation seeds then show whether it holds for correct runs it was not fitted to.

## Choices, and what each buys

- **Restart from the reference, not free running.** Free running mixes a bug's effect
  with amplified noise, which is the loss-parity problem. Restart also makes the
  steps independent, which is what Claim 2 needs.
- **Restart the Adam moments too.** If only weights were restarted, the candidate's
  moments would drift from the reference's with the noise, and the moments carry
  memory across steps (β2 = 0.999 keeps about 1,000 steps), so the per-step errors
  would be correlated and the test's false-alarm guarantee would not hold.
- **Shard-aware dot products.** `flatten_shards` scales replicated tensors by
  `1 / sqrt(world)`, so a dot product summed over ranks counts every element once,
  for any mix of sharded (TP) and replicated (DDP, norms) tensors. The drift test then
  needs one `all_reduce` of a K x K Gram matrix, never the full errors.
- **The Gram matrix, not the errors, for the null.** Every flipped ratio is
  `sᵀ G s / trace(G)`, so 4,000 sign patterns cost 4,000 small matrix-vector
  products.
- **One limit per tensor across all bug-free candidates.** A test suite sets one
  tolerance per check, and the TP candidate, whose shards round differently from the
  dense layers, sets most of them. Calibrating on the candidates assumes they are
  correct: the first run of this study calibrated on the shipped TP plan, which is
  how a buggy implementation raises the tolerance it is checked against. The drift
  test has no such input.

## Routes not taken

- **A tolerance on the free-running loss.** It is what T7 did. It must cover the
  amplified drift of correct implementations, which is the drift the bugs hide in.
- **A tolerance from machine epsilon.** TTrace's analysis shows this is too tight
  once errors accumulate through layers; the bug-free candidates here reach
  update-relative errors of a few percent each step under bf16 autocast.
- **A t-test on a scalar projection of each error.** Projecting onto the reference
  update loses any bias orthogonal to it (weight decay on the gains is a fixed
  direction, not along the update), and a t-test needs a Gaussian assumption that the
  sign-flip test does not.
- **Bias through the optimizer state.** A bug that corrupts Adam's moments without
  touching this step's update shows up only one step later, and restart overwrites
  the moments first. Comparing the moments after each step, with their own drift
  test, would cover it; this study does not.

## Limits

The model is tiny (about 1e5 parameters, 2 layers), the data is a synthetic bigram
chain, everything runs on CPU with gloo, and 200 steps is short. Real bugs from the
issue trackers are reproduced here as plants in this repo's own implementations, not
in DeepSpeed itself.
