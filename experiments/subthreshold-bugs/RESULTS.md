# Sub-threshold bugs: results

CPU only (4 vCPU, gloo, 2 ranks). Tiny Qwen3 (2 layers, d_model 64, about 1e5
parameters), bf16 autocast with fp32 weights, AdamW at lr 1e-3 and weight decay 0.1 on
matrices, 200 steps of 8 x 32 tokens from a random bigram chain. Command:
`uv run python app/src/training/experiments/subthreshold.py`; it ran twice with
bitwise-identical CSVs. Design and claims are in [DESIGN.md](DESIGN.md).

______________________________________________________________________

## The drift test found a real bug in T7's tensor-parallel plans

The first run used the hand TP plan as shipped in T7 as a bug-free control. The drift
test rejected it on all three seeds (ratio 1.11 to 1.21 at 200 steps, p = 0.0002),
while every DDP and ZeRO-2 control passed. The cause is a missing all-reduce. The
QK-norm gains are replicated, but each rank applies them only to its own heads, so
each rank's gradient covers half the heads and nothing sums it. Measured against the
dense model in fp32, those four tensors' gradients are 62% to 82% off, and every other
tensor matches to 6e-7. T7's DTensor plan has the identical gap, which is why T7's
check that the hand and DTensor plans agree passed. T7's 50-step loss parity passed
too: a 2.6e-4 loss difference that RESULTS.md attributed to Adam amplifying
reordering noise. `tp_fixed`, which sums those gradients with a tensor hook, matches
the dense gradients to 6e-7 and passes the drift test on every seed.

The study now keeps the shipped plan as `tp`, a found bug, and uses `tp_fixed` as the
TP control. The library plans in `model/qwen3.py` are unchanged.

## Detection ([detection.csv](detection.csv))

Each cell is the first step a check flags, or a dash if it never does in 200 steps.
Per-step checks use limits from the bug-free candidates on seed 0
([limits.csv](limits.csv)); the drift test uses none and is evaluated at steps 10,
25, 50, 100, and 200 with α = 0.01. Seeds 1 and 2; one cell when both agree.

| Candidate | Loss parity | Whole-model update | Per-tensor weight | Per-tensor update | Drift test |
|---|---|---|---|---|---|
| DDP (naive, bucketed, small buckets) | - | - | - | - | - |
| ZeRO-2 | - | - | - | - | - |
| `tp_fixed` (correct) | - | **0** | **0** | **0** | - |
| `tp` (shipped, QK-norm bug) | 153 / - | 0 | 0 | 0 | 200 / 100 |
| `grad_bf16` | - | - | - | - | - |
| `microbatch_bf16` | - | - | - / 0 | - / 0 | - |
| `bf16_master` | 18 / 20 | 0 | 0 | 0 | 10 |
| `decay_on_norms` | 42 / 44 | - | 0 | 0 | 10 |
| `adam_step_offset` | 2 | 0 | 0 | 0 | 10 |

Bold marks false alarms on a correct implementation. The three DDP wrappers compute
the same thing bit for bit, so the bug-free evidence is six distinct runs: DDP, ZeRO-2,
and `tp_fixed` on two seeds each.

- **The drift test raised no false alarm and found every biased bug.** It flagged
  the three biased plants at the first checkpoint, step 10, and the shipped TP plan by
  step 100 or 200. Its lowest p-value on a correct run was 0.046 (DDP, seed 2, at
  step 10).
- **The per-step checks cannot tell the correct TP plan from the buggy one.** With
  tolerances fitted on seed 0, all three flag `tp_fixed` at step 0 on both held-out
  seeds. The correct TP plan's own noise moves from seed to seed by more than the
  limits allow.
- **Loss parity caught the three strong plants late and the TP bug once in two.**
  It flagged `decay_on_norms` at steps 42 and 44 against the drift test's 10, and the
  TP bug only on seed 1, at step 153.
- **Two bugs were caught by no check, and they do not change training.**
  `grad_bf16` and `microbatch_bf16` are unbiased roundings. Their final loss gaps
  (-2.4e-4 to 2.7e-5) sit inside the bug-free range (-4.5e-4 to 8.3e-5). A test
  built on bias is blind to them by design, and here that is the right answer.

## How much margin a per-step tolerance would need

Each value is the largest ratio of a run's statistic to its limit over the 200 steps
(above 1 flags). For a correct candidate, the larger of its two seeds (the run a
tolerance must pass); for a bug, the smaller (the run a tolerance must still catch).

| Candidate | Loss parity | Whole-model update | Per-tensor weight | Per-tensor update |
|---|---|---|---|---|
| DDP, ZeRO-2 (correct) | 0.95 | 0.45 | 0.91 | 0.94 |
| `tp_fixed` (correct) | 0.80 | 1.04 | 29.9 | 21.2 |
| `tp` (QK-norm bug) | 0.68 | 1.08 | 177 | 129 |
| `microbatch_bf16` | 0.61 | 0.43 | 0.91 | 0.94 |
| `bf16_master` | 254 | 3.22 | 872 | 205 |
| `decay_on_norms` | 18.6 | 0.43 | 29.4 | 77.5 |
| `adam_step_offset` | 5.31 | 3.29 | 75.2 | 52.5 |

- **A per-tensor update tolerance works here only with a margin picked after the
  fact.** Passing `tp_fixed` needs a margin above 21.2, and catching
  `adam_step_offset` needs one below 52.5. The window exists on this data, but a test
  author would have to choose it without knowing either number.
- **The per-tensor weight check has no working margin.** The correct TP plan reaches
  29.9 and `decay_on_norms` only 29.4, so any tolerance that passes the first misses
  the second.
- **The whole-model update check cannot separate the TP plans at any margin.** The
  correct plan reaches 1.04 and the buggy one 1.08, because the bug lives in four
  tensors of 16 elements each, and a whole-model norm averages it away.

## Drift ratio over time ([drift.csv](drift.csv))

| Candidate | K = 10 | 25 | 50 | 100 | 200 |
|---|---|---|---|---|---|
| DDP, ZeRO-2, `tp_fixed` (range) | 0.999 to 1.002 | 0.999 to 1.002 | 0.999 to 1.002 | 0.998 to 1.002 | 0.998 to 1.002 |
| `tp` (seed 2) | 0.991 | 1.014 | 1.036 | 1.063 | 1.214 |
| `bf16_master` (seed 1) | 2.75 | 7.70 | 21.5 | 45.0 | 70.0 |
| `decay_on_norms` (seed 1) | 3.18 | 11.7 | 31.2 | 76.8 | 174 |
| `adam_step_offset` (seed 1) | 2.01 | 1.93 | 1.91 | 1.98 | 2.09 |

- **A constant bias makes the ratio grow linearly, as Claim 3 predicts.**
  `decay_on_norms` grows about linearly from step 25 on, so its bias carries a steady
  share of each step's error energy, close to ρ ≈ 0.9 by the end.
- **A fading bias stops the ratio growing.** `adam_step_offset` holds near 2. The
  bias-correction error is large in the first steps and fades as the correction
  factors approach 1, so the early steps carry the signal and later steps add noise.
- **The TP bug's signal is weak but steady.** Its bias carries ρ ≈ 1e-3 of the error
  energy. If the noise were isotropic over 1e5 parameters, Claim 3 predicts detection
  in about 10 steps; it took 100 to 200. Detection at K ≈ 150 implies an effective
  noise dimension of a few hundred, so the per-step noise lives in far fewer directions than
  the parameter count, and the isotropic estimate is optimistic by about 10x in steps.

## Claims against the data

| Claim (DESIGN.md) | Verdict |
|---|---|
| 1. A weight-relative tolerance of bf16 size cannot see bf16 storage | Holds for the matrices, whose bug-free weight limits are 2.7e-3 to 4.9e-3, the size of `u`. The per-tensor weight check still caught `bf16_master`, through the norm gains, whose limits are under 1.1e-5. |
| 2. The drift test's false-alarm rate is at most α | No false alarm in six distinct bug-free runs at five checkpoints each. That is consistent with α = 0.01 but too few runs to measure the rate; the closest call is DDP seed 2 at step 10, p = 0.046. |
| 3. A bias far below the per-step noise is found in few steps | Holds in form: linear growth, and detection of a bias carrying 0.1% of the error energy. Its isotropic step count is optimistic by about 10x for this model. |

## Not run here

The study does not cover:

- larger models, real text, GPUs, or NCCL
- bugs that live only in the optimizer state, which the restart overwrites (DESIGN.md, Routes not taken)
- reproducing the DeepSpeed issues in DeepSpeed itself
- a measured false-alarm rate, which needs hundreds of bug-free runs
