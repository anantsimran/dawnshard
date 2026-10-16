# Multi30k EN→DE translation: a scale study

`Multi30kTranslator` is a pre-norm encoder-decoder trained by teacher forcing on
Multi30k's English–German pairs. This page records 27 training runs made between
2026-10-13 and 2026-10-15 (9 arms, 3 seeds each, 25 epochs) that ask what changes when
the model gets bigger. It follows the
[128-wide ablation of 2026-10-13](../2026-10-13/README.md), whose control is the
reference row below. Every number comes from [runs.csv](runs.csv) and
[epochs.csv](epochs.csv), exported from the runs' history files.

The short version: **at 256 wide, a warmup starts to matter, and more capacity does
not.** Doubling the width from 128 to 256 improved best val loss by 0.056. Both warmup
schedules then beat the 256 control by more than any seed's spread, the best by 0.034.
Going further, to 512 wide or to 12+12 blocks, made the model worse at a constant
learning rate. A warmup rescued the 512 model, but it still trails the 256 model with
the same warmup on loss.

**Accuracy disagrees with loss on width.** By best val accuracy, 512 wide with a warmup
is the best model on the page, at 67.0% against 65.6% for 256 with the same warmup, and
the schedule arms keep gaining accuracy for 15 epochs after their loss starts to
climb. Section 5 has the curves; which metric to trust is a question only a
translation metric can settle.

______________________________________________________________________

## Results at a glance

| Arm | Params | Best val loss (mean of 3) | Seed range | Δ vs control | Best epochs | Accuracy at best loss | Best accuracy | Time per epoch |
|---|---|---|---|---|---|---|---|---|
| `warmup_cosine` | 22.1M | 2.077 | 2.064–2.099 | −0.034 | 7, 8, 9 | 62.9% | 65.6% | 118 s |
| `one_cycle` | 22.1M | 2.091 | 2.091–2.092 | −0.020 | 9, 8, 9 | 62.7% | 65.5% | 118 s |
| `d_model=512`, `warmup_cosine` | 72.5M | 2.093 | 2.079–2.105 | −0.019 | 7, 8, 7 | 63.0% | 67.0% | 331 s |
| `d_model=512`, `one_cycle` | 72.5M | 2.102 | 2.093–2.115 | −0.009 | 8, 8, 7 | 62.6% | 66.9% | 329 s |
| `dropout_rate=0.3` | 22.1M | 2.108 | 2.098–2.114 | −0.003 | 10, 10, 12 | 63.8% | 65.6% | 118 s |
| **control (`d_model=256`)** | **22.1M** | **2.111** | **2.106–2.118** | | **6, 6, 6** | **62.7%** | **63.8%** | **117 s** |
| 128 control (2026-10-13) | 7.51M | 2.167 | 2.148–2.182 | +0.056 | 8, 9, 9 | 62.3% | 63.3% | 51 s |
| `dropout_rate=0.0` | 22.1M | 2.187 | 2.178–2.199 | +0.076 | 5, 5, 5 | 61.4% | 61.6% | 115 s |
| `d_model=512` | 72.5M | 2.221 | 2.209–2.234 | +0.109 | 6, 6, 6 | 61.6% | 62.8% | 329 s |
| 12+12 blocks | 36.3M | 2.260 | 2.248–2.274 | +0.149 | 7, 8, 7 | 61.1% | 62.2% | 225 s |

"Best accuracy" is each seed's highest val accuracy over the run, averaged over the
three seeds; it usually comes at a different epoch from the best loss.

The control's three seeds span 0.013, but other arms' seeds span up to 0.035, so the
control's spread is likely an underestimate of the noise. This page counts a Δ as a
result only when the arm's seed range does not overlap the control's.

- **Width helped once.** The 256 control beats the 128 control by 0.056, with no
  overlap between their seed ranges, and peaks two to three epochs sooner.
- **A warmup is worth 0.02 to 0.03 at 256 wide.** Both warmup schedules sit entirely
  below the control's range. At 128 wide the same schedules landed inside noise.
- **Past 256, capacity hurts at a constant learning rate.** 512 wide costs 0.109, and
  12+12 blocks cost 0.149, the worst arm on the page.
- **The warmup recovers the 512 model, but not more.** It improves 512 by about 0.12,
  to 2.093, still behind the 256 model with the same schedule (2.077).
- **Dropout 0.3 matches the control at its best epoch and overfits far more slowly.**
  Its val loss at epoch 25 is 2.49, against the control's 3.20.
- **Ranked by best accuracy, width with a warmup wins.** 512 with either warmup
  reaches 66.9–67.0%, and no seed range overlaps the 256 warmup arms' 65.3–65.6%.

______________________________________________________________________

## Setup

### Task, data, and metrics

As in the [128 study](../2026-10-13/README.md#setup):
[multi30k.py](../../../app/src/training/dataload/multi30k.py) reads Multi30k task 1,
with one byte-level BPE tokenizer per language (10,000 merges, 10,262 tokens), and
pads or truncates each side to `seq_len=32`. Both studies load the same pinned EN and
DE merges files, so their val losses compare directly.

- **Train:** 29,000 pairs.
- **Val:** 1,014 pairs, 15,917 target tokens, the same for every run.
- **Loss:** cross-entropy of the next target token at every non-pad target position,
  with the true previous tokens as input (teacher forcing). Accuracy is top-1 over the
  same positions. Neither measures translation quality; there is no decoding and no
  BLEU.

### Model

[multi30k_translator.py](../../../app/src/training/model/multi30k_translator.py). The
control is the 128 study's control at `d_model=256`: 6 encoder and 6 decoder blocks, 4
heads, an FFN 6× `d_model` wide, dropout 0.1, QK norm on, and LayerNorm. Most of the
parameters are in the embeddings and the output head: 7.9M of the control's 22.1M,
and 15.8M of the 512 model's 72.5M.

### Training

Adam at a peak learning rate of `1e-3`, batch size 64, no weight decay, 25 epochs. The
study ran every arm once before any arm twice, so after the first pass every arm had
one seed. All runs trained on an Apple-silicon GPU (MPS), in about 38 hours of wall
clock. Every run drew its own train seed, recorded in its history file.

25 epochs rather than the 128 study's 30, because every run there peaked by epoch 13.
Every run here peaked by epoch 12.

### The schedules at 25 epochs

`fit` steps the scheduler once per epoch, so the LR changes in steps of one epoch.
Both warmups start at 4% of the peak.

| Epoch | 1 | 3 | 5 | 6 | 7 | 8 | 9 | 10 | 15 | 20 | 25 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `warmup_cosine` (× 1e-3) | 0.04 | 0.42 | 0.81 | 1.00 | 0.99 | 0.98 | 0.95 | 0.90 | 0.58 | 0.21 | 0.01 |
| `one_cycle` (× 1e-3) | 0.04 | 0.25 | 0.69 | 0.88 | 0.99 | 1.00 | 0.98 | 0.95 | 0.61 | 0.19 | 0.00 |

The shapes differ from the 128 study's 30-epoch schedules, which is why each schedule
arm here is compared with its own control rather than with its 128 counterpart.

______________________________________________________________________

## 1. Width: 128 to 256 to 512

| Model | Params | Best val loss | Best epochs | Time per epoch |
|---|---|---|---|---|
| 128 control (2026-10-13) | 7.51M | 2.167 | 8, 9, 9 | 51 s |
| 256 control | 22.1M | 2.111 | 6, 6, 6 | 117 s |
| 512, constant LR | 72.5M | 2.221 | 6, 6, 6 | 329 s |

![Best val loss per arm, one dot per seed, a black tick at each arm's mean, and the 256 control's seed range shaded. The two 256 warmup arms and the 512 warmup_cosine arm sit left of the band; dropout 0.3 and 512 with one_cycle overlap it; the 128 control, dropout 0.0, 512 at a constant LR, and 12+12 blocks sit to its right, 12+12 furthest at 2.26](assets/best_val_by_arm.png)

- **On loss, 256 is the only width step that paid.** It is 0.056 better than 128 for
  three times the parameters, and every seed of it beats every seed of the 128
  control. On accuracy, 512 pays too once it has a warmup (section 5).
- **512 at a constant 1e-3 is worse than 128.** It peaks at epoch 6 in every seed, the
  same as the 256 control, but 0.109 higher.
- **The 512 loss is an optimization problem more than a capacity one.** A warmup
  closes all of its 0.110 gap to the 256 control and lands 0.019 below it (next
  section). A model that was simply too large for 29,000 pairs would not recover by
  changing the first few epochs' learning rate.

______________________________________________________________________

## 2. Schedules at 256 and 512

| Arm | Compared with | Best val loss | Δ | Best epochs | LR at best epoch |
|---|---|---|---|---|---|
| `warmup_cosine` | 256 control | 2.077 | −0.034 | 7, 8, 9 | 0.95–0.99 × peak |
| `one_cycle` | 256 control | 2.091 | −0.020 | 9, 8, 9 | 0.98–1.00 × peak |
| 512, `warmup_cosine` | 512 constant | 2.093 | −0.128 | 7, 8, 7 | 0.98–0.99 × peak |
| 512, `one_cycle` | 512 constant | 2.102 | −0.119 | 8, 8, 7 | 0.99–1.00 × peak |

![Mean val loss by epoch over 3 seeds for four 25-epoch arms. The 256 control bottoms out at 2.11 at epoch 6 and climbs to 3.20 by epoch 25. warmup_cosine starts above 4.0, crosses the control at epoch 7, bottoms out at 2.08 at epoch 8, and flattens to 2.78 as its learning rate decays. d_model=512 bottoms out at 2.22 at epoch 6 and climbs steepest, to 3.56. Dropout 0.3 bottoms out at 2.11 at epoch 10 and climbs slowest, to 2.49](assets/val_curves.png)

- **The gain is the warmup's, not the decay's.** Every schedule run reaches its best
  val loss at epoch 7, 8, or 9, when its LR is still 95–100% of the peak. The cosine
  decay has barely started at the epoch that counts; the warmup epochs are what
  differ from the control. There is no cosine-only arm here to confirm this, so it
  rests on the LR table above.
- **A warmup did nothing at 128 and helps at 256 and 512.** At 128 wide all three
  schedules landed within 0.005 of the control. Here the gain grows with width:
  0.02–0.03 at 256, 0.12 at 512. The usual account is that a wider model's early
  Adam updates at full LR are too large, and a warmup keeps them small until the
  weights settle. That fits this ordering but is not something these runs test.
- **The two warmups land together.** At 256, `warmup_cosine` (linear over 5 epochs)
  beats `one_cycle` (cosine-shaped, peaking at epoch 8) by 0.014; at 512, by 0.009.
  Both pairs' seed ranges overlap, so this page doesn't rank them.
- **Decay shapes the overfitting, not the minimum.** After its best epoch the
  `warmup_cosine` run's val loss rises to 2.78 by epoch 25, against the control's 3.20,
  because the LR falls toward zero and the model stops moving. Its training loss
  falls to 0.036: the schedule runs memorize the training set. Best-epoch loss
  ignores that tail, but accuracy doesn't: it is where the schedule arms reach their
  best accuracy (section 5).

______________________________________________________________________

## 3. Depth: 12+12 blocks

| Arm | Params | Best val loss | Δ vs control | Best epochs | Time per epoch |
|---|---|---|---|---|---|
| 12+12 blocks | 36.3M | 2.260 | +0.149 | 7, 8, 7 | 225 s |

- **Doubling the depth is the worst change on the page.** It costs 0.149 at the
  control's constant LR, more than doubling the width does.
- **It is untested with a warmup.** The 512 result says a constant 1e-3 is too high
  for a bigger model, and depth may behave the same way. A 12+12 arm with
  `warmup_cosine` is the run that would tell depth apart from the learning rate.
- **The 128 study's depth result does not carry over.** There, cutting either side
  from 6 blocks to 2 cost 0.06 to 0.09. Removing depth hurt; adding it, at least at a
  constant LR, hurts too.

______________________________________________________________________

## 4. Dropout at 256

| Arm | Best val loss | Δ vs control | Best epochs | Val loss at epoch 25 | Train loss at epoch 25 |
|---|---|---|---|---|---|
| `dropout_rate=0.0` | 2.187 | +0.076 | 5, 5, 5 | 3.86 | 0.098 |
| control (0.1) | 2.111 | | 6, 6, 6 | 3.20 | 0.203 |
| `dropout_rate=0.3` | 2.108 | −0.003 | 10, 10, 12 | 2.49 | 0.560 |

- **Removing dropout costs less at 256 than at 128.** +0.076 here, against +0.171 at
  128 wide, with the best epoch at 5 in both. The gap shrank rather than grew.
- **More dropout delays the minimum without lowering it.** Dropout 0.3 peaks four to
  six epochs later than the control, at the same loss, and then climbs at less than
  half the control's rate. For a run that keeps the best-loss checkpoint the two are
  equal; for a run that doesn't, 0.3 is much safer. On accuracy it is clearly better
  (section 5).
- **Train loss here includes dropout.** It is measured with dropout active, so the
  three arms' train losses aren't directly comparable with each other.

______________________________________________________________________

## 5. Accuracy and perplexity

### Perplexity: the loss on another scale

Perplexity is the exponential of val loss, so it ranks every arm and every epoch
exactly as loss does, and its minimum falls at the loss minimum: 8.0 for
`warmup_cosine`, 8.3 for the control. What the exponential adds is scale. By epoch 25
the control's perplexity has tripled, to 24.6, and the 512 model's has nearly
quadrupled, to 35.0, while dropout 0.3 sits at 12.0.

![Mean val perplexity by epoch over 3 seeds for five 25-epoch arms. All five bottom out between 8 and 9.3 at epochs 6 to 10. After that, d_model=512 climbs steepest, to 35 by epoch 25, and the 256 control to 25. The 512 and 256 warmup_cosine arms level off near 17 and 16 as their learning rate decays, and dropout 0.3 climbs slowest, to 12. Epoch-1 values of the warmup arms sit above the plotted range](assets/val_perplexity.png)

### Accuracy: a different ranking

| Arm | Best accuracy (mean of 3) | Seed range | Epochs of best accuracy | Epochs of best loss |
|---|---|---|---|---|
| 512, `warmup_cosine` | 67.0% | 67.0–67.1% | 24, 24, 23 | 7, 8, 7 |
| 512, `one_cycle` | 66.9% | 66.7–67.1% | 23, 24, 24 | 8, 8, 7 |
| `warmup_cosine` | 65.6% | 65.5–65.6% | 23, 25, 25 | 7, 8, 9 |
| `dropout_rate=0.3` | 65.6% | 65.5–65.9% | 22, 19, 24 | 10, 10, 12 |
| `one_cycle` | 65.5% | 65.3–65.6% | 24, 24, 23 | 9, 8, 9 |
| **control (256)** | **63.8%** | **63.6–63.9%** | **10, 16, 10** | **6, 6, 6** |
| 128 control (2026-10-13) | 63.3% | 63.1–63.7% | 16, 15, 19 | 8, 9, 9 |
| `d_model=512` | 62.8% | 62.7–62.9% | 20, 20, 13 | 6, 6, 6 |
| 12+12 blocks | 62.2% | 62.1–62.2% | 14, 11, 19 | 7, 8, 7 |
| `dropout_rate=0.0` | 61.6% | 61.4–61.9% | 5, 6, 5 | 5, 5, 5 |

![Mean val accuracy by epoch over 3 seeds for five 25-epoch arms. The 256 control rises fastest, peaks at 63.7% at epoch 10, and holds between 63.0% and 63.5% after. d_model=512 levels off lower, near 62.5%. The two warmup_cosine arms and dropout 0.3 keep rising through epoch 25: 512 with warmup_cosine to 67.0%, 256 with warmup_cosine to 65.6%, and dropout 0.3 to 65.4%](assets/val_accuracy.png)

- **Loss climbs while accuracy holds.** After its loss minimum at epoch 6, the
  control's accuracy keeps rising until epoch 10 and then stays between 63.0% and
  63.5%, while its loss climbs from 2.11 to 3.20. The model's top choice stays as
  good; what grows is its confidence in the tokens it gets wrong, which cross-entropy
  punishes and accuracy ignores. The 128 study's 100-epoch run showed the same split.
- **On accuracy, the decay is what pays.** `warmup_cosine` sits at 62.8% at its loss
  minimum (epoch 8) and gains 2.8 points over the next 17 epochs, as its LR falls from
  0.98 to 0.01 of the peak. Section 2 found the loss gain came before the decay
  started; the accuracy gain comes during it.
- **Width pays on accuracy when it has a warmup.** 512 with `warmup_cosine` reaches
  67.0% against 65.6% for 256 with the same schedule, with no overlap between their
  seed ranges. At a constant LR, 512 stays below the 256 control on accuracy as well
  (62.8% against 63.8%).
- **Dropout 0.3 matches the 256 warmup arms on accuracy without any schedule.** It
  reaches 65.6%, at epochs 19 to 24, with the slowest-climbing loss of any arm.
- **The two rankings pick different checkpoints.** Keeping the best-loss checkpoint
  keeps an epoch 6–9 model; keeping the best-accuracy checkpoint keeps an epoch 20–25
  model that is right more often and badly calibrated. Greedy decoding uses only the
  top token at each step, which argues for accuracy, but teacher-forced accuracy is
  not decoded accuracy either. Only a translation metric on both checkpoints settles
  which ranking to use.

"Best accuracy" is a maximum over a noisy series, so it flatters an arm whose accuracy
jumps around. For the schedule arms this barely matters: their final-epoch accuracy is
within 0.1 points of their best. The control's final epoch is at 63.2%, 0.6 below its
best.

______________________________________________________________________

## Predictions against results

The predictions were written into the run plan before the runs started.

| Arm | Prediction | Result |
|---|---|---|
| control | Slightly better than 128's 2.167, with an earlier best epoch | Confirmed: 2.111, best epoch 6 against 8–9 |
| 12+12 blocks | Inside noise | Wrong: +0.149, the worst arm |
| `d_model=512` | Inside noise or worse, peaking by epoch 5 | Partly: worse by 0.109, but peaking at epoch 6 |
| `one_cycle` | Inside noise, as at 128 | Wrong: −0.020, with no overlap |
| `warmup_cosine` | Inside noise, as at 128 | Wrong: −0.034, the best arm |
| 512, `one_cycle` | Better than 512 by more than the noise floor | Confirmed: −0.119 against 512 |
| 512, `warmup_cosine` | Better than 512, close to 512 `one_cycle` | Confirmed: −0.128 against 512, 0.009 from `one_cycle` |
| `dropout_rate=0.0` | Worse than at 128 (+0.171), best epoch by 4 | Wrong: +0.076, best epoch 5 |
| `dropout_rate=0.3` | Better than the control | Not established: −0.003 |

Four predictions were wrong, and they share one assumption: that what held at 128 would
hold at 256. It failed for the schedules, for depth, and for the cost of removing
dropout.

______________________________________________________________________

## What this doesn't show

- **Nothing here measures translation quality.** Teacher-forced next-token loss and
  accuracy reward a good German language model as much as a good translator. A BLEU
  score from greedy or beam decoding on the test split is the missing measure.
- **Whether the warmup or the decay helps is inferred, not measured.** There is no
  cosine-only arm, so the claim that the warmup carries the loss gain rests on the LR
  being 95–100% of the peak at every schedule arm's best epoch.
- **Loss and accuracy disagree, and nothing here says which is right.** By loss, 256
  with a warmup is best; by accuracy, 512 with a warmup is. The predictions were
  written against loss, which is why the table above judges them on it.
- **Depth is confounded with the learning rate.** 12+12 ran only at a constant 1e-3,
  the setting that also hurt 512.
- **One learning rate.** Every arm peaks at 1e-3. A lower constant LR for the bigger
  models might match a warmup, and this study can't say.
- **Three seeds bound the noise roughly.** The control's 0.013 spread is narrower than
  most arms' spread, which is why the page asks for non-overlapping ranges.
- **Post-norm is not an arm.** `EncoderBlock` and `DecoderBlock` hardcode the pre-norm
  order.

______________________________________________________________________

## Next runs

In order of what they would settle:

1. **12+12 blocks with `warmup_cosine`**, to separate depth from the learning rate.
1. **A cosine-only arm at 256**, to confirm that the warmup, not the decay, carries the
   schedule gain.
1. **BLEU from greedy decoding** on the test split for each arm's best-loss and
   best-accuracy checkpoints, to settle which of the two rankings tracks translation
   quality.

______________________________________________________________________

## Data and reproducing

### Files

- [runs.csv](runs.csv): one row per run (27 rows), with the same columns as the
  [128 study's](../2026-10-13/runs.csv). The `arm` column names what differs from the
  256 control.
- [epochs.csv](epochs.csv): one row per epoch per run (675 rows). Join it to the runs
  file on `run_id`.
- `app/history/<run_id>.json`: the full history file for each run, with an attention
  capture per epoch under `app/history/attention_probes/<run_id>/`. These are gitignored
  and exist only on the machine that trained them. The history files don't record the
  config, so the `arm` column comes from the run log, which names each run's config
  next to its run id.

The `experiment` column is `scale_25ep` for every run.

### Code

All 27 runs ran commit `4082134` and record it. To rerun them, from the repo root:

```bash
caffeinate -is uv run python app/src/training/experiments/multi30k_translator_ablation.py scale
```

Later commits encode the loader's sentences with `bpe.encode_many`, which returns the
same ids as the per-sentence `encode` these runs used.

### Known issues in the recorded data

None found. All 27 runs finished with a history file, and the log shows no errors or
restarts.
