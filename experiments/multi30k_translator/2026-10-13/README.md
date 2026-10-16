# Multi30k EN→DE translation: a component, schedule, and language ablation

`Multi30kTranslator` is a pre-norm encoder-decoder trained by teacher forcing on
Multi30k's English–German pairs. This page records a 30-run ablation made on
2026-10-13 (10 arms, 3 seeds each, 30 epochs) and the 100-epoch control run before it
that set the length. Every number below comes from [runs.csv](runs.csv) and
[epochs.csv](epochs.csv), exported from the runs' history files.

The short version: **regularization and decoder depth matter, and the learning-rate
schedule does not, at this size.** Every run reached its best val loss by epoch 13.
Removing dropout cost 0.171, cutting the decoder to 2 blocks cost 0.092, and all three
schedules landed inside the control's own seed spread.

______________________________________________________________________

## Results at a glance

| Arm | Params | Best val loss (mean of 3) | Seed range | Δ vs control | Best epochs | Val accuracy |
|---|---|---|---|---|---|---|
| `cosine` | 7.51M | 2.162 | 2.154–2.175 | −0.005 | 8, 8, 9 | 62.3% |
| `one_cycle` | 7.51M | 2.164 | 2.160–2.170 | −0.003 | 13, 11, 13 | 62.1% |
| `warmup_cosine` | 7.51M | 2.165 | 2.146–2.183 | −0.002 | 10, 10, 11 | 61.7% |
| **control** | **7.51M** | **2.167** | **2.148–2.182** | | **8, 9, 9** | **62.3%** |
| `norm="rmsnorm"` | 7.51M | 2.180 | 2.172–2.195 | +0.013 | 8, 7, 8 | 61.6% |
| `qk_norm=False` | 7.51M | 2.189 | 2.177–2.205 | +0.022 | 9, 8, 9 | 61.7% |
| `num_encoders=2` | 6.46M | 2.225 | 2.213–2.235 | +0.058 | 8, 8, 7 | 61.3% |
| `num_decoders=2` | 6.19M | 2.259 | 2.247–2.274 | +0.092 | 9, 8, 9 | 61.3% |
| `dropout_rate=0.0` | 7.51M | 2.338 | 2.330–2.348 | +0.171 | 5, 5, 5 | 59.1% |
| `src_lang="cs"` | 7.51M | 2.641 | 2.636–2.651 | +0.474 | 9, 8, 8 | 54.9% |

The control's three seeds span 0.034. That spread is the noise floor: a Δ inside it,
or an arm whose seed range overlaps the control's, is not a result.

- **Every run overfits early.** The best epoch is 5 to 13 in all 30 runs. The model has
  7.5M parameters and the training set is 29,000 sentence pairs.
- **Dropout is the largest architectural effect.** Without it the model peaks at epoch
  5 and finishes 0.171 worse.
- **The decoder's depth matters more than the encoder's.** Cutting either side to 2
  blocks costs more than the noise floor, and the two arms' seed ranges don't overlap.
- **No schedule beats a constant learning rate.** All three are within 0.005 of the
  control, inside its spread.
- **A Czech source costs 0.474**, the largest Δ on the page, for reasons that are
  partly about the data rather than the language.

______________________________________________________________________

## Setup

### Task and data

The loader, [multi30k.py](../../../app/src/training/dataload/multi30k.py), reads
Multi30k task 1, whose four languages are line-aligned translations of the same
captions. Each language has its own byte-level BPE tokenizer with 10,000 merges, a
10,262-token vocabulary, trained on that language's train split only. Each side is
padded or truncated to `seq_len=32`; the source and the target input start with
`BOS_ID`, and the target output is the target shifted one left and ended with
`EOS_ID`.

- **Train:** 29,000 pairs.
- **Val:** 1,014 pairs, the same for every run.
- **Loss:** cross-entropy of the next target token at every non-pad target position,
  with the true previous tokens as input (teacher forcing). Perplexity is its
  exponential, and accuracy is top-1 over the same positions
  ([masked_token_metrics.py](../../../app/src/training/metrics/masked_token_metrics.py)).
  None of the three measures translation quality; there is no decoding and no BLEU.

### Model

[multi30k_translator.py](../../../app/src/training/model/multi30k_translator.py): a
source and a target embedding, dropout, one fixed sinusoidal position table, a stack
of pre-norm `EncoderBlock`s, a norm over the encoder output, a stack of pre-norm
`DecoderBlock`s with causal self-attention and cross-attention to the encoder, a final
norm, and a linear head over the target vocabulary. The control is `d_model=128`, 6
encoder and 6 decoder blocks, 4 heads, an FFN 6× `d_model` wide, dropout 0.1, QK norm
on, and LayerNorm: the best setting the
[Multi30k predictor's 2026-09-29 study](../../multi30ken_predictor/2026-09-29/README.md)
found, borrowed because the translator had no sweep of its own.

### Training

Adam at a learning rate of `1e-3`, batch size 64, no weight decay, 30 epochs. `fit`
steps a scheduler once per epoch, so every schedule is sized in epochs. All runs
trained on an Apple-silicon GPU (MPS), at about 50 s per epoch for the full-depth
arms. Every run drew its own train seed, recorded in its history file.

______________________________________________________________________

## 1. Setting the length: one control run over 100 epochs

The plan was 100 epochs per run. The first control run (`2ec412ae`) reached its best
val loss, 2.178, at epoch 9, and climbed for the remaining 91 epochs, to 4.08 by the
end. Over the same stretch its val accuracy held at 62–63% while train accuracy rose to
96%. The model kept getting its wrong predictions more confidently wrong, which
cross-entropy punishes and accuracy doesn't see. Nothing after epoch 9 was learning
that generalized, so the study ran at 30 epochs, which covers every arm's minimum at
under a third of the cost.

![Mean val loss by epoch over 3 seeds for four arms. The control bottoms out at 2.17 at epoch 8 and climbs to 2.99 by epoch 30. Dropout 0.0 bottoms out earliest, at epoch 5, and climbs steepest, to 4.50. The 2-decoder-block arm bottoms out higher than the control but climbs slower and ends below it at 2.74. The Czech-source arm sits above the others throughout and ends at 3.63](assets/val_curves.png)

______________________________________________________________________

## 2. The ablation

Every arm is the control with exactly one thing changed, so a gap against the control
is that change's doing. Predictions were written into the run plan before the runs
started.

![Best val loss per arm, one dot per seed, a black tick at each arm's mean, and the control's seed range shaded. The three schedule arms, RMSNorm, and QK norm off sit inside or at the edge of the shaded band; the two depth cuts, dropout 0.0, and the Czech source sit clearly to its right, the Czech source furthest at 2.64](assets/best_val_by_arm.png)

### Components

| Arm | Prediction | Result |
|---|---|---|
| `dropout_rate=0.0` | Clearly worse, early best epoch | Confirmed: +0.171, best epoch 5 in all three seeds |
| `num_decoders=2` | Close to the control | Wrong: +0.092, the worse of the two depth cuts |
| `num_encoders=2` | Worse than `num_decoders=2` | Wrong: +0.058, better than the decoder cut |
| `qk_norm=False` | Worse, as on the predictor (+0.095 there) | Not established: +0.022, its range overlaps the control's |
| `norm="rmsnorm"` | Inside noise | Confirmed: +0.013 |

- **Dropout carries the regularization.** Without it, val loss bottoms out three
  epochs sooner and then climbs steepest of any arm, to 4.50 by epoch 30 against the
  control's 2.99. On 29,000 pairs, a 7.5M-parameter model memorizes fast.
- **The decoder prediction was wrong, and the metric is the likely reason.** It cited
  Kasai et al. (2021), who found that a deep encoder with a shallow decoder loses
  little *BLEU*. This study measures next-token loss on German under teacher forcing,
  much of which is predicting German from German: the decoder's job. Losing decoder
  depth should hurt that metric more than losing encoder depth, whatever it does to
  translation quality. That is an explanation of the result, not something these runs
  test.
- **A smaller decoder also overfits more slowly.** The 2-decoder arm's best is worse,
  but its val loss climbs slower afterwards and ends epoch 30 at 2.74, below the
  control's 2.99. Best-epoch loss is the right ranking for an ablation that keeps the
  best checkpoint; a ranking by final loss would invert this pair.
- **QK norm did not reproduce its predictor result.** On the masked-token predictor at
  50 epochs it was worth 0.095; here 0.022, inside the noise. The predictor's runs were
  still improving when they stopped, and these peak by epoch 9, so the two studies
  measure it at different points in training.

### Learning-rate schedules

Each schedule peaks at the control's 1e-3.

| Arm | LR over 30 epochs | Prediction | Result |
|---|---|---|---|
| `cosine` | 1e-3 decaying to ~0 | Better, because decay lowers the LR through the overfitting epochs | Not established: −0.005 |
| `warmup_cosine` | Linear 4e-5 → 1e-3 over 5 epochs, then cosine | Inside noise against `cosine` | Confirmed: +0.003 against `cosine` |
| `one_cycle` | Cosine-shaped 4e-5 → 1e-3 over 9 epochs, then cosine | Slightly worse than `warmup_cosine` | Not established: −0.001 against it |

- **Decay comes too late to matter.** The best epoch is 8 or 9, and by epoch 9 a
  30-epoch cosine has only lowered the LR to 83% of its peak. The run overfits before
  the decay does anything; the schedule shapes the epochs after the minimum, which the
  best-epoch ranking ignores.
- **A warmup moves the best epoch, not the best loss.** `one_cycle` reaches its peak
  LR at epoch 9 and its best val loss at epochs 11 to 13; `warmup_cosine`, peaking at
  epoch 6, reaches its best at 10 to 11. Both land at the control's loss, two to four
  epochs later.

### Source language

| Arm | Prediction | Result |
|---|---|---|
| `src_lang="cs"` | Worse, by more than the noise floor | Confirmed: +0.474, val accuracy 54.9% against 62.3% |

The target stays German, so this arm uses the control's German tokenizer and val
sentences and its loss compares directly. **The Δ is not language distance alone.**
The German and the Czech captions are both translations of the English originals:
English→German pairs an original with its translation, while Czech→German pairs two
independent translations of a third sentence, so some of the German is not recoverable
from the Czech at all. Truncation is not the cause either: Czech captions are shorter
than English ones (9.1 words against 11.9), so `seq_len=32` removes less of them.

______________________________________________________________________

## What this doesn't show

- **Nothing here measures translation quality.** Teacher-forced next-token loss and
  accuracy reward a good German language model as much as a good translator. A BLEU
  score from greedy or beam decoding on the test split is the missing measure.
- **These results hold at `d_model=128` only.** Three seeds of a 7.5M-parameter model on
  29,000 pairs. Whether schedules, dropout, and depth matter differently with more
  capacity is the question the
  [scale study](../2026-10-15/README.md) asks next, at `d_model=256` and `512`.
- **Three seeds bound the noise roughly.** The control's 0.034 spread is one sample of
  three; Δs within about twice that (QK norm, RMSNorm, every schedule) are not
  established in either direction.
- **Post-norm is not an arm.** `EncoderBlock` and `DecoderBlock` hardcode the pre-norm
  order, so testing it means changing shared code, not setting a flag.

______________________________________________________________________

## Data and reproducing

### Files

- [runs.csv](runs.csv): one row per run (31 rows). It has the experiment, the arm, the
  full config, epochs, parameter count, train seed, best val loss with its epoch,
  perplexity, and accuracy, the final train and val loss, the mean time per epoch, and
  the commit the history file recorded.
- [epochs.csv](epochs.csv): one row per epoch per run (1,000 rows). It has train loss,
  val loss, val perplexity, val accuracy, and time per epoch. Join it to the runs file
  on `run_id`.
- `app/history/<run_id>.json`: the full history file for each run, with an attention
  capture per epoch under `app/history/attention_probes/<run_id>/`. These are gitignored
  and exist only on the machine that trained them.

The `experiment` column groups the runs: `ablation_30ep` (30) and
`first_control_100ep` (1).

### Code

The 30 ablation runs ran commit `a97caac` and record it. To rerun them, from the repo
root:

```bash
caffeinate -is uv run python app/src/training/experiments/multi30k_translator_ablation.py
```

The 100-epoch run ran `bb4abad`, where `ABLATION_EPOCHS` was 100, and records it. Later
commits add a study name to the command, defaulting to this ablation, and encode the
loader's sentences with `bpe.encode_many`, which returns the same ids as the per-sentence
`encode` these runs used.

### Known issues in the recorded data

- **The 100-epoch run's terminal log was overwritten** by the 30-epoch launch. Its
  history file is complete; only the log is gone.
- **Two runs were stopped partway and left no history file:** the second control seed
  of the 100-epoch launch, stopped at epoch 13 when the length changed, and the first
  run of a 12+12-block follow-up, stopped while its loaders were building when the
  scale study replaced it.
