# TODO: the component and LR-schedule ablation for `Multi30kEnPredictor`

**Status: not run.** The code is in place and smoke-tested; the training has not started.
This file is the run plan. When the runs finish it becomes the write-up at
`experiments/multi30ken_predictor/<run date>/README.md` and this file goes away.

Two questions, sharing one control. First: of the components `Multi30kEnPredictor` is
built from, which ones are earning their place? The
[2026-09-29 study](2026-09-29/README.md) answered that for two of them (QK norm,
dropout) at 50 epochs. Second: does the constant LR every run so far has used cost
anything, and if a schedule helps, is it the decay or the warmup? Both are run at a
length where the model has stopped improving, so a gap measures the change rather than
how far each arm got.

______________________________________________________________________

## Before you start

1. **Decide the length.** `ABLATION_EPOCHS = 100` in
   [multi30ken_predictor_ablation.py](../../app/src/training/experiments/multi30ken_predictor_ablation.py). The
   2026-09-29 runs were all still improving at epoch 50, which is the reason for 100.
   If the control is still improving at 100 too, the ablation has the same flaw as last
   time and the numbers are about partial training again. Check the control's last ten
   epochs before trusting any arm.
1. **Run the right script.** The ablation is `multi30ken_predictor_ablation.py`; the
   model's own `multi30ken_predictor.py` trains `SINGLE_RUN` instead and you lose the
   night.
1. **Pin HEAD.** `fit` reads the commit when a run *finishes*, so anything that moves
   HEAD mid-run mislabels every run still going. Four runs in the last study recorded
   the wrong commit this way. Commit first, then don't touch the branch.
1. **Stop the machine sleeping:** `caffeinate -is uv run python ...`. Recorded epoch
   durations exclude sleep, so a slept run looks fine in the data and wrong on the
   clock.
1. **Budget about 11 hours.** 7 arms x 3 seeds x 100 epochs, at the 19 s/epoch the
   last study measured for `d_model=128`. No arm is more expensive than the control.
   To cut it, drop `ABLATION_SEEDS` to 2 — but then the control's own spread is a
   single gap again, which is the weakest part of the last study.

## The run

```bash
caffeinate -is uv run python app/src/training/experiments/multi30ken_predictor_ablation.py
```

All 21 runs share one loader and one tokenizer, because `get_loaders` is cached by
`seq_len` and every arm uses `seq_len=32`. That removes the tokenizer confound the
2026-09-29 sweep had across its different `seq_len` values.

Each run writes `app/history/<run_id>.json` and one attention capture per epoch under
`app/history/attention_probes/<run_id>/`. Both are gitignored. 21 runs x 100 epochs of
probes is a lot of files; check free space first.

## The arms

Every arm is `ABLATION_REFERENCE` (`d_model=128`, dropout 0.1, QK norm on, 6 blocks,
4 heads, FFN 6x, LayerNorm, sinusoidal positions, final norm, constant LR 1e-3) with
exactly one thing changed. Three seeds each; the spread inside the control is the noise
floor every other Δ has to clear.

### Components

| Arm | What comes out | What it settles | Prediction |
|---|---|---|---|
| control | nothing | The noise floor, and whether 100 epochs converges | Below 2.06; the 50-epoch control hit 2.063 still improving |
| `qk_norm=False` | RMSNorm on q and k | Whether QK norm's +0.095 at 50 epochs survives convergence | Still clearly worse, +0.05 to +0.12 |
| `dropout_rate=0.0` | Dropout everywhere | The one verdict that should flip with length | Worse than at 50 epochs, where it was +0.004 and overfitting hard |
| `final_norm=False` | The norm after the stack | Whether a pre-norm stack needs its closing norm | Worse, and the arm most likely to be unstable rather than merely worse |

### LR schedules

Each replaces the control's constant LR and peaks at the same 1e-3. `fit` steps the
scheduler once per epoch, so every schedule is sized in epochs: an epoch is 454 Adam
updates, and a warmup ramps in jumps of that size.

| Arm | LR over 100 epochs | Compared with | What it settles | Prediction |
|---|---|---|---|---|
| `schedule="cosine"` | 1e-3 decaying to ~0 | control | What decay buys | Better, if the control converges. In the 20-epoch sweep cosine lost all 14 of its pairs, because every run was still improving and decay cut it short; if the control is still falling at 100, it loses again |
| `schedule="warmup_cosine"` | Linear 4e-5 → 1e-3 over 5 epochs, then cosine | cosine | What warmup buys | Inside noise. Warmup protects post-norm stacks, and `EncoderBlock` is pre-norm; expect only a smoother first few epochs |
| `schedule="one_cycle"` | Cosine-shaped 4e-5 → 1e-3 over 30 epochs, then cosine | warmup_cosine | What a long warmup costs | Slightly worse if the run is budget-limited, since a third of it trains below peak; a wash if it converges. Momentum cycling is off, so only the LR shape differs |

The predictions are written down now on purpose. An ablation whose predictions are
decided after the numbers arrive is just a ranking.

**Cut from this run.** Four component arms were dropped to spend the budget where the
outcome is least certain. `num_heads=1` and `norm="rmsnorm"` were predicted inside
noise. `num_encoders=3` already has 20-epoch sweep data. `position_encoding=False` was
predicted catastrophic with near certainty: with `nn.Identity` in place of the table the
model is permutation-equivariant (verified to 6e-07), a bag of words that cannot tell
`a dog bit` from `bit a dog`. The model keeps every flag, so each is one line to restore
in `COMPONENT_ARMS`.

**Post-norm is deliberately not an arm.** `EncoderBlock` hardcodes the pre-norm order,
so testing it means changing shared code that `AGNewsClassifier` also uses, not setting
a flag on this model. Post-norm is reported elsewhere to reach better final quality when
it trains at all, so this is an open choice rather than a settled one; the forward and
gradient paths are in [norm.md](../../tutorials/transformer.md#pre-norm-vs-post-norm). If you want it in
this study, add the branch to `EncoderBlock` first and treat it as another arm.

## After the run

1. **Read the control's tail first.** If it is still falling at epoch 100, say so at the
   top of the write-up and treat every Δ as provisional.
1. **Check the three control seeds.** Their spread is the noise floor. Any Δ smaller
   than it is not a result, however good the story is.
1. **Export the CSVs.** There is no export script — the 2026-09-29 `runs.csv` and
   `epochs.csv` were produced ad hoc from the history files. Match their columns so the
   two studies can be compared, and write one if you want this repeatable.
1. **Write it up** as `experiments/multi30ken_predictor/<run date>/README.md`, following
   the 2026-09-29 page: results at a glance, setup, the arms table with Δ and the noise
   floor, then a "what it doesn't show" section. Keep the Δ column against the control
   mean.
1. **Register the new page** in `EXTRA_PAGES` in
   [hooks.py](../../pages/hooks.py), add its images to `EXTRA_ASSETS`, and add a `nav:`
   entry under Experiments in [mkdocs.yml](../../mkdocs.yml). The page builds but is
   invisible without the nav line.
1. **Delete this file** once the write-up exists.
