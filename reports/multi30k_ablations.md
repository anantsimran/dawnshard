# The training setup changes the ablation result

A schedule that barely moves a small translator improves a wider one. Dropout
remains useful when a masked-token model reaches its plateau. The setup changes
which effects are clear and which remain unresolved.

Three Multi30k studies put that principle to the test: **66 comparison runs,
22 arms, and three independent training seeds per arm**, completed between
October 13 and 15, 2026. Each arm is evaluated against its own study's control.
The translator's separate 100-epoch pilot informs the training budget but is
excluded from those counts and the comparisons below. Each result belongs to
its task, tokenizer, model size, and training budget.

## Start with a control that has enough time

Both tasks use 29,000 training captions or caption pairs, sequences of 32 tokens,
and byte-level BPE with 10,000 merges. Their losses answer different questions.
The translator predicts the next German token with the true preceding tokens
supplied; the English encoder predicts selected masked tokens. A lower number
on one task does not rank it against the other.

| Study | Question | Comparison runs | Training budget |
|---|---|---|---|
| [Translator components](../experiments/multi30k_translator/2026-10-13/README.md) | Which components matter at width 128? | 30 · 10 arms | 30 epochs |
| [Translator scale](../experiments/multi30k_translator/2026-10-15/README.md) | Do those results survive more capacity? | 27 · 9 arms | 25 epochs |
| [Masked-token convergence](../experiments/multi30ken_predictor/2026-10-15/README.md) | Do dropout and decay help after training plateaus? | 9 · 3 arms | 100 epochs |

The translator's pilot reached its best validation loss at epoch 9 and then
deteriorated through epoch 100. The 30-epoch comparison covered every arm's
minimum, and the later scale study used 25 epochs. The masked-token control took
much longer: its three best epochs were 87, 87, and 93. The budgets follow the
observed training trajectories rather than forcing both tasks to stop together.

Throughout this report, **best validation loss** means the mean of each run's
own minimum, even when different seeds reach it at different epochs. The plotted
ranges show three observed seeds, not confidence intervals or a significance
test. A small gap with overlapping ranges remains unresolved.

## At width 128, dropout and depth have the clearest effects

The translator control has six encoder blocks, six decoder blocks, width 128,
dropout 0.1, QK normalization, and LayerNorm. It averages **2.167** at its best
epoch. Removing dropout raises that to **2.338**, a cost of **0.171**. Every
no-dropout seed is worse than every control seed.

![Three seed dots per translator arm show that removing dropout or reducing encoder or decoder depth raises best validation loss beyond the control's range. RMSNorm and removing QK normalization overlap the control.](assets/multi30k_ablations/translator_components.png)

*Each dot is one run's minimum; the dark tick is the arm mean. The shaded band
is the width-128 control's observed seed range, 2.148–2.182.*

Reducing the decoder from six blocks to two costs **0.092**; reducing the
encoder to two costs **0.058**. Both ranges lie above the control's, and the
decoder cut is worse in all observed seeds. That result applies to teacher-forced
next-token loss. It does not establish which depth allocation produces better
decoded translations.

Removing QK normalization costs 0.022 on average, and switching to RMSNorm costs
0.013. Their ranges overlap the control. None of the three learning-rate
schedules improves the mean by more than 0.005. These comparisons leave their
effect unresolved at this model size.

The separate Czech-to-German arm costs 0.474 while retaining the German target
tokenizer. Its source captions are translations of the English originals, so
changing source language also changes the relationship between source and target
captions. The [component study](../experiments/multi30k_translator/2026-10-13/README.md#source-language)
explains why this is not a clean measure of language distance.

## More width changes the schedule result

Increasing the translator from width 128 to 256 lowers the constant-rate
control's mean minimum from **2.167 to 2.111**. Increasing it again to 512
raises the loss to **2.221**. More parameters alone did not produce a better
result at the tested constant learning rate of `1e-3`.

![At widths 128, 256, and 512, constant-rate translator losses average 2.167, 2.111, and 2.221. Warmup plus cosine gives 2.165, 2.077, and 2.093, respectively. Ranges overlap at 128 but separate at 256 and 512.](assets/multi30k_ablations/width_and_schedule.png)

*The translator studies share pinned English and German tokenizers. Width 128
uses a 30-epoch budget; widths 256 and 512 use 25. Schedule durations change with
the budget, so each schedule is compared with the control at its own width.*

At width 256, warmup plus cosine improves the mean by **0.034**, and one-cycle
by **0.020**. Both seed ranges sit below the constant-rate control's. At width
512, those schedules recover **0.128** and **0.119**, respectively. The schedule
comparison that was unresolved at width 128 now produces separated ranges.

The lowest mean belongs to **width 256 with warmup plus cosine: 2.077**, at
22.1 million parameters and about 118 seconds per epoch on the study machine.
Width 512 with that schedule averages 2.093, at 72.5 million parameters and
about 331 seconds per epoch. Their seed ranges overlap; the larger model offers
no demonstrated improvement for roughly **3.3× the parameters and 2.8× the
epoch time** in these runs.

This supports changing the optimization recipe when scaling. It does not isolate
the warmup's contribution from the decay's: there is no cosine-only arm at
width 256 or 512. All schedule runs peak while the learning rate remains close to
its maximum, which suggests an early-training benefit, but a warmup-only
comparison is needed to attribute it.

Doubling depth to 12 encoder and 12 decoder blocks also hurts at a constant rate,
by 0.149. That arm was not run with a warmup schedule. Its result cannot yet
separate the effect of depth from the optimization setting.

## Dropout still matters at masked-token convergence

The 100-epoch English masked-token study compares a width-128 control with
no dropout and with cosine decay. Removing dropout raises the mean minimum
from **2.526 to 2.643**, a cost of **0.117**, with no overlap between seed
ranges. Its best epochs are 51–53; the control keeps improving until 87–93.

![The masked-token control averages 2.526, cosine decay 2.470, and no dropout 2.643. The no-dropout seeds are all worse than the control; cosine's range narrowly overlaps it.](assets/multi30k_ablations/masked_token_convergence.png)

*These are minima within the 100-epoch study. They share one tokenizer and one
set of masked validation positions; they are not comparable with translation
losses or the earlier 2,000-merge masked-token study.*

Cosine decay leads by **0.056**, with a much smoother late-training curve. Its
worst seed is 2.4874 and the control's best is 2.4859, so the observed ranges
still overlap. The mean of the final ten epochs separates them more clearly,
but that measure was chosen after looking at the results. It supports a
follow-up with more seeds and a metric fixed in advance.

The [September study](../experiments/multi30ken_predictor/2026-09-29/README.md)
used a smaller vocabulary and stopped while its runs were still improving.
Its absolute losses cannot be compared with this study's. Longer training and
a different tokenizer both changed the setup, so a reversal between the studies
cannot be assigned to training length alone.

## What to carry into the next experiment

For this translator, the economical candidate is width 256 with a warmup
schedule. Keep dropout and the six-block decoder while testing the next change.
For the masked-token predictor, retain dropout and give decay more seeds before
treating its lead as settled. These are starting points supported by these runs,
not general rules for transformer training.

Three follow-ups address the remaining uncertainty directly:

1. **Separate warmup from decay** at width 256, with cosine-only and warmup-only
   arms compared against the same control.
1. **Give the deeper translator a warmup**, to test depth under the optimization
   setting that recovered the wider model.
1. **Evaluate decoded translations**, using a held-out translation metric such
   as BLEU. Teacher-forced loss alone does not measure translation quality.

## Data and reproduction

The [summary CSV](assets/multi30k_ablations/summary.csv) contains all 22 arms,
their means, observed ranges, and differences from their own controls. It is
generated from the checked-in records below. Figures are also available as SVG:
[components](assets/multi30k_ablations/translator_components.svg),
[width and schedule](assets/multi30k_ablations/width_and_schedule.svg), and
[masked-token convergence](assets/multi30k_ablations/masked_token_convergence.svg).

| Study record | Run-level data | Per-epoch data |
|---|---|---|
| [Translator components · October 13](../experiments/multi30k_translator/2026-10-13/README.md) | [31 runs, including the pilot](../experiments/multi30k_translator/2026-10-13/runs.csv) | [1,000 epochs](../experiments/multi30k_translator/2026-10-13/epochs.csv) |
| [Translator scale · October 15](../experiments/multi30k_translator/2026-10-15/README.md) | [27 runs](../experiments/multi30k_translator/2026-10-15/runs.csv) | [675 epochs](../experiments/multi30k_translator/2026-10-15/epochs.csv) |
| [Masked-token convergence · October 15](../experiments/multi30ken_predictor/2026-10-15/README.md) | [9 runs](../experiments/multi30ken_predictor/2026-10-15/runs.csv) | [900 epochs](../experiments/multi30ken_predictor/2026-10-15/epochs.csv) |

Rebuild the summary and figures without retraining or downloading data:

```bash
uv run python reports/generate_multi30k_ablations.py
```

The [generator](generate_multi30k_ablations.py) checks each comparison run against
its complete epoch record and confirms three distinct seeds per arm. The study
records retain the original configurations, recorded commits, rerun commands,
predictions, and limitations.
