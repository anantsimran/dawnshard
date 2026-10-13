# TODO: the component, LR-schedule, and language ablation for `Multi30kTranslator`

**Status: not run.** The code is in place and smoke-tested on synthetic data; the
training has not started. This file is the run plan. When the runs finish it becomes the
write-up at `experiments/multi30k_translator/<run date>/README.md` and this file goes
away.

Four questions, sharing one control. Which components of the EN→DE translator earn
their place? Is the model's depth worth more in the encoder or in the decoder? Does the
constant LR cost anything, and if a schedule helps, is it the decay or the warmup? And
how much harder does German get to predict when the source is Czech instead of English?
The translator has no sweep of its own, so the control borrows the best setting of the
[Multi30k predictor's 2026-09-29 study](../multi30ken_predictor/2026-09-29/README.md).
That makes the control a reasonable starting point, not a tuned optimum.

______________________________________________________________________

## Before you start

1. **Set the default tokenizers.** The ablation loads the default EN, CS, and DE merges
   files rather than retraining BPE, so every arm shares one tokenizer per language.
   Train and pin them once, one call per source:
   ```bash
   uv run python -c "from training.dataload.multi30k import get_multi30k_translation_dataloader as g; [g(seq_len=32, batch_size=64, src_lang=s, tgt_lang='de', retrain_tokenizer=True, set_default_tokenizer=True) for s in ('en', 'cs')]"
   ```
   Both flags reach both sides, so the second call retrains and re-pins DE too. That is
   harmless before the run: every arm loads whichever DE default is pinned last. Don't
   rerun it once the run has started.
1. **Run the right script.** The ablation is
   [multi30k_translator_ablation.py](../../app/src/training/experiments/multi30k_translator_ablation.py).
   The model's own `multi30k_translator.py` trains one run from its module constants
   (d_model=512, cosine) instead.
1. **Pin HEAD.** `fit` reads the commit when a run *finishes*, so anything that moves
   HEAD mid-run mislabels every run still going. Commit first, then don't touch the
   branch.
1. **Stop the machine sleeping:** `caffeinate -is uv run python ...`.
1. **Budget about 12.5 hours.** A full-depth epoch measured 51 s on MPS at batch 64,
   validation included. The eight full-depth arms cost that; the two depth arms have 8
   blocks instead of 12 and should cost about two thirds of it. 10 arms x 3 seeds x 30
   epochs comes to roughly 12 hours, plus about 40 minutes building the two pairs of
   loaders, because `bpe.encode` is slow. To cut it, drop the schedule arms first (about
   4 hours saved) and keep three seeds, because the control's own spread is the noise
   floor every Δ has to clear.
1. **Why 30 epochs.** A first control run at 100 epochs (run `2ec412ae`, commit
   `bb4abad`) reached its best val loss, 2.178, at epoch 9, and climbed steadily to 4.08
   by epoch 100 while val accuracy held at 0.62–0.63. The rest of that run was
   overconfidence, not learning. Its history file is still in `app/history/`. Leave it
   out of the write-up's tables, since it ran 100 epochs.

## The run

```bash
caffeinate -is uv run python app/src/training/experiments/multi30k_translator_ablation.py
```

All 30 runs share one tokenizer per language, and every EN→DE run shares one pair of
loaders, because `get_loaders` is cached by `seq_len` and the language pair. The CS→DE
arm builds its own pair once.

Each run writes `app/history/<run_id>.json` and one attention capture per epoch under
`app/history/attention_probes/<run_id>/`, with encoder, decoder self-, and
cross-attention for every block. Both are gitignored. 30 runs x 30 epochs of probes is
a lot of files; check free space first.

## The arms

Every arm is `ABLATION_REFERENCE` (`d_model=128`, dropout 0.1, QK norm on, 6 encoder and
6 decoder blocks, 4 heads, FFN 6x, LayerNorm, constant LR 1e-3) with exactly one thing
changed. Three seeds each.

Val loss is the teacher-forced next-token loss over the target's non-pad positions, and
accuracy is top-1 over the same positions. Neither is a translation-quality metric.

### Components

| Arm | What changes | What it settles | Prediction |
|---|---|---|---|
| control | nothing | The noise floor, and whether 30 epochs reaches its minimum | Best val epoch well before 30, near the first run's 9. 29,000 pairs is small for 12 blocks, so expect overfitting rather than a run still improving |
| `qk_norm=False` | RMSNorm on q and k comes out | Whether QK norm's gain on the predictor carries over to a model with cross-attention | Worse, as on the predictor (+0.095 at 50 epochs there) |
| `dropout_rate=0.0` | Dropout everywhere | How much regularization a 12-block model needs on 29,000 pairs | Clearly worse, and its best epoch comes early |
| `norm="rmsnorm"` | Every LayerNorm becomes RMSNorm | Whether mean-centering matters | Inside noise |
| `num_decoders=2` | Decoder cut from 6 to 2 blocks | What decoder depth buys | Close to the control. Kasai et al. (2021) report that a deep encoder with a shallow decoder loses little quality |
| `num_encoders=2` | Encoder cut from 6 to 2 blocks | What encoder depth buys | Worse than `num_decoders=2`, at the same 8 blocks in total |

### LR schedules

Each replaces the control's constant LR and peaks at the same 1e-3. `fit` steps the
scheduler once per epoch, so every schedule is sized in epochs.

| Arm | LR over 30 epochs | Compared with | What it settles | Prediction |
|---|---|---|---|---|
| `schedule="cosine"` | 1e-3 decaying to ~0 | control | What decay buys | Better, if the control overfits: decay lowers the LR through the epochs where the constant-LR run is already past its best |
| `schedule="warmup_cosine"` | Linear 4e-5 → 1e-3 over 5 epochs, then cosine | cosine | What warmup buys | Inside noise. Warmup mainly protects post-norm stacks, and both block types here are pre-norm |
| `schedule="one_cycle"` | Cosine-shaped 4e-5 → 1e-3 over 9 epochs, then cosine | warmup_cosine | What a long warmup costs | Slightly worse, since a third of the run trains below peak LR |

### Source language

The target stays German, so the arm loads the control's DE tokenizer and is scored on
the same val sentences, and its val loss compares with the control's directly. Multi30k
task 1 is line-aligned across EN, DE, FR, and CS, which is what makes that true. An arm
with a different target would not compare: val loss is a mean per target BPE token, and
another target language has its own tokenizer and token count.

| Arm | What changes | What it settles | Prediction |
|---|---|---|---|
| `src_lang="cs"` | Czech captions replace the English source | What a distant, morphologically rich source costs | Worse, by more than the noise floor |

Czech captions are shorter than English ones in the train split (9.1 words against
11.9, and 147 against 808 above 20 words), so truncation at `seq_len=32` removes less of
the source, not more. French was the other candidate and was left out for the opposite
reason: 1,356 of its captions run past 20 words.

**The Δ is not language distance alone.** The German and Czech captions are both
translations of the English ones. EN→DE pairs an original with its translation; CS→DE
pairs two independent translations of a third sentence, so some of the German is not
recoverable from the Czech at all. The write-up has to say so next to the number.

The predictions are written down before the run on purpose. Edit them now if you
disagree, not after the numbers arrive.

**Post-norm is deliberately not an arm.** `EncoderBlock` and `DecoderBlock` hardcode
the pre-norm order, so testing it means changing shared code, not setting a flag on this
model.

## After the run

1. **Read the control's tail first.** Note where its best val epoch falls. If it is
   still falling at epoch 30, say so at the top of the write-up and treat every Δ as
   provisional.
1. **Check the three control seeds.** Their spread is the noise floor. Any Δ smaller
   than it is not a result.
1. **Compare the depth arms with each other** as well as with the control. That pair is
   the only comparison here at equal block count.
1. **Write it up** as `experiments/multi30k_translator/<run date>/README.md`, following
   the predictor's 2026-09-29 page: results at a glance, setup, the arms table with Δ
   against the control mean and the noise floor, then a "what it doesn't show" section.
   No BLEU is the first entry in that section.
1. **Register the new page** in `EXTRA_PAGES` in [hooks.py](../../pages/hooks.py), add
   its images to `EXTRA_ASSETS`, and add a `nav:` entry under Experiments in
   [mkdocs.yml](../../mkdocs.yml).
1. **Delete this file** once the write-up exists.
