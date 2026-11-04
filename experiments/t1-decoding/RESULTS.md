# T1: results

Measured on CPU (4 vCPU, no GPU), so speed numbers are CPU numbers. The checkpoint is
the scale study's best arm (`d_model=256`, warmup-cosine, 8 epochs) retrained here with
`multi30k_translator_ablation.py best`: final val loss 2.009. That is lower than the
study's 2.064-2.099 because the default tokenizers were unavailable and BPE was
retrained, so the val loss is per a different set of tokens and does not compare
directly. The bundle pins those merges. Benchmark: Multi30k test (1,000 sentences),
max output length 32, batches of 64, BLEU from sacrebleu on detokenized text.
Rows are in [results.csv](results.csv), outputs in [predictions/](predictions/).

## v0-v2: what the cache buys (greedy)

| Version | Cache | BLEU | Tokens/s | ms per sentence | ms per step (first / middle / last) | Peak cache |
|---|---|---|---|---|---|---|
| v0 | none | 32.07 | 78 | 195.3 | 232 / 187 / 583 | 0 |
| v1 | self-attention | 32.07 | 298 | 51.4 | 93 / 93 / 97 | 24.0 MB |
| v2 | self + cross-attention | 32.07 | 503 | 30.4 | 107 / 51 / 52 | 48.0 MB |

- **Cached greedy is token-identical to uncached greedy**: the three prediction files
  are byte-identical, not just equal in BLEU.
- **Without a cache, step time grows with output position** (187 ms mid-sequence to
  583 ms at position 32): each step reruns the whole prefix. With the self-attention
  cache it is flat.
- **Caching cross-attention is the larger gain here** (3.8x for v1, then a further 1.7x
  for v2): the source is as long as the output (32 tokens), and recomputing its keys and
  values costs a full projection of the encoder output per layer per step.
- **Measured cache bytes equal the formula**: 2 x 6 layers x 64 rows x 4 heads x 64 dims
  x 32 positions x 4 bytes = 24.0 MiB for self-attention, and the same again for
  cross-attention at a 32-token source.

## v3: beam width (test, KV cache, no length penalty)

| Beam width | BLEU | Gain over previous width | Tokens/s | ms per sentence | Peak cache |
|---|---|---|---|---|---|
| 1 | 32.07 | | 431 | 35.5 | 48 MB |
| 2 | 33.96 | +1.89 | 120 | 121.9 | 96 MB |
| 4 | 34.13 | +0.17 | 67 | 213.4 | 192 MB |
| 5 | 34.15 | +0.02 | 54 | 265.2 | 240 MB |
| 8 | 34.29 | +0.14 | 36 | 394.9 | 384 MB |

- **Beam search with k = 1 equals greedy**: the prediction files are byte-identical.
- **The BLEU gain per extra beam shrinks**, as predicted: almost all of the +2.2 comes
  from the second beam.
- **Tokens/s falls about linearly in k from k = 2 on** (120 to 36 for 4x the beams).
  From 1 to 2 it falls 3.6x, more than linear, because the Python bookkeeping in
  `BeamSearch.step` runs per sentence per step and is a large share of a CPU step.
- **Cache memory is linear in k**: every beam holds its own rows.

## v4: length penalty (tuned on val, one test run)

| Split | Beam 4, alpha | BLEU | Mean output length (words) |
|---|---|---|---|
| val | 0.0 | 34.11 | |
| val | 0.6 | 34.08 | |
| val | 1.0 | 34.41 | |
| val | 1.5 | **34.53** | |
| test | 0.0 | 34.13 | 10.70 |
| test | 1.5 (chosen on val) | 33.58 | 11.16 |

The test references average 10.90 words; greedy averages 11.27.

- **Without a penalty, beams prefer short outputs**, as predicted: 10.70 words against
  the references' 10.90. alpha = 1.5 lengthens them to 11.16.
- **The val gain did not transfer.** alpha = 1.5 was best on val by +0.42 BLEU and is
  0.55 worse on test. Both differences are within the noise of a 1,000-sentence set
  with captions this short; the length effect is the robust result, the BLEU ranking
  of penalties is not.

## v5: batched beam

The beam search is batched from the start: (sentences x beams) are flattened into one
batch of 64 x k rows and the cache is reordered by index every step, so every v3 row
above is the batched version. There is no unbatched baseline to compare against.

## Misses against the predictions

| Prediction | Result |
|---|---|
| v0 ms/token grows with position | Yes: 187 ms mid-sequence, 583 ms at the last step |
| v1 tokens identical, ms/token flat, bytes match the formula | All three |
| v2 extra gain over v1 | Larger than v1's own (1.7x on top of 3.8x): source and target are the same length here, so recomputing cross-attention K/V costs as much as the prefix did |
| v3 BLEU gain shrinks with k; tokens/s about linear in k | Yes, except 1 to 2, where per-sentence Python bookkeeping dominates on CPU |
| v4 beams prefer short outputs without a penalty | Yes in length; the penalty's BLEU gain on val did not hold on test |

## Teacher-forced loss vs BLEU

The checkpoint's val loss is 2.009 nats per target token (a 7.5 perplexity), with about
63% next-token accuracy under teacher forcing, and it decodes at BLEU 32-34 on its own
outputs. The two disagree in kind: loss scores each token given the true prefix, BLEU
scores whole sequences built from the model's own prefixes (exposure bias), so loss
alone cannot rank decoders, and it never sees the length effect above.
