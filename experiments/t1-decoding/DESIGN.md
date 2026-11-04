# T1: KV cache and beam search, design notes

This ticket makes the Multi30k translator translate. The code is a KV cache
([kv_cache.py](../../app/src/training/transformer/kv_cache.py)), one decode loop with
two strategies ([decoding.py](../../app/src/training/transformer/decoding.py)), a
cached path through `DecoderBlock` and `MultiHeadAttentionLayer`
([modules.py](../../app/src/training/transformer/modules.py)), and an inference
bundle plus a `decode` command in
[multi30k_translator.py](../../app/src/training/model/multi30k_translator.py). The
measured numbers are in [RESULTS.md](RESULTS.md).

______________________________________________________________________

## Cache layout: preallocated tensors, written in place

`KVCache` allocates `(layers, rows, h, max_len, d_k)` for keys and values up front.
Each step writes one slice and reads a view of the filled prefix.

- **Pros.** No allocation per step, so per-step time stays flat and the peak size is
  known before decoding starts. The size is a closed-form formula
  (`kv_cache_bytes`), which the tests check against the tensors. One tensor per
  kind (not one per layer) makes `reorder` two `index_select` calls.
- **Cons.** Memory is reserved for `max_len` even when every row stops early, and
  the attention reads a strided view rather than a contiguous tensor.

**Alternatives considered.**

- *Concatenate per step* (`torch.cat` onto a growing tensor, Hugging Face's
  `DynamicCache`). Simplest code and no wasted memory, but every step reallocates
  and copies the whole cache, which is quadratic in bytes moved.
- *Paged blocks* (vLLM). Shares memory across requests and makes beam forks
  copy-on-write, but needs a block table and a custom attention kernel. It belongs in
  the serving ticket (T8), not here.

## The cached path is a second method, not a flag on `forward`

`DecoderBlock.forward_step` and `Multi30kTranslator.decode_step` exist beside the
teacher-forced `forward`, which is unchanged.

- **Pros.** Training cannot be affected by a decoding change; the equality tests
  (cached vs uncached greedy and beam, `decode_step` vs `forward` logits) pin the
  two paths together. Attention weights are not collected while decoding.
- **Cons.** The block's three sublayers are written twice, so a change to the block
  has to be made in both places. The parity tests catch a mismatch but not the
  omission.

**Alternative considered.** One `forward` with an optional cache argument, as most
libraries do. Less code, but every training call would carry decoding branches, and
the attention-map side effect that probes rely on would need its own flag.

`MultiHeadAttentionLayer` gained `project_kv` and a `kv` argument instead of a cache
argument: the layer stays stateless, and the cache is the block's concern.

## One loop, strategies plug in

`generate` owns the loop; `Greedy` and `BeamSearch` answer "which token, which
parent row" each step. The cache is passed to the loop, and the model's step function
decides from it whether to feed the whole prefix or the newest token.

- **Pros.** "Cache on/off" and "greedy/beam" are independent axes, so the four
  combinations need no extra code, and the tests compare them directly. Beam search
  is batched from the start: rows are `sentence * k + beam`, so the ticket's v5
  (batched beam) is the only implementation rather than a later rewrite.
- **Cons.** The per-sentence bookkeeping in `BeamSearch.step` is a Python loop over
  `2k` candidates per sentence per step. At `k <= 8` and 64 sentences it is small
  next to the model, but it is the first thing to vectorize for larger beams.

**Alternatives considered.**

- *Separate greedy and beam functions.* Easier to read alone, but cache handling
  and the stopping rule would be written twice.
- *Fully vectorized beam search* (masks instead of a Python loop over sentences).
  Faster at large `k` and batch, much harder to read and test.
- *Beam as greedy with `k = 1`.* Fewer classes, but the ticket's check that beam
  `k = 1` equals greedy would then test nothing.

## Beam search details

- **Top `2k` candidates.** At most `k` of them can be EOS, so `k` live hypotheses
  always exist. Only an EOS ranked in the top `k` finishes a hypothesis, as in
  Hugging Face; a low-ranked EOS would only crowd the finished set.
- **Stop at `k` finished per sentence.** Simple and matches `early_stopping=True` in
  Hugging Face. It can stop before a longer, better-normalized hypothesis would have
  finished. The alternative, stopping when no live beam can beat the best finished
  score, is exact only without a length penalty.
- **GNMT length penalty** `((5 + length) / 6) ** alpha`, applied to finished
  hypotheses only. Live beams compete on raw summed log-prob, so the penalty affects
  which hypothesis wins, not which are explored.

## The bundle pins the tokenizers

`save_bundle` writes the constructor arguments, the weights, and both tokenizers'
merges as id pairs (`bpe.merge_pairs`) into one file.

- **Pros.** A retrained default tokenizer can no longer change what a checkpoint's
  ids mean (the failure behind #52). Loading needs one path.
- **Cons.** The merges are stored once per bundle, about 160 KB for two 10,000-merge
  tokenizers. The existing `save_state` checkpoint is kept for resuming training, so
  there are now two file formats.

**Alternative considered.** Store the merges file paths in the checkpoint. Smaller,
but the bundle would break when the files move or are overwritten.

## BLEU and timing

BLEU is `sacrebleu.corpus_bleu` on detokenized hypotheses (`bpe.decode`) against the
raw reference sentences read by `read_multi30k_split`, never tokenized or truncated
ids. Step time is wall time around each `decode_step`, synchronized on CUDA, so
per-position cost is measured directly. Every results row carries the device, the
git commit, and the bundle's content hash.

## Routes not to take

- **Recomputing the whole prefix every step** (no cache). Correct, and the reference the
  cached path is tested against, but quadratic: step t reprocesses t tokens.
- **Caching only self-attention.** Cross-attention keys and values depend only on the
  source; recomputing them every step pays the whole encoder-output projection per
  token for nothing.
- **Growing the cache with `torch.cat`.** Every step reallocates and copies everything
  cached so far.
- **Forgetting the position offset in the incremental path.** The model still produces
  fluent text, just wrong: the cached-vs-uncached equality test is the only thing that
  catches it.
- **Beam search without reordering the cache.** A forked beam keeps attending over its
  old parent's keys; again fluent and wrong.
- **Comparing beam scores without a length penalty.** Summed log-probs favor short
  outputs, so beams stop early.
- **Scoring BLEU on token ids or on truncated, tokenized text.** BLEU is defined on
  detokenized text against the raw references; anything else is not comparable with
  published numbers (Post, 2018).
- **Loading a checkpoint with whatever tokenizer is the current default.** A retrained
  default silently changes what the ids mean; pin the merges in the bundle.
