# Attention is All You Need — Project Plan

## Problem Statement

Implement the Transformer architecture from scratch (no `nn.TransformerEncoder` wrappers) and reproduce the WMT 2014 English→German translation results from the paper, validating that the attention mechanism alone — without recurrence or convolution — is sufficient for sequence-to-sequence tasks.

---

## Success Criteria

Graded levels — hit them in order:

| Level | Criterion | How to verify |
|---|---|---|
| **L1** | Model trains without exploding/vanishing gradients | Loss decreases smoothly for 10k steps on a small dataset |
| **L2** | Learns copy task (input = output sequence) | Achieves >95% token accuracy on synthetic copy task |
| **L3** | English→German BLEU ≥ 25 (paper: 27.3) | `sacrebleu` on WMT14 newstest2014 |
| **L4** | Matches paper BLEU within 1 point at same parameter count | Full WMT14 training run, `d_model=512`, `N=6` |

L3 is the realistic bar for a solo project. L4 requires significant compute (~8 P100 GPUs × 12 hours per the paper).

---

## Suggested Stack

| Component | Choice |
|---|---|
| Language | Python 3.11+ |
| Framework | PyTorch (raw — no HuggingFace transformers) |
| Data | HuggingFace datasets (WMT14 en-de) |
| Tokenization | sentencepiece (BPE, shared vocab of 37k) |
| Evaluation | sacrebleu |
| Experiment tracking | wandb or tensorboard |

---

## Architecture Components (from paper)

1. Multi-Head Self-Attention — scaled dot-product, `h=8` heads
2. Positional Encoding — sinusoidal, added to embeddings
3. Encoder stack — 6 identical layers (self-attention + FFN + LayerNorm + residual)
4. Decoder stack — 6 layers (masked self-attention + cross-attention + FFN)
5. Label smoothing — `ε=0.1`
6. Learning rate schedule — warmup steps with inverse sqrt decay
7. Beam search — width 4, length penalty `α=0.6`

---

## Project Structure

```
transformer/
  model/           # attention, encoder, decoder, embeddings
  training/        # optimizer schedule, loss, train loop
  data/            # WMT14 loading + BPE tokenization
  eval/            # beam search, BLEU scoring
  tests/           # unit tests (especially masking)
```

---

## Todo List

### Setup
- [ ] Setup Docker image, CLI and uv
- [ ] Initialize repo and virtual environment
- [ ] Install dependencies (torch, sentencepiece, sacrebleu, datasets, wandb)
- [ ] Download and preprocess WMT14 en-de dataset
- [ ] Train shared BPE tokenizer (37k vocab) with sentencepiece

### Model
- [ ] Implement scaled dot-product attention
- [ ] Implement multi-head attention (self + cross)
- [ ] Implement positional encoding (sinusoidal)
- [ ] Implement encoder layer + encoder stack (N=6)
- [ ] Implement decoder layer + decoder stack (N=6)
- [ ] Implement full Transformer (encoder + decoder + output projection)
- [ ] Tie input/output embedding weights

### Training
- [ ] Implement label-smoothed cross-entropy loss (ε=0.1)
- [ ] Implement warmup + inverse-sqrt learning rate schedule
- [ ] Implement token-batching (batch by tokens, not sentences)
- [ ] Add gradient clipping (max_norm=1.0)
- [ ] Set up experiment tracking (wandb/tensorboard)
- [ ] Write training loop

### Evaluation
- [ ] Implement beam search (width=4, length penalty α=0.6)
- [ ] Wire up sacrebleu scoring on WMT14 newstest2014

### Validation (gates before scaling)
- [ ] Pass L1: loss decreases on small dataset without instability
- [ ] Pass L2: copy task >95% token accuracy (tiny model, CPU)
- [ ] Pass L3: BLEU ≥ 25 on WMT14 en-de

### Tests
- [ ] Unit test: attention output shapes
- [ ] Unit test: padding mask correctness
- [ ] Unit test: causal (look-ahead) mask correctness
- [ ] Unit test: combined masking in decoder
