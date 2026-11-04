# T4: results

This container has no GPU, so no ledger row could be measured: the ticket's numbers
need an RTX 4090 or A100 and do not count on CPU or MPS. What is here is the v0
memory prediction for the benchmark, written before any run, and CPU checks that
every code path the ledger needs runs and agrees with the formulas.

## Prediction for the benchmark (Qwen3-0.6B, seq 2048, one sequence per micro-batch)

From `predict_memory(shape=model_shape(config=QWEN3_0_6B), tokens=2048, ...)`, in GB:

| Version | Weights | Grads | Adam | Activations | Logits | Total |
|---|---|---|---|---|---|---|
| v0 fp32 | 2.38 | 2.38 | 4.77 | 5.87 | 2.49 | 17.90 |
| v2 bf16 autocast | 2.38 | 2.38 | 4.77 | 2.94 | 2.49 | 14.96 |
| v6 full checkpointing (fp32) | 2.38 | 2.38 | 4.77 | 0.23 | 2.49 | 12.26 |
| v7 chunked CE (bf16) | 2.38 | 2.38 | 4.77 | 2.94 | 0.00 | 12.47 |

- **Micro-batch for v0 on 24 GB: 1.** Two sequences predict 26.3 GB. 64K tokens per
  step is then 32 micro-batches of accumulation.
- **The static 9.54 GB is 16 bytes per parameter** (0.596B x 16), the same under
  bf16 autocast, which saves activation memory and time, not static memory.
- **Logits are the largest single activation term** at 2.49 GB per sequence (fp32
  logits and their gradient over a 151,936-word vocabulary), which is what v7
  removes.

FLOPs per token at seq 2048: 4.99e9 (6N = 3.58e9, attention 1.41e9). At 9,000
tokens/s on an RTX 4090 (165 TFLOPS dense bf16), MFU = 27%.

## CPU checks (code paths, not performance)

Tiny config (107K parameters), 30 steps of 4 x 64 tokens, random tokens, on a CPU
shared with another training job:

| Flags | Train loss | Static memory, predicted / measured (MiB) |
|---|---|---|
| `--precision fp32` | 6.2521 | 1.63 / 1.63 |
| `--precision bf16 --accumulation 2` | 6.2533 | 1.63 / 1.63 |
| `--checkpointing full --chunked-ce` | 6.2521 | 1.63 / 1.63 |
| `--checkpointing selective --sdpa math --fused-adamw` | 6.2521 | 1.63 / 1.63 |

Checkpointing, the chunked loss, the math SDPA backend, and fused AdamW leave the
fp32 loss unchanged to four decimals, as they should: none changes the math. bf16
moves it in the fourth decimal.

Tests that back the formulas: `test_perf.py` (6N equals counted matmul FLOPs
exactly, the 0.6B FLOPs and MFU examples, 16 bytes per parameter, logits term,
intensity of the ticket's matmul at 647 FLOP/byte), `test_losses.py` (chunked equals
full cross-entropy in loss and gradients), `test_qwen3.py` (checkpointing modes give
identical gradients), `test_train_loop.py` (accumulation over four batches equals one
batch of four).

## To run on the GPU

Each version is one command; the perf record lands in `app/history/<run>.perf.json`
with the prediction next to the measurement:

```bash
uv run python -m training.model.qwen3 --config 0.6b --tokens <shard.bin> --seq-len 2048 --batch-size 1 --accumulation 32 --steps 300 --precision fp32 --peak-tflops 165
```

```bash
uv run python -m training.model.qwen3 --config 0.6b --tokens <shard.bin> --seq-len 2048 --batch-size 1 --accumulation 32 --steps 300 --precision bf16 --chunked-ce --fused-adamw --compile --peak-tflops 165 --memory-snapshot v7.pickle
```

```bash
uv run python app/src/training/viz/roofline.py --peak-tflops 165 --bandwidth-gbs 1008
```
