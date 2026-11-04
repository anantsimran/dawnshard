# T5: results

No GPUs here, so none of the ticket's speed or memory rows could run: those need the
4x 24 GB box and the 8x H100 session. What runs on CPU with gloo is correctness, which
the ticket puts first ("loss parity before any speed claim").

## Parity, from the tests (2 processes, gloo)

| Check | Result |
|---|---|
| `NaiveDDP` gradients equal single-process gradients on the combined batch | equal (`test_ddp.py`) |
| `BucketedDDP` gradients equal single-process gradients, with buckets small enough to force several | equal |
| Every parameter in exactly one bucket, in reverse registration order | yes |
| `no_sync()` backward issues zero collectives; the next backward issues one per bucket | yes |
| `Zero2AdamW` after 3 steps equals `torch.optim.AdamW` on the combined batch | within 1e-6 (`test_zero.py`) |
| Each rank holds gradients only for the parameters it owns | yes |
| FSDP2 on CPU: every parameter a DTensor, one SGD step equals single-process SGD | within 1e-5 (`test_fsdp.py`) |
| Static bytes per parameter at 4 GPUs: 16, 7, 5.5, 4 (27.5, 12.0, 9.5, 6.9 GB at 1.72B) | matches the table (`test_perf.py`) |

## The entry point under `torchrun`, every strategy

Tiny Qwen3, 2 ranks, 30 steps of 2 micro-batches of 4 x 64 random tokens, on a CPU
shared with another job. Rank losses are the mean over each rank's own batches.

| Strategy | Step-0 weight checksum (both ranks) | Rank losses | Wall clock |
|---|---|---|---|
| `ddp-naive` | 373.668130 | 6.2525 / 6.2506 | 10.4 s |
| `ddp-bucketed` | 373.668130 | 6.2525 / 6.2506 | 8.9 s |
| `ddp-torch` | 373.668130 | 6.2525 / 6.2506 | 10.1 s |
| `zero2` | 373.668130 | 6.2525 / 6.2506 | 12.6 s |
| `fsdp` (bf16 params) | per-shard sums 184.736 + 188.932 = 373.668 | 6.2687 / 6.2646 | 13.0 s |

- **The four data-parallel strategies agree to four decimals**, as they must: they
  compute the same update.
- **FSDP differs in the third decimal** because it computes in bf16 by default. The
  checksum in this run summed each rank's shard; it now gathers the full tensors.
- Wall-clock differences are start-up and CPU contention, not communication; gloo on
  one host says nothing about NCCL.

## To run on GPUs

```bash
uv run torchrun --standalone --nproc-per-node=4 -m training.model.qwen3 --config 0.6b --tokens <shard.bin> --seq-len 2048 --batch-size 1 --accumulation 8 --steps 100 --precision bf16 --strategy ddp-bucketed --peak-tflops 165
```

```bash
uv run torchrun --standalone --nproc-per-node=4 -m training.model.qwen3 --config 1.7b --tokens <shard.bin> --seq-len 2048 --batch-size 1 --accumulation 8 --steps 100 --precision bf16 --strategy fsdp --peak-tflops 165
```

Prediction to check first: `ddp-torch` at 1.7B runs out of memory at about 27.5 GB of
static memory per GPU; `zero2` fits at about 9.5 GB; `fsdp` at about 6.9 GB plus one
gathered block.
