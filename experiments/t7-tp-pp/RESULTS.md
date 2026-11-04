# T7: results

CPU only (4 vCPU, gloo, fp32). The speed rows need NVLink GPUs. Command:
`uv run python app/src/training/experiments/tp_pp.py all`.

## Loss parity, TP=2 against one process ([parity.csv](parity.csv))

Tiny Qwen3, full init then shard, the same 50 batches, AdamW lr 1e-3, fp32:

| Plan | Step-0 abs. loss difference | Max over 50 steps | Final loss TP / single |
|---|---|---|---|
| hand (`parallelize_tensor_parallel`) | 4.8e-07 | 2.6e-04 | 6.236471 / 6.236206 |
| DTensor (`parallelize_dtensor`) | 0 | 2.7e-04 | 6.236472 / 6.236206 |

- **Step 0 is within the 1e-5 prediction.** The difference is the order of fp32
  additions in the all-reduces.
- **The drift to 2.6e-4 over 50 steps is Adam, not the layers.** Adam divides each
  gradient by its running RMS, so a 1e-7 difference on a near-zero gradient becomes a
  visible step. The single-step gradient tests (`test_tensor.py`) agree to 1e-5.

## Pipeline bubble, p = 4, backward = 2 x forward ([bubble.csv](bubble.csv))

| m | Formula (p-1)/(m+p-1) | GPipe simulated | 1F1B simulated | Peak activations held, GPipe / 1F1B |
|---|---|---|---|---|
| 4 | 42.9% | 42.9% | 42.9% | 4 / 4 |
| 8 | 27.3% | 27.3% | 27.3% | 8 / 4 |
| 16 | 15.8% | 15.8% | 15.8% | 16 / 4 |
| 32 | 8.6% | 8.6% | 8.6% | 32 / 4 |

Prediction: 1F1B lowers peak activation memory, not the bubble. It holds at most p
micro-batches against GPipe's m. With equal stage times the simulation reproduces
the formula exactly; measured GPU timelines will be above it (launch and send costs).

## Stage balancing, Qwen3-1.7B over 4 stages ([balance.csv](balance.csv))

| Method | Layers per stage | Cost per stage (block-equivalents) | Slowest / mean |
|---|---|---|---|
| equal layer count | 7 / 7 / 7 / 8 | 7.0 / 7.0 / 7.0 / 13.2 | 1.54 |
| `partition` by FLOPs | 8 / 9 / 9 / 3 | 8.0 / 9.0 / 9.0 / 8.2 | 1.05 |

The last stage holds the tied head (6.2 blocks) and 2 blocks, as predicted.

## TP communication, predicted ([comm.csv](comm.csv))

Qwen3-1.7B, micro-batch 2 x 2048, bf16, 4 all-reduces per block (2 forward, 2
backward):

| TP | Bytes per GPU per micro-step | NVLink ~200 GB/s | PCIe no P2P ~10 GB/s |
|---|---|---|---|
| 2 | 1.88 GB | 9.4 ms | 188 ms |
| 4 | 2.82 GB | 14.1 ms | 282 ms |

Against roughly 30 ms of compute per micro-step on an H100, TP=2 over PCIe spends
six times its compute on communication: the v8 prediction that TP=2 on two RTX 4090s
is slower than one GPU.

## Tests

`test_tensor.py` (2 processes): column-then-row MLP equals the dense MLP in output,
input gradient, and weight-shard gradients; head-sharded GQA attention equals dense
attention; vocabulary-parallel cross-entropy equals `F.cross_entropy` (1e-10,
float64); vocabulary-parallel embedding and the sequence-parallel round trip; the hand
and DTensor Qwen3 plans match the dense loss. `test_pipeline.py`: 1F1B valid with at
most p - s in flight for p \<= 8, m \<= 32; simulated bubble equals the formula; the
partitioner within 10% of the mean; 2-stage GPipe and 1F1B gradients equal one
process. `test_mesh.py`: TP is the innermost, consecutive-rank dimension.

## Not run here

v4 throughput within 10%, v5 measured GPU bubbles, v6 `torch.distributed.pipelining`
interleaved and zero-bubble, v7 2D and 3D throughput, v8 PCIe vs NVLink: all need
GPUs.
