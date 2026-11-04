# T7: tensor and pipeline parallelism, design notes

This ticket adds tensor parallelism by hand
([tensor.py](../../app/src/training/parallel/tensor.py)), pipeline parallelism by
hand ([pipeline.py](../../app/src/training/parallel/pipeline.py)), one mesh layout
for every combination ([mesh.py](../../app/src/training/parallel/mesh.py)), and two
Qwen3 plans in the model module: `parallelize_tensor_parallel` (hand) and
`parallelize_dtensor` (PyTorch's `parallelize_module`). The CPU study is
[tp_pp.py](../../app/src/training/experiments/tp_pp.py); numbers are in
[RESULTS.md](RESULTS.md).

______________________________________________________________________

## Tensor parallelism: Megatron's f and g, as two autograd functions

`copy_to_tp` (identity forward, all-reduce backward) and `reduce_from_tp` (all-reduce
forward, identity backward) are the only communication. `ColumnParallelLinear` starts
with `copy_to_tp`; `RowParallelLinear` ends with `reduce_from_tp`. A column layer
followed by a row layer is one all-reduce each way, and everything between them
(SwiGLU, per-head attention, QK-norm, RoPE) runs on local shards with no
communication.

- **Pros.** The communication is visible in two 10-line functions; the layers are
  plain `nn.Module`s holding plain tensors, so a test can compare any shard with the
  dense weight. Every layer is built from an initialized dense layer, so the sharded
  model starts bit-identical to the dense one.
- **Cons.** Each layer needs its own sharding rule, and the attention module's head
  counts must be divided by hand.

## Vocabulary parallelism and its cross-entropy

The embedding and the tied head are split by vocabulary rows. The loss is a custom
autograd function: all-reduce the row max, the sum of exponentials, and the target
logit; backward is `softmax - one_hot` on the local slice. It matches
`F.cross_entropy` to 1e-10 in float64, in loss and gradients.

**Why.** Full fp32 logits for one Qwen3 micro-batch are 2.5 GB; gathering them to
compute a standard loss would undo the point of splitting the head.

## Pipelines: schedules are data

`gpipe` and `one_f_one_b` return each stage's action list. `validate` checks the
invariants (every F before its B, at most p - s in flight for 1F1B), `simulate`
replays it with per-stage times and gives the bubble, and `run_stage` executes one
stage with point-to-point sends.

- **Pros.** The bubble table and the in-flight bound are checked before any GPU is
  booked; a new schedule (interleaved, zero-bubble) is a new list generator, and the
  same validator and simulator apply.
- **Cons.** The executor is synchronous per stage and recomputes nothing; it does
  not overlap send/receive with compute the way production schedules do.

Sends are `isend`: under 1F1B two neighbouring stages can send to each other at the
same moment (activations forward, gradients backward), and two blocking sends
deadlock. The first version used blocking sends and hung the test.

## Stages balanced by cost

`partition` is an exact dynamic program over contiguous splits minimizing the most
expensive stage. For Qwen3-1.7B it puts the tied LM head (about 6.2 blocks of
compute) with 2 blocks on the last stage.

## The mesh fixes the order

`ParallelismSpec(dp_replicate, dp_shard, tp, pp)` builds one mesh with dims
`("pp", "dp_replicate", "dp_shard", "tp")`: TP innermost (consecutive ranks, one
node), PP outermost. `validate` asserts the TP degree divides heads, KV heads, FFN
width, and vocabulary.

## Routes not to take

- **Two all-reduces per MLP (row then column, or column then column with a gather).**
  Splitting the first matmul by input features makes its output a partial sum that
  must be reduced before the elementwise SwiGLU, then the second matmul needs another
  collective. Column then row is the only order with one all-reduce each way.
- **Initializing each shard with its own random init.** It looks efficient and
  breaks parity with the dense model from step 0, which then looks like a math bug
  in the layers. Initialize once in full, then shard.
- **Gathering the logits for the loss.** It brings back the 2.5 GB tensor that
  vocabulary parallelism exists to avoid, on every rank.
- **Balancing pipeline stages by layer count.** With a tied 151,936-word head the
  last stage gets 13.2 block-equivalents against 7.0 elsewhere; the whole pipeline
  then runs at the slowest stage's pace (1.54x the mean).
- **Expecting 1F1B to shrink the bubble.** It does not; it caps held activations at
  p micro-batches. Shrinking the bubble takes more micro-batches (m >= 4p) or
  virtual stages (interleaving).
- **Blocking sends between pipeline stages.** Deadlocks under 1F1B (see above).
- **A TP group across nodes.** Nothing errors; it is just 10-20x slower. The mesh
  order makes the safe layout the default.
- **Jumping straight to DTensor or `torch.distributed.pipelining`.** They are the
  production path (v4, v6), but without the hand version the placements and
  schedules are opaque. Here the DTensor plan is checked against the same dense
  model as the hand plan.

## Not built

Sequence parallelism is implemented as primitives and layer flags
(`scatter_sequence`, `gather_sequence`, `sequence_parallel=True`) with a round-trip
test, but not wired through the full model: Qwen3's attention reads the sequence
length from its input, so the whole residual stream would need sequence shards. The
ticket cuts it first after 3D. `torch.distributed.pipelining` (v6), the GPU timelines,
2D/3D throughput, and PCIe vs NVLink need GPUs.
