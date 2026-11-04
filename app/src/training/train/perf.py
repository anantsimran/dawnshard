"""perf.py -- predict FLOPs, MFU, and memory from a model's shape, and time steps.

Everything here works from a `ModelShape`, plain numbers any decoder config can fill
in, so a prediction can be written down before the model is built. The predictions
are meant to miss: a ledger row records the prediction, the measurement, and why
they differ (allocator slack, temporaries, kernels that save more than the formula).

- FLOPs per token = 6N + 12 * layers * heads * head_dim * seq. Each matmul weight
  does a multiply-add per token forward (2 FLOPs) and two backward (4 FLOPs); the
  second term is the attention scores and their weighted sum, which have no weights.
  N counts weights that take part in a matmul: an untied embedding lookup is not
  one, a tied LM head is.
- MFU = achieved FLOPs/s / the device's dense peak.
- Static memory = weights + grads + optimizer state (+ fp32 master weights under
  mixed precision). Activation memory counts the tensors each layer saves for
  backward; logits are their own term because a large vocabulary makes them the
  biggest single activation.
- Arithmetic intensity (FLOPs per byte moved) places a kernel on the roofline.
"""

import time
from dataclasses import dataclass, field
from typing import Literal, Optional

import torch

type Checkpointing = Literal["none", "full", "selective"]


@dataclass(frozen=True)
class ModelShape:
    """The numbers the formulas need from a decoder-only transformer config."""

    num_parameters: int
    num_layers: int
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    tied_head: bool


def matmul_parameters(shape: ModelShape) -> int:
    """Parameters that take part in a matmul: all of them, less an untied embedding lookup."""
    embedding = shape.vocab_size * shape.hidden_size
    return shape.num_parameters if shape.tied_head else shape.num_parameters - embedding


def flops_per_token(shape: ModelShape, seq_len: int) -> float:
    """Training FLOPs per token, forward plus backward: 6N + attention."""
    attention = 12 * shape.num_layers * shape.num_heads * shape.head_dim * seq_len
    return 6 * matmul_parameters(shape=shape) + attention


def model_flops_utilization(
    tokens_per_s: float, flops_per_token: float, peak_flops: float
) -> float:
    """MFU: the share of the device's dense peak the model's FLOPs achieve."""
    return tokens_per_s * flops_per_token / peak_flops


@dataclass(frozen=True)
class MemoryPrediction:
    """Predicted peak bytes, term by term."""

    weights: int
    grads: int
    optimizer: int
    activations: int
    logits: int

    @property
    def static(self) -> int:
        """Bytes held for the whole run, whatever the batch."""
        return self.weights + self.grads + self.optimizer

    @property
    def total(self) -> int:
        """Static plus the activations and logits of one micro-batch."""
        return self.static + self.activations + self.logits


def activation_elements_per_token(shape: ModelShape, checkpointing: Checkpointing) -> int:
    """Elements one layer saves for backward per token.

    Without checkpointing: the two norm inputs and their outputs (4d), queries and
    keys before their norms (h*dk + kv*dk), the post-RoPE q, k, v and the attention
    output that a fused attention kernel saves (2*h*dk + 2*kv*dk), and the SwiGLU's
    gate, activation, up, and product (4*ff). Selective checkpointing recomputes the
    MLP and drops its 4*ff; full checkpointing keeps only the layer's input.
    """
    d, ff = shape.hidden_size, shape.intermediate_size
    q, kv = shape.num_heads * shape.head_dim, shape.num_kv_heads * shape.head_dim
    if checkpointing == "full":
        return d
    attention = 4 * d + 3 * q + 3 * kv
    return attention if checkpointing == "selective" else attention + 4 * ff


def predict_memory(
    shape: ModelShape,
    tokens: int,
    weight_bytes: int = 4,
    grad_bytes: int = 4,
    optimizer_bytes: int = 8,
    activation_bytes: int = 4,
    logits_bytes: int = 4,
    checkpointing: Checkpointing = "none",
    chunked_loss: bool = False,
) -> MemoryPrediction:
    """Predict peak memory for one micro-batch of `tokens` tokens.

    Args:
        weight_bytes: Per parameter for weights, plus any fp32 master copy: 4 in
            fp32; 2 + 4 for bf16 weights with fp32 masters.
        optimizer_bytes: Per parameter: 8 for Adam's two fp32 moments.
        logits_bytes: Per logit. The loss holds the logits and backward allocates
            their gradient, so the term is twice `tokens * vocab * logits_bytes`.
        chunked_loss: The loss never materializes full logits; the term drops to 0.
    """
    n = shape.num_parameters
    per_token = activation_elements_per_token(shape=shape, checkpointing=checkpointing)
    logits = 0 if chunked_loss else 2 * tokens * shape.vocab_size * logits_bytes
    return MemoryPrediction(
        weights=n * weight_bytes,
        grads=n * grad_bytes,
        optimizer=n * optimizer_bytes,
        activations=shape.num_layers * tokens * per_token * activation_bytes,
        logits=logits,
    )


type ShardingStrategy = Literal["ddp", "zero1", "zero2", "zero3"]


def static_bytes_per_parameter(
    strategy: ShardingStrategy,
    world_size: int,
    low_precision_bytes: int = 2,
    fp32_state_bytes: int = 12,
) -> float:
    """Static bytes per parameter per GPU under mixed precision.

    Weights and grads in a low-precision dtype (2 bytes each for bf16); fp32 master
    weights and Adam's two moments are `fp32_state_bytes` (12). Each ZeRO stage
    shards one more of them across `world_size` ranks: stage 1 the fp32 state,
    stage 2 also the grads, stage 3 (FSDP) also the weights.
    """
    w, g, state, n = low_precision_bytes, low_precision_bytes, fp32_state_bytes, world_size
    per_strategy = {
        "ddp": w + g + state,
        "zero1": w + g + state / n,
        "zero2": w + (g + state) / n,
        "zero3": (w + g + state) / n,
    }
    return per_strategy[strategy]


def static_bytes(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    """Measured static memory: every parameter, gradient, and optimizer-state tensor."""
    tensors = [p for p in model.parameters()]
    tensors += [p.grad for p in model.parameters() if p.grad is not None]
    tensors += [v for s in optimizer.state.values() for v in s.values() if torch.is_tensor(v)]  # noqa: NAR001
    return sum(t.numel() * t.element_size() for t in tensors)  # noqa: NAR001


def matmul_intensity(m: int, n: int, k: int, bytes_per_element: int) -> float:
    """FLOPs per byte of an (m, k) @ (k, n) matmul that reads both inputs and writes once."""
    return 2 * m * n * k / (bytes_per_element * (m * k + k * n + m * n))


def elementwise_intensity(
    flops_per_element: float, tensors_moved: int, bytes_per_element: int
) -> float:
    """FLOPs per byte of an elementwise op that reads and writes `tensors_moved` tensors."""
    return flops_per_element / (tensors_moved * bytes_per_element)


@dataclass
class StepTimer:
    """Train probe: time each optimizer step, synchronizing CUDA first.

    Called by `train_step` after every micro-batch. With accumulation, an optimizer
    step ends every `accumulation_steps` calls; the timer stamps those ends, so the
    gaps between stamps are whole steps, launch and sync included.

    Attributes:
        warmup_steps: Leading steps left out of `tokens_per_s`: compilation,
            allocator growth, and cuDNN autotuning happen there.
    """

    accumulation_steps: int
    tokens_per_step: int
    warmup_steps: int = 20
    stamps: list[float] = field(default_factory=list)

    def __call__(self, model, x, y, predicted, epoch, batch_index, seed) -> None:  # noqa: NAR002
        if (batch_index + 1) % self.accumulation_steps:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.stamps.append(time.perf_counter())  # noqa: NAR001

    def tokens_per_s(self) -> Optional[float]:
        """Steady-state tokens/s after `warmup_steps`, or None if too few steps ran."""
        steady = self.stamps[self.warmup_steps :]
        if len(steady) < 2:  # noqa: NAR001
            return None
        return (len(steady) - 1) * self.tokens_per_step / (steady[-1] - steady[0])  # noqa: NAR001
