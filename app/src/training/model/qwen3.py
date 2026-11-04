"""
qwen3.py
A Qwen3-style decoder-only language model built from a config, random init, and its
distributed training entry point.

The config carries the official Qwen3 fields (`hidden_size`, `num_key_value_heads`,
`head_dim`, `rope_theta`, ...) and module attributes use the official names
(`q_proj`, `input_layernorm`, `embed_tokens`), so a later ticket can load released
weights by name. `QWEN3_0_6B` and `QWEN3_1_7B` are the released shapes; their
parameter counts are checked against 0.596B and 1.72B on the meta device, without
allocating them.

`main` trains for a fixed number of steps on a flat uint32 token file (or random
tokens) through `fit`, under `torchrun` or alone. It initializes the process group
from the `torchrun` environment unless one exists already (Ray Train creates its
own), wraps the model in DistributedDataParallel when the world is larger than one,
and records launch telemetry: when the process started, when the process group was
ready, and when the first step finished, next to the image digest and backend the
launcher put in the environment. Rank 0 writes it to
`HISTORY_PATH/<run_id>.launch.json`.

    uv run torchrun --standalone --nproc-per-node=2 \\
        -m training.model.qwen3 --config tiny --synthetic
"""

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path
from uuid import uuid4

import torch
import torch.distributed as dist
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, RandomSampler
from training.constants import HISTORY_PATH, TEST_SEED, VAL_SEED
from training.dataload.tokens import TokenWindows, lm_collate, write_tokens
from training.metrics.loss_only_metrics import (
    accumulate_metrics,
    calculate_metrics,
    reduce_metrics,
)
from training.setup import DEVICE
from training.train.model import EpochSpec, Metrics, TrainState
from training.train.train_loop import fit
from training.transformer.constants import D_MODEL, B, L, V

PROCESS_START = time.time()


@dataclass(frozen=True)
class Qwen3Config:
    """The fields of a Qwen3 `config.json` that decide the architecture."""

    vocab_size: int = 151936
    hidden_size: int = 1024
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 3072
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 40960
    tie_word_embeddings: bool = True


QWEN3_0_6B = Qwen3Config()
QWEN3_1_7B = replace(QWEN3_0_6B, hidden_size=2048, intermediate_size=6144)  # noqa: NAR001
# The benchmark size for launch and performance work: 4 layers, 512 wide.
QWEN3_SMALL = replace(  # noqa: NAR001
    QWEN3_0_6B,
    hidden_size=512,
    num_hidden_layers=4,
    num_attention_heads=8,
    num_key_value_heads=4,
    head_dim=64,
    intermediate_size=1536,
    max_position_embeddings=4096,
)
# For tests and CPU smoke runs.
QWEN3_TINY = replace(  # noqa: NAR001
    QWEN3_SMALL,
    vocab_size=512,
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    intermediate_size=128,
    max_position_embeddings=256,
)
CONFIGS = {"tiny": QWEN3_TINY, "small": QWEN3_SMALL, "0.6b": QWEN3_0_6B, "1.7b": QWEN3_1_7B}


def count_parameters(config: Qwen3Config) -> int:
    """Return the parameter count `Qwen3(config)` has, in closed form.

    Per layer: q, k, v, o projections (no biases), two per-head QK norms, the three
    SwiGLU matrices, and two RMSNorms. Plus the embedding, the final norm, and an LM
    head only when it is not tied to the embedding.
    """
    d, kv = config.hidden_size, config.num_key_value_heads * config.head_dim
    q = config.num_attention_heads * config.head_dim
    attention = d * q + 2 * d * kv + q * d + 2 * config.head_dim
    mlp = 3 * d * config.intermediate_size
    per_layer = attention + mlp + 2 * d
    embedding = config.vocab_size * d
    head = 0 if config.tie_word_embeddings else embedding
    return config.num_hidden_layers * per_layer + embedding + d + head


def rope_tables(config: Qwen3Config) -> tuple[torch.Tensor, torch.Tensor]:
    """Return `(cos, sin)`, each `(max_position_embeddings, head_dim)`.

    Frequencies `rope_theta ** (-2i / head_dim)` for the first half of the head,
    repeated for the second half, the layout `rotate_half` expects.
    """
    half = torch.arange(start=0, end=config.head_dim, step=2, dtype=torch.float32)
    inv_freq = config.rope_theta ** (-half / config.head_dim)
    positions = torch.arange(end=config.max_position_embeddings, dtype=torch.float32)
    angles = torch.outer(input=positions, vec2=inv_freq)
    angles = torch.cat(tensors=[angles, angles], dim=-1)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate each head's feature pairs `(i, i + head_dim/2)` by its position's angle.

    Args:
        x: `(B, heads, L, head_dim)`.
        cos: `(L, head_dim)` rows for positions `0 .. L-1`; `sin` likewise.
    """
    first, second = x.chunk(chunks=2, dim=-1)
    rotated = torch.cat(tensors=[-second, first], dim=-1)
    return x * cos + rotated * sin


class Qwen3Attention(nn.Module):
    """Grouped-query causal self-attention with per-head QK RMSNorm and RoPE.

    In and out are `(B, L, d_model)`. Queries get `num_attention_heads` heads; keys
    and values get only `num_key_value_heads`, each shared by a group of
    `num_attention_heads / num_key_value_heads` query heads.

    ```
    x (B, L, d_model)
        ├────────────────────┬────────────────────┐
        ▼                    ▼                    ▼
     q_proj               k_proj               v_proj       no biases
        │  (B, L, h·d_k)     │  (B, L, kv·d_k)    │
        ▼                    ▼                    │
     q_norm               k_norm                  │         RMSNorm over d_k, per head
        ▼                    ▼                    │
     apply_rope           apply_rope              │         cos/sin rows 0..L-1
        │  (B, h, L, d_k)    │  (B, kv, L, d_k)   │  (B, kv, L, d_k)
        └────────────────────┼────────────────────┘
                             ▼
          F.scaled_dot_product_attention(is_causal=True, enable_gqa=True)
                             │  (B, h, L, d_k)
                             ▼
                  transpose(1, 2) + reshape
                             │  (B, L, h·d_k)
                             ▼
                          o_proj
                             ▼  (B, L, d_model)
    ```

    Grouped-query attention cuts the KV cache by `h / kv` at decode time, which is
    what it is for; in training it only saves the two smaller projections. SDPA does
    the grouping itself (`enable_gqa`), so no `repeat_interleave` copy of K and V is
    made, and SDPA can pick a flash or memory-efficient kernel.

    QK-norm before RoPE, as Qwen3 does, bounds attention logits without a softcap.

    `forward` holds no state between calls.
    """

    def __init__(self, config: Qwen3Config) -> None:
        """Build the four projections and the two per-head norms."""
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        d, q, kv = (
            config.hidden_size,
            self.num_heads * self.head_dim,
            self.num_kv_heads * self.head_dim,
        )
        self.q_proj = nn.Linear(in_features=d, out_features=q, bias=False)
        self.k_proj = nn.Linear(in_features=d, out_features=kv, bias=False)
        self.v_proj = nn.Linear(in_features=d, out_features=kv, bias=False)
        self.o_proj = nn.Linear(in_features=q, out_features=d, bias=False)
        self.q_norm = nn.RMSNorm(normalized_shape=self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(normalized_shape=self.head_dim, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Return the attention output for `x`, causally masked.

        Args:
            x: `(B, L, d_model)`.
            cos: RoPE rows for positions `0 .. L-1`, `(L, head_dim)`; `sin` likewise.
        """
        batch, length, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(batch, length, self.num_heads, self.head_dim))  # noqa: NAR001
        k = self.k_norm(self.k_proj(x).view(batch, length, self.num_kv_heads, self.head_dim))  # noqa: NAR001
        v = self.v_proj(x).view(batch, length, self.num_kv_heads, self.head_dim)  # noqa: NAR001
        q = apply_rope(x=q.transpose(dim0=1, dim1=2), cos=cos, sin=sin)
        k = apply_rope(x=k.transpose(dim0=1, dim1=2), cos=cos, sin=sin)
        out = F.scaled_dot_product_attention(
            query=q, key=k, value=v.transpose(dim0=1, dim1=2), is_causal=True, enable_gqa=True
        )
        return self.o_proj(out.transpose(dim0=1, dim1=2).reshape(batch, length, -1))  # noqa: NAR001


class Qwen3MLP(nn.Module):
    """SwiGLU feed-forward: `down_proj(silu(gate_proj(x)) * up_proj(x))`.

    Applied to each position alone; in and out are `(B, L, d_model)`.

    ```
    x (B, L, d_model)
        ├──────────────┐
        ▼              ▼
     gate_proj      up_proj          (B, L, intermediate_size), no biases
        ▼              │
       silu            │
        └────► * ◄─────┘
               ▼
           down_proj
               ▼  (B, L, d_model)
    ```

    The gate is a learned, smooth per-feature switch on the up projection. Three
    matrices instead of two cost 50% more parameters at the same width, which is why
    `intermediate_size` is 3x rather than 4x `hidden_size`.

    `forward` holds no state between calls.
    """

    def __init__(self, config: Qwen3Config) -> None:
        """Build the gate, up, and down projections."""
        super().__init__()
        d, ff = config.hidden_size, config.intermediate_size
        self.gate_proj = nn.Linear(in_features=d, out_features=ff, bias=False)
        self.up_proj = nn.Linear(in_features=d, out_features=ff, bias=False)
        self.down_proj = nn.Linear(in_features=ff, out_features=d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the gated MLP output for `x`, `(B, L, d_model)`."""
        return self.down_proj(F.silu(input=self.gate_proj(x)) * self.up_proj(x))  # noqa: NAR001


class Qwen3Block(nn.Module):
    """Pre-norm decoder layer: RMSNorm → attention → add, RMSNorm → SwiGLU → add.

    In and out are `(B, L, d_model)`, so layers stack by composition.

    ```
    x ──┬─────────────────────────────────► + ──┬──────────────────────────────► + ──►
        │                                   ▲   │                                ▲
        └─ input_layernorm ─ self_attn ─────┘   └─ post_attention_layernorm ─ mlp ┘
    ```

    No dropout: Qwen3 pretrains without it, and at this data scale a single pass
    over the tokens does not overfit. The residual stream is never normed inside a
    layer; `Qwen3.norm` normalizes it once after the last layer.

    `forward` holds no state between calls.
    """

    def __init__(self, config: Qwen3Config) -> None:
        """Build the attention and MLP sublayers and their input norms."""
        super().__init__()
        d, eps = config.hidden_size, config.rms_norm_eps
        self.input_layernorm = nn.RMSNorm(normalized_shape=d, eps=eps)
        self.self_attn = Qwen3Attention(config=config)
        self.post_attention_layernorm = nn.RMSNorm(normalized_shape=d, eps=eps)
        self.mlp = Qwen3MLP(config=config)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Return the layer output for `x`; `cos` and `sin` as `Qwen3Attention.forward`."""
        x = x + self.self_attn(x=self.input_layernorm(x), cos=cos, sin=sin)  # noqa: NAR001
        return x + self.mlp(x=self.post_attention_layernorm(x))  # noqa: NAR001


class Qwen3(nn.Module):
    """Embedding → `num_hidden_layers` pre-norm decoder layers → RMSNorm → tied LM head.

    A causal language model: position `l` of the output predicts token `l + 1`. In is
    `(B, L)` token ids; out is `(B*L, vocab)` logits, flattened like the repo's other
    sequence models so `CrossEntropyLoss` takes a flat `(B*L,)` target.

    ```
    input_ids (B, L)
        ▼
    embed_tokens                          (vocab, d_model), normal(0, 0.02) init
        │  (B, L, d_model)
        ▼
    Qwen3Block × num_hidden_layers        rope_cos / rope_sin rows 0..L-1
        │  (B, L, d_model)
        ▼
      norm                                final RMSNorm
        │  hidden_states (B, L, d_model)
        ▼
    @ embed_tokens.weight.T               tied LM head; lm_head only if untied
        │  (B, L, vocab)
        ▼
    reshape                               row b*L + l
        ▼  (B*L, vocab)
    ```

    `hidden_states` stops before the head. A loss that never materializes the full
    `(B*L, vocab)` logits, such as a chunked cross-entropy, starts there and reads
    `lm_head_weight`.

    The RoPE tables are non-persistent buffers: they follow `.to(device)` but are a
    pure function of the config, so they stay out of `state_dict`.

    Output: raw logits (no softmax). `forward` holds no state between calls.
    """

    rope_cos: torch.Tensor
    rope_sin: torch.Tensor

    def __init__(self, config: Qwen3Config) -> None:
        """Build the embedding, the layers, the final norm, the head, and the RoPE tables."""
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
        )
        self.layers = nn.ModuleList(
            modules=[Qwen3Block(config=config) for _ in range(config.num_hidden_layers)]  # noqa: NAR001
        )  # noqa: NAR001
        self.norm = nn.RMSNorm(normalized_shape=config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = (
            None
            if config.tie_word_embeddings
            else nn.Linear(
                in_features=config.hidden_size, out_features=config.vocab_size, bias=False
            )
        )
        cos, sin = rope_tables(config=config)
        self.register_buffer(name="rope_cos", tensor=cos, persistent=False)
        self.register_buffer(name="rope_sin", tensor=sin, persistent=False)
        self.apply(self._init_weights)  # noqa: NAR001

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Normal(0, 0.02) for every linear and embedding weight, as Qwen3's config sets."""
        if isinstance(module, (nn.Linear, nn.Embedding)):  # noqa: NAR001
            nn.init.normal_(tensor=module.weight, mean=0.0, std=0.02)

    @property
    def lm_head_weight(self) -> torch.Tensor:
        """The `(vocab, d_model)` output projection: the embedding when tied."""
        return self.embed_tokens.weight if self.lm_head is None else self.lm_head.weight

    @jaxtyped(typechecker=beartype)
    def hidden_states(
        self, input_ids: Int[torch.Tensor, f"{B} {L}"]
    ) -> Float[torch.Tensor, f"{B} {L} {D_MODEL}"]:
        """Return the final-normed hidden states, before the LM head."""
        length = input_ids.shape[1]
        cos = self.rope_cos[:length].to(dtype=self.embed_tokens.weight.dtype)
        sin = self.rope_sin[:length].to(dtype=self.embed_tokens.weight.dtype)
        x = self.embed_tokens(input_ids)  # noqa: NAR001
        for layer in self.layers:
            x = layer(x=x, cos=cos, sin=sin)
        return self.norm(x)  # noqa: NAR001

    @jaxtyped(typechecker=beartype)
    def forward(
        self, input_ids: Int[torch.Tensor, f"{B} {L}"]
    ) -> Float[torch.Tensor, f"{B}*{L} {V}"]:
        """Return next-token logits for every position, flattened to `(B*L, vocab)`."""
        hidden = self.hidden_states(input_ids=input_ids)
        return (hidden @ self.lm_head_weight.T).reshape(-1, self.config.vocab_size)  # noqa: NAR001


def init_distributed() -> tuple[int, int, torch.device]:
    """Join the `torchrun` process group if there is one; return `(rank, world, device)`.

    Uses NCCL on CUDA and gloo otherwise. Leaves an existing group alone, which is
    how it runs under Ray Train.

    Side effects:
        May initialize the default process group and set the CUDA device.
    """
    world = int(os.environ.get("WORLD_SIZE", "1"))  # noqa: NAR001
    if world > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    rank = dist.get_rank() if dist.is_initialized() else 0
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))  # noqa: NAR001
        torch.cuda.set_device(device=local_rank)
        return rank, world, torch.device("cuda", local_rank)  # noqa: NAR001
    return rank, world, DEVICE if world == 1 else torch.device("cpu")  # noqa: NAR001


def record_first_step(telemetry: dict, model, x, y, predicted, epoch, batch_index, seed) -> None:  # noqa: NAR002
    """Train probe: stamp the first finished step's wall time into `telemetry`."""
    if batch_index == 0 and epoch == 1:
        telemetry["first_step"] = time.time()


def parse_args() -> argparse.Namespace:
    """Parse the training command line; see the module docstring."""
    parser = argparse.ArgumentParser(description="Train a Qwen3-style decoder for N steps.")
    parser.add_argument("--config", default="small", choices=list(CONFIGS))  # noqa: NAR001
    parser.add_argument("--tokens", type=Path, help="Flat uint32 token file.")  # noqa: NAR001
    parser.add_argument("--synthetic", action="store_true", help="Random tokens instead.")  # noqa: NAR001
    parser.add_argument("--steps", type=int, default=200)  # noqa: NAR001
    parser.add_argument("--batch-size", type=int, default=8)  # noqa: NAR001
    parser.add_argument("--seq-len", type=int, default=256)  # noqa: NAR001
    parser.add_argument("--lr", type=float, default=3e-4)  # noqa: NAR001
    parser.add_argument("--seed", type=int, default=0)  # noqa: NAR001
    return parser.parse_args()


def build_loader(args: argparse.Namespace, config: Qwen3Config, rank: int) -> DataLoader:
    """Return a loader of exactly `args.steps` batches, drawn with a per-rank seed.

    Side effects:
        With `--synthetic`, writes a random token file under HISTORY_PATH.
    """
    path = args.tokens
    if args.synthetic:
        path = HISTORY_PATH / f"synthetic_tokens_{config.vocab_size}.bin"
        generator = torch.Generator().manual_seed(args.seed)  # noqa: NAR001
        ids = torch.randint(
            low=0, high=config.vocab_size, size=(args.seq_len * 4096 + 1,), generator=generator
        )
        if rank == 0 and not path.exists():
            write_tokens(path=path, ids=ids.tolist())
        if dist.is_initialized():
            dist.barrier()
    dataset = TokenWindows(path=path, seq_len=args.seq_len)
    sampler = RandomSampler(
        data_source=dataset,
        replacement=True,
        num_samples=args.steps * args.batch_size,
        generator=torch.Generator().manual_seed(args.seed + rank),  # noqa: NAR001
    )
    return DataLoader(
        dataset=dataset, batch_size=args.batch_size, sampler=sampler, collate_fn=lm_collate
    )


def main() -> None:
    """Train for `--steps` steps, alone or under `torchrun`, and write launch telemetry."""
    args = parse_args()
    rank, world, device = init_distributed()
    telemetry: dict = {"process_start": PROCESS_START, "process_group_ready": time.time()}
    config = CONFIGS[args.config]
    torch.manual_seed(seed=args.seed)
    model: nn.Module = Qwen3(config=config).to(device=device)
    if world > 1:
        model = DistributedDataParallel(module=model)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr)
    state = TrainState(model=model, optimizer=optimizer, criterion=nn.CrossEntropyLoss())
    loader = build_loader(args=args, config=config, rank=rank)
    spec = EpochSpec(
        device=device,
        metrics_type=Metrics,
        calculate_metrics=calculate_metrics,
        accumulate_metrics=accumulate_metrics,
        reduce_metrics=reduce_metrics,
        train_probe=partial(record_first_step, telemetry),  # noqa: NAR001
    )
    run_id = os.environ.get("DAWNSHARD_RUN_ID") or str(uuid4())  # noqa: NAR001
    fit(
        state=state,
        epoch_spec=spec,
        train_loader=loader,
        val_loader=loader,
        num_epochs=1,
        val_epoch_list=[],
        train_seed=args.seed,
        val_seed=VAL_SEED,
        test_seed=TEST_SEED,
        history_path=HISTORY_PATH / f"{run_id}.json"
        if rank == 0
        else HISTORY_PATH / f"{run_id}.rank{rank}.json",
    )
    if rank == 0:
        write_launch_telemetry(run_id=run_id, telemetry=telemetry, config=config, world=world)
    if dist.is_initialized():
        dist.destroy_process_group()


def write_launch_telemetry(run_id: str, telemetry: dict, config: Qwen3Config, world: int) -> None:
    """Write the launch timeline and what launched it to `HISTORY_PATH/<run_id>.launch.json`.

    The launcher's side of the timeline (submitted, scheduled, image pulled) comes from
    the environment variables the backend sets and from the orchestrator's events.

    Side effects:
        Writes the file.
    """
    payload = {
        **telemetry,
        "submitted": float(os.environ.get("DAWNSHARD_SUBMIT_TIME", "nan")),  # noqa: NAR001
        "backend": os.environ.get("DAWNSHARD_BACKEND"),  # noqa: NAR001
        "image": os.environ.get("DAWNSHARD_IMAGE"),  # noqa: NAR001
        "world_size": world,
        "config": asdict(obj=config),
        "num_parameters": count_parameters(config=config),
    }
    payload = {
        key: (None if isinstance(value, float) and math.isnan(value) else value)  # noqa: NAR001
        for key, value in payload.items()
    }  # noqa: NAR001
    HISTORY_PATH.mkdir(parents=True, exist_ok=True)
    (HISTORY_PATH / f"{run_id}.launch.json").write_text(data=json.dumps(obj=payload, indent=2))


if __name__ == "__main__":
    main()
