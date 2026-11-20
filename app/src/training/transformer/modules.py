"""Transformer building blocks: position encodings, attention, encoder and decoder.

Shapes use the axis vocabulary in this package's README: B batch, L sequence
length, S key length, T cached key length, d_model embedding width, h heads,
d_k = d_model // h. The same README
explains how the blocks name their norms (before each sublayer) and dropouts
(after it).
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
from beartype import beartype
from jaxtyping import Bool, Float, jaxtyped
from torch import nn
from training.transformer.constants import D_MODEL, B, H, L, S, T
from training.transformer.functions import attention

# A config names its norm as a string, not the class, so it reads the same in a log
# line, a history file, and a checkpoint; this maps it to the class a block's `norm`
# argument takes.
NORMS: dict[str, type[nn.LayerNorm] | type[nn.RMSNorm]] = {
    "layernorm": nn.LayerNorm,
    "rmsnorm": nn.RMSNorm,
}


class SinusoidalEmbedding(nn.Module):
    """Add fixed sinusoidal position encodings to token embeddings.

    Attention is permutation invariant: it sees a set of vectors, not a
    sequence. This is what tells it where each token sat, by adding a
    position-dependent vector to every embedding. In and out are both
    (B, L, d_model), so it drops in front of an encoder stack.

    The table is built once at construction and never trained:

    ```
        positions p = 0 … seq_length-1     feature pairs i = 0 … d_model/2-1
                    │                        d_model_by_2_arr (d_model/2,)
                    │                                │
                    │        10000 ** (-2i / d_model)│
                    └────────────► outer product ◄───┘
                                        │  angles (seq_length, d_model/2)
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
                sin(angles)                            cos(angles)
                    │                                       │
                    ▼  even columns                         ▼  odd columns
                pe[:, 0::2]                             pe[:, 1::2]
                    └───────────────────┬───────────────────┘
                                        │  .pe (seq_length, d_model)
                                        ▼
                                     pe[0:L]                 forward reads the
                                        │  (L, d_model)      first L rows only
    batch (B, L, d_model) ─────────────► + ──────────► (B, L, d_model)
    ```

    Each feature *pair* `(2i, 2i+1)` shares one frequency and differs by a
    quarter turn, so the pair traces a circle as `p` advances — which is what
    makes a shift in position a fixed rotation of that pair, the property the
    encoding exists for. The frequencies span wavelengths from 2π at `i = 0` to
    10000·2π at the last pair, so one table carries both adjacent-token order
    and sentence-scale position.

    Every entry is a sine or a cosine, so the table is bounded in [-1, 1] and
    sits on the same scale as an `nn.Embedding` output. The paper's `√d_model`
    scaling of the embeddings is not needed here, because `nn.Embedding`
    initializes to N(0, 1) rather than to something smaller.

    Fixed, not learned: no parameters, and any position below `seq_length`
    works whether or not training ever saw a sequence that long. `forward`
    refuses anything above it rather than reading off the end of the table.

    `pe` is a non-persistent buffer: it moves with `.to(device)` like a
    parameter, but stays out of `state_dict`, because it is a pure function of
    `seq_length` and `d_model` and there is nothing in it to checkpoint.
    """

    pe: torch.Tensor

    def __init__(self, seq_length: int, d_model: int) -> None:
        """Precompute the (seq_length, d_model) position table.

        Args:
            seq_length: Longest sequence the table covers. `forward` raises
                above it, so this is the model's hard context limit.
            d_model: Embedding width, matching the embeddings this is added to.
                Even widths only: the sine half and the cosine half have to be
                the same size.
        """
        super().__init__()
        d_model_by_2_arr = torch.arange(0, d_model, 2)  # noqa: NAR001
        d_model_by_2_arr = 10000 ** (-d_model_by_2_arr / d_model)
        pos_arr = torch.arange(0, seq_length)  # noqa: NAR001
        angles = pos_arr.unsqueeze(dim=1) * d_model_by_2_arr
        pe = torch.empty(seq_length, d_model)  # noqa: NAR001
        pe[:, 0::2] = torch.sin(input=angles)
        pe[:, 1::2] = torch.cos(input=angles)
        self.register_buffer(name="pe", tensor=pe, persistent=False)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, batch: Float[torch.Tensor, f"{B} {L} {D_MODEL}"]
    ) -> Float[torch.Tensor, f"{B} {L} {D_MODEL}"]:
        """Add the position table to `batch`, broadcasting over the batch.

        Args:
            batch: Token embeddings. Positions are read off the sequence axis,
                so a right-padded row gets encodings on its pad positions too;
                the padding mask is what keeps those out of attention and the
                pool.

        Raises:
            ValueError: If the sequence is longer than the table.
        """
        seq_length = batch.shape[1]
        if seq_length > self.pe.shape[0]:
            raise ValueError(  # noqa: NAR001
                f"sequence length {seq_length} exceeds table {self.pe.shape[0]}"
            )
        return batch + self.pe[0:seq_length, :]

    def at_pos(self, pos: int) -> Float[torch.Tensor, f"{D_MODEL}"]:
        """Return the encoding vector for one position.

        Raises:
            ValueError: If `pos` is outside [0, seq_length).
        """
        if not 0 <= pos < self.pe.shape[0]:
            raise ValueError(  # noqa: NAR001
                f"position {pos} outside table of length {self.pe.shape[0]}"
            )
        return self.pe[pos]


@dataclass
class KVCache:
    """One attention layer's keys and values for one decode, filled from the front.

    The caller allocates the buffers and passes the cache to the layer on every
    step, so the layer itself still keeps nothing between calls.

    Attributes:
        k: Key buffer, (B, h, capacity, d_k). Rows from `length_filled` on hold
            nothing yet. A cross-attention cache's capacity must equal S.
        v: Value buffer, the same shape as `k`.
        length_filled: How many rows of `k` and `v` hold real keys and values.
    """

    k: torch.Tensor
    v: torch.Tensor
    length_filled: int = 0

    def append(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Write `k` and `v` after the filled rows, then return every filled row.

        Raises:
            ValueError: If the new rows would run past the buffer's capacity.

        Side effects:
            Writes into `self.k` and `self.v` and advances `self.length_filled`.
        """
        end = self.length_filled + k.shape[2]
        if end > self.k.shape[2]:
            raise ValueError(  # noqa: NAR001
                f"cache holds {self.k.shape[2]} rows, append needs {end}"
            )
        self.k[:, :, self.length_filled : end] = k
        self.v[:, :, self.length_filled : end] = v
        self.length_filled = end
        return self.k[:, :, : self.length_filled], self.v[:, :, : self.length_filled]


class MultiHeadAttentionLayer(nn.Module):
    """Multi-head attention: self-attention over `x`, or cross-attention to `memory`.

    Queries always come from `x`. Keys and values come from `memory` when one is
    passed and from `x` otherwise, so one class serves both an encoder's
    self-attention and a decoder's cross-attention to the encoder output. `x` is
    (B, L, d_model), `memory` is (B, S, d_model), and the output is
    (B, L, d_model): one vector per query, whatever the key length. Each source is
    projected into `h` heads of width `d_k = d_model // h`, optionally QK-normed,
    attended within each head, then merged and mixed with `w_o`.

    ```
          x (B, L, d_model)       memory (B, S, d_model)     or x if None, S = L
              │                          │
              │                 ┌────────┴────────┐
              ▼                 ▼                 ▼
          @ w_q             @ w_k             @ w_v      (h, d_model, d_k)
              ▼                 ▼                 │
           q_norm            k_norm               │      RMSNorm over d_k,
              │                 │                 │      only if qk_norm
              │                 ▼                 ▼
              │           ┌─────┴─────────────────┴──┐   self: append, then
              │           │ kv_cache, if passed      │   attend to all filled
              │           └─────┬─────────────────┬──┘   rows, so S = T; cross:
              │                 │                 │      fill once, then read
              ▼                 ▼                 ▼
            q (B,h,L,d_k)     k (B,h,S,d_k)     v (B,h,S,d_k)
              └────────────────►│◄────────────────┘
                                ▼
                     attention(q, k, v, mask=mask)      weights (B, h, L, S)
                                │  out (B, h, L, d_k)
                                ▼
                     transpose(1, 2) + reshape          merge the heads
                                │  (B, L, d_model)
                                ▼
                             @ w_o                      (d_model, d_model),
                                │                       mixes across heads
                                ▼  (B, L, d_model)
    ```

    The three projections are one batched matmul each, not `h` separate ones:
    `x.unsqueeze(1)` is (B, 1, L, d_model) and broadcasts against the (h,
    d_model, d_k) weight, so the head axis appears without a reshape. The
    per-head outputs only become a single vector at `w_o` — attention itself
    never mixes across heads.

    A `kv_cache` sits after `k_norm`, so it holds exactly the keys attention
    consumed and QK-norm runs once per token. The caller allocates it and passes it
    on every step, which keeps decode state off the module; the package README's
    "KV cache" section explains why.

    `forward` returns `(out, weights)` and holds no state between calls; a passed
    `kv_cache` is the only thing it writes.
    """

    def __init__(self, d_model: int, h: int, qk_norm: bool) -> None:
        """Raises ValueError if `d_model` is not divisible by `h`.

        Args:
            qk_norm: RMSNorm each query and key over `d_k`, per head, before the
                dot product, so attention scores stay bounded as weights grow.
        """
        super().__init__()
        if d_model % h != 0:
            raise ValueError(  # noqa: NAR001
                f"d_model={d_model} must be divisible by h={h}"
            )
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        scale = 1.0 / math.sqrt(d_model)  # noqa: NAR001
        self.w_q = nn.Parameter(
            data=torch.randn(h, d_model, self.d_k) * scale  # noqa: NAR001
        )
        self.w_k = nn.Parameter(
            data=torch.randn(h, d_model, self.d_k) * scale  # noqa: NAR001
        )
        self.w_v = nn.Parameter(
            data=torch.randn(h, d_model, self.d_k) * scale  # noqa: NAR001
        )
        self.w_o = nn.Parameter(
            data=torch.randn(d_model, d_model) * scale  # noqa: NAR001
        )
        self.q_norm = nn.RMSNorm(normalized_shape=self.d_k) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(normalized_shape=self.d_k) if qk_norm else nn.Identity()

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[torch.Tensor, f"{B} {L} {D_MODEL}"],
        mask: Optional[Bool[torch.Tensor, f"#{B} #{H} #{L} {S}"]],
        memory: Optional[Float[torch.Tensor, f"{B} {S} {D_MODEL}"]] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> tuple[
        Float[torch.Tensor, f"{B} {L} {D_MODEL}"],
        Float[torch.Tensor, f"{B} {H} {L} {S}"],
    ]:
        """Attend from `x` to itself, or to `memory` when one is given.

        Args:
            mask: Passed straight to `attention`; True marks a visible key. Build
                it with `transformer.masks.mask_tokenizer_pad_ids`, combined with
                `mask_later_positions` by `&` where needed. For cross-attention it
                masks the padding of `memory`, not of `x`. None attends
                everywhere.
            memory: Source of keys and values, such as an encoder's output. None
                makes this self-attention over `x`.
            kv_cache: This layer's cache for one decode, or None to cache nothing.
                Self-attention appends the keys and values of `x` and attends to
                every filled row, so `x` holds only the new positions. Cross-
                attention projects `memory` on the first call and reads the cache
                after that.

        Returns:
            (out, weights): the mixed output, and each head's post-softmax
            attention weights as returned by `attention`.

        Raises:
            ValueError: If a filled cross-attention cache holds a different number
                of source rows than `memory`, or a self-attention append would run
                past the cache's capacity.

        Side effects:
            Writes into `kv_cache` and advances its `length_filled`.
        """
        batch_size, n, d_model = x.shape
        q = self.q_norm(x.unsqueeze(dim=1) @ self.w_q)  # noqa: NAR001
        if memory is not None:
            if kv_cache is not None and kv_cache.length_filled > 0:
                if memory.shape[1] != kv_cache.length_filled:
                    raise ValueError(  # noqa: NAR001
                        f"cross cache holds {kv_cache.length_filled} source rows, "
                        f"memory has {memory.shape[1]}"
                    )
                k, v = kv_cache.k, kv_cache.v
            else:
                k = self.k_norm(memory.unsqueeze(dim=1) @ self.w_k)  # noqa: NAR001
                v = memory.unsqueeze(dim=1) @ self.w_v
                if kv_cache is not None:
                    kv_cache.k.copy_(other=k)
                    kv_cache.v.copy_(other=v)
                    kv_cache.length_filled = k.shape[2]
        else:
            k = self.k_norm(x.unsqueeze(dim=1) @ self.w_k)  # noqa: NAR001
            v = x.unsqueeze(dim=1) @ self.w_v
            if kv_cache is not None:
                k, v = kv_cache.append(k=k, v=v)
        out, weights = attention(q=q, k=k, v=v, mask=mask)
        concat = out.transpose(dim0=1, dim1=2).reshape(  # noqa: NAR001
            batch_size, n, d_model
        )
        return concat @ self.w_o, weights


class EncoderBlock(nn.Module):
    """One pre-norm encoder block: self-attention, then a position-wise FFN.

    Each sublayer is wrapped in a residual connection, with the norm on the
    sublayer branch and dropout on the sublayer output. In and out are both
    (B, L, d_model), so blocks stack by composition.

    ```
    x ──┬────────────────────────────► + ──┬──────────────────────────► + ──►
        │                              ▲   │                           ▲
        └─ attn_in_norm ─ attn ─ drop ─┘   └─ ffn_norm ─ ffn ─ drop ────┘
    ```

    The FFN is the same two-layer MLP applied to every position independently:

    ```
    (B, L, d_model)
          │
          ▼
        ffn[0]             Linear, .weight (d_ff, d_model),
          │                d_ff = d_ff_multiplier * d_model
          ▼  (B, L, d_ff)
         ReLU
          │
          ▼
        ffn[2]             Linear, .weight (d_model, d_ff)
          │
          ▼  (B, L, d_model)
    ```

    Pre-norm, not post-norm: `x` reaches every add untouched, so there is a
    path from input to output that passes through no norm at all and gradients
    reach early blocks undamped. The trade is that block outputs grow with
    depth, which a final norm outside the stack is expected to absorb.

    One `dropout` module serves both sublayers. A Dropout holds no state — it
    samples a fresh mask on every call — so a second instance would add an entry
    to `named_modules` and nothing else.

    `norm` picks the class behind both `attn_in_norm` and `ffn_norm`, so a block
    never mixes the two kinds. `nn.LayerNorm` is the default; `nn.RMSNorm` drops
    the mean-centering and the bias, which is cheaper and usually just as stable.

    Attention is the only place positions talk to each other; the norms, the
    FFN and the dropout all act per position.
    """

    def __init__(
        self,
        d_model: int,
        h: int,
        qk_norm: bool = False,
        d_ff_multiplier: int = 4,
        dropout: float = 0.1,
        norm: type[nn.LayerNorm] | type[nn.RMSNorm] = nn.LayerNorm,
    ) -> None:
        """Build the block's two sublayers and their norms.

        Args:
            qk_norm: Forwarded to `MultiHeadAttentionLayer`; RMSNorms queries
                and keys per head before the dot product.
            d_ff_multiplier: FFN hidden width as a multiple of `d_model`.
            dropout: Drop probability on each sublayer's output, before its
                residual add.
            norm: Class instantiated for `attn_in_norm` and `ffn_norm`, called
                as `norm(normalized_shape=d_model)`.

        Raises:
            ValueError: If `d_model` is not divisible by `h`.
        """
        super().__init__()

        self.attn_in_norm = norm(normalized_shape=d_model)

        self.attn = MultiHeadAttentionLayer(d_model=d_model, h=h, qk_norm=qk_norm)

        self.ffn = nn.Sequential(  # noqa: NAR001
            nn.Linear(in_features=d_model, out_features=d_ff_multiplier * d_model),
            nn.ReLU(),
            nn.Linear(in_features=d_ff_multiplier * d_model, out_features=d_model),
        )
        self.ffn_norm = norm(normalized_shape=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        batch: Float[torch.Tensor, f"{B} {L} {D_MODEL}"],
        pad_mask: Optional[Bool[torch.Tensor, f"#{B} #{H} #{L} {L}"]],
    ) -> tuple[
        Float[torch.Tensor, f"{B} {L} {D_MODEL}"],
        Float[torch.Tensor, f"{B} {H} {L} {L}"],
    ]:
        """Run `batch` through the attention sublayer, then the FFN sublayer.

        Args:
            batch: Token embeddings.
            pad_mask: Passed through to the attention sublayer; True marks a
                visible key. Build it with `transformer.masks.mask_tokenizer_pad_ids`.
                None attends everywhere.

        Returns:
            (out, weights): the block output, same shape as `batch`, and the
            attention sublayer's weights.
        """
        attn_in_normalized = self.attn_in_norm(batch)  # noqa: NAR001
        attn, weights = self.attn(x=attn_in_normalized, mask=pad_mask)
        batch = batch + self.dropout(attn)  # attn layer is completed  # noqa: NAR001
        ffn_in_normalized = self.ffn_norm(batch)  # noqa: NAR001
        ffn_output = self.ffn(ffn_in_normalized)  # noqa: NAR001
        return batch + self.dropout(ffn_output), weights  # noqa: NAR001


@dataclass
class DecoderBlockCache:
    """One decoder block's caches, one per attention sublayer.

    Attributes:
        self_attn: Gains one row per decoded token; capacity is the longest target.
        cross_attn: Filled from `memory` on the first step; capacity is S.
    """

    self_attn: KVCache
    cross_attn: KVCache


class DecoderBlock(nn.Module):
    """Pre-norm decoder block: causal self-attention → cross-attention → FFN.

    The decoder half of an encoder-decoder. `x` is the target so far,
    (B, L, d_model), and `memory` is the encoder stack's output, (B, S, d_model).
    Self-attention lets each target position read the ones before it,
    cross-attention lets it read every real source position, and the FFN then
    transforms each position alone. In and out are both (B, L, d_model), so blocks
    stack by composition, and every block in the stack reads the same `memory`.

    ```
    x (B, L, d_model)                                     target embeddings
    │
    ├──────────────────► self_attn_norm
    │                          ▼
    │                      self_attn                      mask=x_mask: causal
    │                          ├──► self_attn_weights     & target padding;
    │                          ▼        (B, h, L, T)      kv_cache=self_cache
    │                       dropout
    ▼                          │
    + ◄────────────────────────┘
    │  (B, L, d_model)
    ├──────────────────► cross_attn_norm   memory (B, S, d_model)
    │                          ▼              │           keys and values only
    │                      cross_attn ◄───────┘
    │                          ├──► cross_attn_weights    mask=memory_mask:
    │                          ▼        (B, h, L, S)      source padding;
    │                          │                          kv_cache=cross_cache
    │                       dropout
    ▼                          │
    + ◄────────────────────────┘
    │  (B, L, d_model)
    ├──────────────────► ffn_norm
    │                          ▼
    │                         ffn                         Linear ─► ReLU ─► Linear,
    │                          ▼                          ffn[0].weight (d_ff, d_model)
    │                       dropout                       ffn[2].weight (d_model, d_ff)
    ▼                          │
    + ◄────────────────────────┘
    │
    ▼  (B, L, d_model)
    ```

    Every sublayer is wrapped the way `EncoderBlock`'s are: the norm on the way in,
    `dropout` on the way out, and the residual stream `x` never normed. The
    package README's "Inside a block" section lays out the naming. Each norm reads
    the running sum `x`, not the previous sublayer's output, so cross-attention's
    queries carry the token's own embedding directly, not only what self-attention
    chose to pass along.

    `cross_attn_norm` normalizes the queries only. `memory` goes into `cross_attn`
    as it arrives, so the caller passes the encoder stack's output after its final
    norm, once, to every block.

    `self_attn` and `cross_attn` are two instances of `MultiHeadAttentionLayer`,
    not one shared one: reading the target and reading the source are different
    jobs, so they learn separate projections. The same class serves both because
    passing `memory` is what switches it to cross-attention.

    `x_mask` has no `None` default. Decoder self-attention without a causal mask
    lets every position read its own future, so the caller has to pass one. The
    one exception is a cached decode step with a single new position, which may
    read every cached key; `forward` raises on `None` for anything longer.
    `memory_mask` may be `None` when no source sequence is padded.

    With a `DecoderBlockCache`, `x` holds only the new positions and the
    self-attention keys are the whole prefix so far, so the self-attention weights
    are (B, h, L, T) with T the prefix length. Without one, T is L.

    `forward` returns `(out, self_attn_weights, cross_attn_weights)` and holds no
    state between calls; a passed cache is the only thing it writes.
    """

    def __init__(
        self,
        d_model: int,
        h: int,
        qk_norm: bool = False,
        d_ff_multiplier: int = 4,
        dropout: float = 0.1,
        norm: type[nn.LayerNorm] | type[nn.RMSNorm] = nn.LayerNorm,
    ) -> None:
        """Build the block's three sublayers, their norms, and the shared dropout.

        Args:
            qk_norm: Forwarded to both attention sublayers; RMSNorms queries and
                keys per head before the dot product.
            d_ff_multiplier: FFN hidden width as a multiple of `d_model`.
            dropout: Drop probability on each sublayer's output, before its
                residual add.
            norm: Class instantiated for `self_attn_norm`, `cross_attn_norm`, and
                `ffn_norm`, called as `norm(normalized_shape=d_model)`.

        Raises:
            ValueError: If `d_model` is not divisible by `h`.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.self_attn_norm = norm(normalized_shape=d_model)
        self.self_attn = MultiHeadAttentionLayer(d_model=d_model, h=h, qk_norm=qk_norm)

        self.cross_attn_norm = norm(normalized_shape=d_model)
        self.cross_attn = MultiHeadAttentionLayer(d_model=d_model, h=h, qk_norm=qk_norm)

        self.ffn_norm = norm(normalized_shape=d_model)
        self.ffn = nn.Sequential(  # noqa: NAR001
            nn.Linear(in_features=d_model, out_features=d_ff_multiplier * d_model),
            nn.ReLU(),
            nn.Linear(in_features=d_ff_multiplier * d_model, out_features=d_model),
        )

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[torch.Tensor, f"{B} {L} {D_MODEL}"],
        memory: Float[torch.Tensor, f"{B} {S} {D_MODEL}"],
        x_mask: Optional[Bool[torch.Tensor, f"#{B} #{H} #{L} {T}"]],
        memory_mask: Optional[Bool[torch.Tensor, f"#{B} #{H} #{L} {S}"]],
        decoder_block_cache: Optional[DecoderBlockCache] = None,
    ) -> tuple[
        Float[torch.Tensor, f"{B} {L} {D_MODEL}"],
        Float[torch.Tensor, f"{B} {H} {L} {T}"],
        Float[torch.Tensor, f"{B} {H} {L} {S}"],
    ]:
        """Run `x` through self-attention, cross-attention to `memory`, then the FFN.

        Args:
            x: Target embeddings.
            memory: The encoder stack's output after its final norm. The block
                does not norm it.
            x_mask: True marks a visible key. Build it as `mask_later_positions &
                mask_tokenizer_pad_ids` over the target ids, shape (B, 1, L, L).
                None only for a single position on a cached decode step.
            memory_mask: True marks a visible source key. Build it with
                `mask_tokenizer_pad_ids` over the source ids, shape (B, 1, 1, S).
                None attends to every source position.
            decoder_block_cache: This block's caches for one decode, or None to
                cache nothing. With a cache, `x` holds only the new positions.

        Returns:
            (out, self_attn_weights, cross_attn_weights): the block output, same
            shape as `x`, and each attention sublayer's post-softmax weights.

        Raises:
            ValueError: If `x_mask` is None and `x` has more than one position.
            ValueError: From either attention sublayer, if its cache is misused.

        Side effects:
            Writes into both caches of `decoder_block_cache`.
        """
        if x_mask is None and x.shape[1] > 1:
            raise ValueError(  # noqa: NAR001
                f"x_mask may be None only for one position, got {x.shape[1]}"
            )
        self_cache, cross_cache = (
            (None, None)
            if decoder_block_cache is None
            else (decoder_block_cache.self_attn, decoder_block_cache.cross_attn)
        )
        self_attn_in_normalized = self.self_attn_norm(x)  # noqa: NAR001
        self_attn, self_attn_weights = self.self_attn(
            x=self_attn_in_normalized, mask=x_mask, kv_cache=self_cache
        )
        x = x + self.dropout(self_attn)  # noqa: NAR001

        cross_attn_in_normalized = self.cross_attn_norm(x)  # noqa: NAR001
        cross_attn, cross_attn_weights = self.cross_attn(
            x=cross_attn_in_normalized,
            memory=memory,
            mask=memory_mask,
            kv_cache=cross_cache,
        )
        x = x + self.dropout(cross_attn)  # noqa: NAR001

        ffn_in_normalized = self.ffn_norm(x)  # noqa: NAR001
        ffn_output = self.ffn(ffn_in_normalized)  # noqa: NAR001
        x = x + self.dropout(ffn_output)  # noqa: NAR001
        return x, self_attn_weights, cross_attn_weights
