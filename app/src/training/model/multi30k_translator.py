"""
multi30k_translator.py
Encoder-decoder translator for Multi30k sentence pairs (see dataload.multi30k for
loading).

The loader yields `(X, Y)` int64 batches. X stacks the source ids and the target input
ids, `(B, 2, SEQ_LEN)`: the target input starts with `BOS_ID`, and Y is the same target
shifted one to the left and ended with `EOS_ID`, flattened to `(B*SEQ_LEN,)`, with
`CROSS_ENTROPY_IGNORE_INDEX` on its padding. So every real target position is scored
on the next token, which teacher forcing needs.

`main` trains one model built from the constants below, or, given `decode`, scores
a saved bundle by decoding a split with greedy or beam search. The ablation is in
`experiments/multi30k_translator_ablation.py`; its `best` study writes the bundle.

A bundle (`save_bundle`) is everything decoding needs in one file: the constructor
arguments, the weights, and both tokenizers' merges as id pairs. Pinning the merges in
the bundle means a later retrained default tokenizer cannot silently change what the
ids mean.

Train from the repo root; losses are logged to the terminal after every epoch:

    uv run python app/src/training/model/multi30k_translator.py

Decode the test split with beam search and a KV cache, appending BLEU, speed, and
cache memory to experiments/t1-decoding/results.csv:

    uv run python app/src/training/model/multi30k_translator.py decode --strategy beam --beam-size 4
"""

import argparse
import csv
import hashlib
import random
import time
from datetime import date
from functools import partial
from pathlib import Path
from typing import Optional, cast
from uuid import uuid4

import sacrebleu
import torch
from beartype import beartype
from jaxtyping import Bool, Float, Int, jaxtyped
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from training.common import bpe
from training.common.constants import BOS_ID, EOS_ID, PAD_ID
from training.constants import (
    ATTENTION_PROBES_PATH,
    CHECKPOINTS_PATH,
    REPO_ROOT,
    TEST_SEED,
    VAL_SEED,
)
from training.dataload.multi30k import get_multi30k_translation_dataloader, read_multi30k_split
from training.metrics.masked_token_metrics import (
    MaskedTokenMetrics,
    accumulate_metrics,
    calculate_metrics,
    reduce_metrics,
)
from training.setup import DEVICE, init_wandb
from training.train.model import EpochSpec, TrainState
from training.train.train_loop import fit
from training.transformer.constants import D_MODEL, B, L, S, V
from training.transformer.decoding import BeamSearch, Greedy, Strategy, generate
from training.transformer.kv_cache import KVCache
from training.transformer.mask import causal_keep_mask, padding_keep_mask
from training.transformer.modules import (
    DecoderBlock,
    EncoderBlock,
    SinusoidalEmbedding,
)
from training.transformer.probes import save_attention_maps
from training.utils.git import get_git_commit
from training.viz.attention_heatmap import plot_attention_probes

SOURCE_LANG = "en"
TARGET_LANG = "de"
D_MODEL_SIZE = 512
SEQ_LEN = 32
NUM_SRC_MERGES = 10000
NUM_TGT_MERGES = 100000
DROPOUT_RATE = 0.3
NUM_ENCODERS = 6
NUM_DECODERS = 6
NUM_HEADS = 4
FFN_MULTIPLIER = 6
QK_NORM = False
NORM: type[nn.LayerNorm] | type[nn.RMSNorm] = nn.LayerNorm
COSINE_ANNEALING = True
PROBE_SENTENCES = 20
BATCH_SIZE = 64
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
BUNDLE_PATH = CHECKPOINTS_PATH / "multi30k_translator_bundle.pt"
DECODING_RESULTS_PATH = REPO_ROOT / "experiments" / "t1-decoding"
DECODE_MAX_LEN = 32
DECODE_BATCH_SIZE = 64


class Multi30kTranslator(nn.Module):
    """Encoder over the source → causal decoder over the target → next-token logits.

    Translates by teacher forcing: given the source sentence and the target sentence so
    far, predicts the next target token at every target position at once. In is one
    `(B, 2, L)` tensor, the source ids at index 0 and the target input ids at index 1;
    out are `(B*L, vocab)` logits over the target vocab, one row per target position.

    ```
                batch (B, 2, L)                     src at index 0, tgt_in at 1
         ┌─────────────┴─────────────┐
         ▼  src (B, L)               ▼  tgt_in (B, L)
     src_embedding             tgt_embedding        padding_idx=PAD_ID, one per
         │                           │              tokenizer
         ▼                           ▼
      dropout                     dropout           on token content only
         │                           │
         ▼                           ▼
    positional_embedding     positional_embedding   one fixed sin/cos table
         │  (B, S, d_model)          │  (B, L, d_model)
         ▼                           │
    EncoderBlock × num_encoders      │              encoder_mask: source padding,
         │                           │              weights (B, h, S, S) each
         ▼                           │
     decoder_norm                    │              norms the memory, once
         │  memory (B, S, d_model)   ▼
         └──────────► DecoderBlock × num_decoders   decoder_mask: causal & target
                                     │              padding; memory_mask=encoder_mask
                                     ▼
                                 pred_norm
                                     │
                                     ▼
                                   pred             Linear to the target vocab
                                     │  (B, L, vocab)
                                     ▼
                                  reshape           one row per position, b*L + l
                                     │
                                     ▼  logits (B*L, vocab)
    ```

    Source and target arrive stacked in one tensor because the training loop calls
    `model(X)` with a single input and has nowhere to pass a second one. Both masks are
    derived inside `forward` from `PAD_ID` for the same reason.

    The two embeddings are separate, not tied: source and target have their own
    tokenizers, so the same id names a different token on each side.

    `dropout` runs before `positional_embedding`, as in `Multi30kEnPredictor`, so it
    zeroes features of the token embedding only and every position keeps its full
    sin/cos signal. One `nn.Dropout` serves both sides, because a Dropout holds no
    state. One position table serves both sides too, sized to the longer of the two
    lengths: it is a fixed function of position, so sharing it shares no weights.

    `decoder_norm` is the encoder stack's final norm. The stack is pre-norm, so its
    output is an unnormed residual, and `DecoderBlock` does not norm `memory` itself;
    norming it here, once, hands every decoder block the same normed memory.
    `pred_norm` is the decoder stack's final norm, for the same reason.

    `forward` flattens the logits to `(B*L, vocab)` because `CrossEntropyLoss` reads
    dim 1 as the class axis. The loader flattens its target the same way, so row
    `b*L + l` of both is the same position.

    Decoding splits `forward` in two. `encode` runs the left column once per source
    and returns the normed memory. `decode_step` runs the right column for one step
    and returns the next-token logits of each row's last position: with a `KVCache`
    it embeds only the newest token, at position offset `cache.length`, and calls
    `DecoderBlock.forward_step`; without one it reruns the whole prefix through
    `DecoderBlock.forward`. The training path is untouched, so a decoding change
    cannot move a training number.

    `hparams` holds the constructor arguments, the norm by class name, so
    `save_bundle` can rebuild the model without the caller restating its shape.

    Output: raw logits (no softmax). Every `forward` overwrites `attention_map`;
    `encode` and `decode_step` touch no attribute.
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int,
        src_seq_length: int,
        tgt_seq_length: int,
        dropout: float,
        num_encoder_heads: int,
        num_decoder_heads: int,
        encoder_ffn_multiplier: int,
        decoder_ffn_multiplier: int,
        num_encoders: int,
        num_decoders: int,
        qk_norm: bool,
        norm: type[nn.LayerNorm] | type[nn.RMSNorm] = nn.LayerNorm,
    ) -> None:
        """Build both embeddings, the position table, both stacks, and the head.

        Args:
            src_vocab_size: `len(src_vocab)` from `get_multi30k_translation_dataloader`.
            tgt_vocab_size: `len(tgt_vocab)` from the same call; the logits' width.
            src_seq_length: Longest source `L` that `forward` accepts.
            tgt_seq_length: Longest target `L` that `forward` accepts.
            dropout: Used on both embeddings and inside every block.
            qk_norm: Forwarded to every block; RMSNorms queries and keys per head
                before the dot product.
            norm: Class behind every norm in the model, each block's and both final
                ones, so one setting switches them together.

        Raises:
            ValueError: If `d_model` is not divisible by `num_encoder_heads` or by
                `num_decoder_heads`, from `EncoderBlock` and `DecoderBlock`.
        """
        super().__init__()
        self.hparams = {
            "src_vocab_size": src_vocab_size,
            "tgt_vocab_size": tgt_vocab_size,
            "d_model": d_model,
            "src_seq_length": src_seq_length,
            "tgt_seq_length": tgt_seq_length,
            "dropout": dropout,
            "num_encoder_heads": num_encoder_heads,
            "num_decoder_heads": num_decoder_heads,
            "encoder_ffn_multiplier": encoder_ffn_multiplier,
            "decoder_ffn_multiplier": decoder_ffn_multiplier,
            "num_encoders": num_encoders,
            "num_decoders": num_decoders,
            "qk_norm": qk_norm,
            "norm": norm.__name__,
        }
        self.dropout = nn.Dropout(p=dropout)

        self.src_embedding = nn.Embedding(
            num_embeddings=src_vocab_size, embedding_dim=d_model, padding_idx=PAD_ID
        )
        self.tgt_embedding = nn.Embedding(
            num_embeddings=tgt_vocab_size, embedding_dim=d_model, padding_idx=PAD_ID
        )
        self.src_seq_length = src_seq_length
        self.tgt_seq_length = tgt_seq_length
        self.positional_embedding = SinusoidalEmbedding(
            seq_length=max(src_seq_length, tgt_seq_length),  # noqa: NAR001
            d_model=d_model,
        )

        self.encoder_blocks = nn.ModuleList(
            modules=[
                EncoderBlock(
                    d_model=d_model,
                    dropout=dropout,
                    h=num_encoder_heads,
                    d_ff_multiplier=encoder_ffn_multiplier,
                    norm=norm,
                    qk_norm=qk_norm,
                )
                for _ in range(num_encoders)  # noqa: NAR001
            ]
        )

        self.decoder_norm = norm(normalized_shape=d_model)

        self.decoder_blocks = nn.ModuleList(
            modules=[
                DecoderBlock(
                    d_model=d_model,
                    dropout=dropout,
                    h=num_decoder_heads,
                    d_ff_multiplier=decoder_ffn_multiplier,
                    norm=norm,
                    qk_norm=qk_norm,
                )
                for _ in range(num_decoders)  # noqa: NAR001
            ]
        )

        self.pred_norm = norm(normalized_shape=d_model)
        self.pred = nn.Linear(in_features=d_model, out_features=tgt_vocab_size)
        # Overwritten by every forward: one detached (B, h, L, S) tensor per block, keyed
        # by kind ("encoder", "decoder_self", "decoder_cross").
        self.attention_map: dict[str, list[torch.Tensor]] = {}

    @jaxtyped(typechecker=beartype)
    def forward(
        self, batch: Int[torch.Tensor, f"{B} 2 {L}"]
    ) -> Float[torch.Tensor, f"{B}*{L} {V}"]:
        """Return next-token logits for every target position of a batch of pairs.

        Args:
            batch: Source ids at index 0 and target input ids at index 1, each
                right-padded with `PAD_ID` and no longer than its `seq_length`. Both
                masks come from `PAD_ID` alone.

        Returns:
            Raw logits, `(B*L, vocab)`. Row `b*L + l` is target position `l` of pair
            `b`, the order the loader flattens its target in.

        Raises:
            ValueError: If either side is longer than the position table.

        Side effects:
            Replaces `self.attention_map` with this call's detached attention weights
            under `"encoder"`, `"decoder_self"`, and `"decoder_cross"`, one tensor per
            block in block order.
        """
        src, tgt_in = batch[:, 0], batch[:, 1]
        src_vector = self.positional_embedding(
            batch=self.dropout(self.src_embedding(src))  # noqa: NAR001
        )
        tgt_vector = self.positional_embedding(
            batch=self.dropout(self.tgt_embedding(tgt_in))  # noqa: NAR001
        )

        # Self attention
        encoder_mask = padding_keep_mask(token_ids=src, pad_id=PAD_ID)
        encoder_attn = []
        for encoder in self.encoder_blocks:
            src_vector, src_attn = encoder(batch=src_vector, pad_mask=encoder_mask)
            encoder_attn.append(src_attn.detach())  # noqa: NAR001
        decoder_pad_mask = padding_keep_mask(token_ids=tgt_in, pad_id=PAD_ID)
        decoder_causal_mask = causal_keep_mask(seq_len=tgt_in.shape[1], device=tgt_in.device)
        decoder_mask = decoder_pad_mask & decoder_causal_mask

        src_vector = self.decoder_norm(src_vector)  # noqa: NAR001
        decoder_self_attn, decoder_cross_attn = [], []
        for decoder in self.decoder_blocks:
            tgt_vector, tgt_self_attn, tgt_cross_attn = decoder(
                x=tgt_vector, memory=src_vector, x_mask=decoder_mask, memory_mask=encoder_mask
            )
            decoder_self_attn.append(tgt_self_attn.detach())  # noqa: NAR001
            decoder_cross_attn.append(tgt_cross_attn.detach())  # noqa: NAR001

        self.attention_map = {
            "encoder": encoder_attn,
            "decoder_self": decoder_self_attn,
            "decoder_cross": decoder_cross_attn,
        }
        logits = self.pred(self.pred_norm(tgt_vector))  # noqa: NAR001
        return logits.reshape(-1, logits.shape[-1])  # noqa: NAR001

    @jaxtyped(typechecker=beartype)
    def encode(
        self, src: Int[torch.Tensor, f"{B} {S}"]
    ) -> tuple[Float[torch.Tensor, f"{B} {S} {D_MODEL}"], Bool[torch.Tensor, f"{B} 1 1 {S}"]]:
        """Return `(memory, memory_mask)`: the normed encoder output and the source keep mask.

        Args:
            src: Source ids starting with `BOS_ID`, right-padded with `PAD_ID`.
        """
        memory = self.positional_embedding(batch=self.dropout(self.src_embedding(src)))  # noqa: NAR001
        memory_mask = padding_keep_mask(token_ids=src, pad_id=PAD_ID)
        for encoder in self.encoder_blocks:
            memory, _ = encoder(batch=memory, pad_mask=memory_mask)
        return self.decoder_norm(memory), memory_mask  # noqa: NAR001

    @jaxtyped(typechecker=beartype)
    def decode_step(
        self,
        tokens: Int[torch.Tensor, f"{B} {L}"],
        memory: Float[torch.Tensor, f"{B} {S} {D_MODEL}"],
        memory_mask: Bool[torch.Tensor, f"{B} 1 1 {S}"],
        cache: Optional[KVCache],
    ) -> Float[torch.Tensor, f"{B} {V}"]:
        """Return next-token logits for the last position of every row of `tokens`.

        Args:
            tokens: Every target token so far, starting with `BOS_ID`. With a cache,
                only `tokens[:, cache.length:]` is read.
            memory: From `encode`, repeated per beam to match `tokens`' rows.
            memory_mask: From `encode`, repeated the same way.
            cache: Earlier positions' keys and values, or None to recompute them.

        Side effects:
            Appends this step's keys and values to `cache`.
        """
        if cache is None:
            x = self.positional_embedding(batch=self.dropout(self.tgt_embedding(tokens)))  # noqa: NAR001
            causal = causal_keep_mask(seq_len=tokens.shape[1], device=tokens.device)
            for decoder in self.decoder_blocks:
                x, _, _ = decoder(x=x, memory=memory, x_mask=causal, memory_mask=memory_mask)
        else:
            offset = cache.length
            x = self.positional_embedding(
                batch=self.dropout(self.tgt_embedding(tokens[:, offset:])),  # noqa: NAR001
                offset=offset,
            )
            for layer, decoder in enumerate(self.decoder_blocks):  # noqa: NAR001
                x = cast(DecoderBlock, decoder).forward_step(  # noqa: NAR001
                    x=x, memory=memory, memory_mask=memory_mask, cache=cache, layer=layer
                )
        return self.pred(self.pred_norm(x[:, -1]))  # noqa: NAR001

    def new_cache(self, batch_size: int, max_len: int, cache_cross: bool = True) -> KVCache:
        """Return an empty `KVCache` sized for this decoder, on its device and dtype.

        Args:
            batch_size: Rows decoded together: sentences x beams.
            max_len: Most target positions fed, BOS included.
            cache_cross: Forwarded to `KVCache`.
        """
        attn = cast(DecoderBlock, self.decoder_blocks[0]).self_attn  # noqa: NAR001
        return KVCache(
            num_layers=len(self.decoder_blocks),  # noqa: NAR001
            batch_size=batch_size,
            h=attn.h,
            d_k=attn.d_k,
            max_len=max_len,
            dtype=attn.w_k.dtype,
            device=attn.w_k.device,
            cache_cross=cache_cross,
        )


def save_bundle(
    model: Multi30kTranslator,
    src_merges: list[bpe.Phrase],
    tgt_merges: list[bpe.Phrase],
    path: Path,
) -> None:
    """Write the model's hparams, weights, and both tokenizers' merges to one file.

    Side effects:
        Creates `path`'s parent and overwrites `path`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        obj={
            "hparams": model.hparams,
            "model": model.state_dict(),
            "src_merges": bpe.merge_pairs(merges=src_merges),
            "tgt_merges": bpe.merge_pairs(merges=tgt_merges),
        },
        f=path,
    )


def load_bundle(
    path: Path, device: torch.device
) -> tuple[Multi30kTranslator, list[bpe.Phrase], list[bpe.Phrase], dict[int, str]]:
    """Return `(model, src_merges, tgt_merges, tgt_vocab)` from a `save_bundle` file.

    The model is on `device` and in eval mode.
    """
    bundle = torch.load(f=path, map_location=device)
    hparams = {**bundle["hparams"], "norm": getattr(nn, bundle["hparams"]["norm"])}  # noqa: NAR001
    model = Multi30kTranslator(**hparams).to(device=device)
    model.load_state_dict(state_dict=bundle["model"])
    model.eval()
    src_merges, _ = bpe.tokenizer_from_pairs(pairs=bundle["src_merges"])
    tgt_merges, tgt_vocab = bpe.tokenizer_from_pairs(pairs=bundle["tgt_merges"])
    return model, src_merges, tgt_merges, tgt_vocab


@torch.inference_mode()
def translate(
    model: Multi30kTranslator,
    src: torch.Tensor,
    strategy: Strategy,
    max_len: int,
    use_cache: bool,
    cache_cross: bool = True,
) -> tuple[list[list[int]], int, list[float]]:
    """Decode a batch of sources; return `(ids, cache_bytes, step_seconds)`.

    Args:
        src: `(B, S)` source ids as `encode` takes them.
        max_len: Most output tokens per sentence; at most the model's
            `tgt_seq_length`, because the cache and the position table end there.
        use_cache: Decode with a `KVCache`; False reruns the prefix every step.
        cache_cross: Forwarded to `new_cache`; ignored without a cache.

    Returns:
        Output ids per sentence without BOS or EOS, the cache's bytes (0 without
        one), and the wall time of each step, synchronized on CUDA.
    """
    model.eval()
    memory, memory_mask = model.encode(src=src)
    memory = memory.repeat_interleave(repeats=strategy.beam_size, dim=0)
    memory_mask = memory_mask.repeat_interleave(repeats=strategy.beam_size, dim=0)
    rows = src.shape[0] * strategy.beam_size
    cache = (
        model.new_cache(batch_size=rows, max_len=max_len, cache_cross=cache_cross)
        if use_cache
        else None
    )
    step_seconds: list[float] = []

    def step_fn(tokens: torch.Tensor, step_cache: Optional[KVCache]) -> torch.Tensor:
        """Time one `decode_step` call."""
        start = time.perf_counter()
        logits = model.decode_step(
            tokens=tokens, memory=memory, memory_mask=memory_mask, cache=step_cache
        )
        if logits.device.type == "cuda":
            torch.cuda.synchronize()
        step_seconds.append(time.perf_counter() - start)  # noqa: NAR001
        return logits

    ids = generate(
        step_fn=step_fn,
        strategy=strategy,
        batch_size=src.shape[0],
        bos_id=BOS_ID,
        max_len=max_len,
        device=src.device,
        cache=cache,
    )
    return ids, cache.nbytes() if cache is not None else 0, step_seconds


def _source_ids(rows: list[str], merges: list[bpe.Phrase], seq_len: int) -> torch.Tensor:
    """Encode `rows` as the training loader does: BOS, ids cut to `seq_len`, PAD after."""
    encoded = [[BOS_ID] + ids[: seq_len - 1] for ids in bpe.encode_many(texts=rows, merges=merges)]
    padded = [ids + [PAD_ID] * (seq_len - len(ids)) for ids in encoded]  # noqa: NAR001
    return torch.tensor(data=padded, dtype=torch.int64, device=DEVICE)


def decode_split(
    model: Multi30kTranslator,
    src: torch.Tensor,
    strategy: Strategy,
    max_len: int,
    use_cache: bool,
    cache_cross: bool,
) -> tuple[list[list[int]], int, list[float]]:
    """Translate every row of `src` in batches of `DECODE_BATCH_SIZE`.

    Returns:
        `(ids, peak_cache_bytes, ms_per_step)`: output ids per sentence, the largest
        cache any batch held, and the mean milliseconds of step `i` across batches,
        which is how per-step cost grows with output position.
    """
    outputs: list[list[int]] = []
    peak_bytes = 0
    seconds_by_step: list[list[float]] = []
    for first in range(0, src.shape[0], DECODE_BATCH_SIZE):  # noqa: NAR001
        ids, cache_bytes, step_seconds = translate(
            model=model,
            src=src[first : first + DECODE_BATCH_SIZE],
            strategy=strategy,
            max_len=max_len,
            use_cache=use_cache,
            cache_cross=cache_cross,
        )
        outputs.extend(ids)  # noqa: NAR001
        peak_bytes = max(peak_bytes, cache_bytes)  # noqa: NAR001
        for step, seconds in enumerate(step_seconds):  # noqa: NAR001
            if step == len(seconds_by_step):  # noqa: NAR001
                seconds_by_step.append([])  # noqa: NAR001
            seconds_by_step[step].append(seconds)  # noqa: NAR001
    ms_per_step = [1000 * sum(times) / len(times) for times in seconds_by_step]  # noqa: NAR001
    return outputs, peak_bytes, ms_per_step


def evaluate(
    bundle_path: Path,
    split: str,
    strategy: Strategy,
    label: str,
    use_cache: bool,
    cache_cross: bool,
    max_len: int,
    out_dir: Path,
) -> dict:
    """Decode a split from a bundle, score it with sacrebleu, and record the run.

    Args:
        label: Names the strategy in the results row and the predictions file.

    Returns:
        The results row: BLEU on detokenized text, speed, peak cache MB, device,
        git commit, and the bundle's content hash as its checkpoint id.

    Side effects:
        Writes `out_dir/predictions/<label>_<cache>_<split>.txt` and appends the row
        to `out_dir/results.csv`; may download the split.
    """
    model, src_merges, _, tgt_vocab = load_bundle(path=bundle_path, device=DEVICE)
    src_rows = read_multi30k_split(split=split, lang=SOURCE_LANG)
    references = read_multi30k_split(split=split, lang=TARGET_LANG)
    src = _source_ids(rows=src_rows, merges=src_merges, seq_len=model.src_seq_length)
    start = time.perf_counter()
    outputs, peak_bytes, ms_per_step = decode_split(
        model=model,
        src=src,
        strategy=strategy,
        max_len=max_len,
        use_cache=use_cache,
        cache_cross=cache_cross,
    )
    seconds = time.perf_counter() - start
    hypotheses = [bpe.decode(ids=ids, vocab=tgt_vocab).strip() for ids in outputs]
    cache_label = ("kv" if cache_cross else "kv_self") if use_cache else "no_cache"
    row = {
        "date": date.today().isoformat(),
        "git_commit": get_git_commit(),
        "bundle_sha256": hashlib.sha256(bundle_path.read_bytes()).hexdigest()[:12],  # noqa: NAR001
        "device": str(DEVICE),  # noqa: NAR001
        "split": split,
        "strategy": label,
        "cache": cache_label,
        "bleu": round(sacrebleu.corpus_bleu(hypotheses, [references]).score, 2),  # noqa: NAR001
        "tokens_per_s": round(sum(len(ids) + 1 for ids in outputs) / seconds, 1),  # noqa: NAR001
        "ms_per_sentence": round(1000 * seconds / len(outputs), 2),  # noqa: NAR001
        "peak_cache_mb": round(peak_bytes / 2**20, 3),  # noqa: NAR001
        "ms_per_step": " ".join(f"{ms:.2f}" for ms in ms_per_step),  # noqa: NAR001
    }
    _write_results(row=row, hypotheses=hypotheses, out_dir=out_dir)
    return row


def _write_results(row: dict, hypotheses: list[str], out_dir: Path) -> None:
    """Write the predictions file and append `row` to results.csv, header on first write."""
    predictions = out_dir / "predictions" / f"{row['strategy']}_{row['cache']}_{row['split']}.txt"
    predictions.parent.mkdir(parents=True, exist_ok=True)
    predictions.write_text(data="\n".join(hypotheses) + "\n", encoding="utf-8")  # noqa: NAR001
    results = out_dir / "results.csv"
    is_new = not results.exists()
    with open(file=results, mode="a", newline="") as results_file:
        writer = csv.DictWriter(f=results_file, fieldnames=list(row))  # noqa: NAR001
        if is_new:
            writer.writeheader()
        writer.writerow(rowdict=row)


def train():
    """Train one Multi30kTranslator from the module constants, optionally to W&B."""
    # Fresh per run, so no single lucky draw is trained on forever; fit writes it to
    # the history file, so a run can be repeated by passing its recorded seed back.
    train_seed = random.getrandbits(32)  # noqa: NAR001
    # fit reseeds from it, but only after the model exists; seed here so weight init
    # uses the recorded seed too. See the README on what the rewind implies.
    torch.manual_seed(seed=train_seed)
    SHOULD_LOG_WANDB: bool = False
    wandb_run = None
    if SHOULD_LOG_WANDB:
        wandb_run = init_wandb(project="dawnshard", run_name="multi30k_translator")
    run_uuid = str(uuid4())  # noqa: NAR001
    epoch_spec = EpochSpec(
        device=DEVICE,
        metrics_type=MaskedTokenMetrics,
        calculate_metrics=calculate_metrics,
        accumulate_metrics=accumulate_metrics,
        reduce_metrics=reduce_metrics,
        eval_probe=partial(  # noqa: NAR001
            save_attention_maps,
            out_dir=ATTENTION_PROBES_PATH / run_uuid,
            num_sentences=PROBE_SENTENCES,
        ),
    )
    train_loader, val_loader, _, _, src_vocab, _, tgt_vocab = get_multi30k_translation_dataloader(
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE,
        cache=False,
        src_lang=SOURCE_LANG,
        tgt_lang=TARGET_LANG,
        retrain_tokenizer=True,
        set_default_tokenizer=True,
    )

    model = Multi30kTranslator(
        src_vocab_size=len(src_vocab),  # noqa: NAR001
        tgt_vocab_size=len(tgt_vocab),  # noqa: NAR001
        d_model=D_MODEL_SIZE,
        src_seq_length=SEQ_LEN,
        tgt_seq_length=SEQ_LEN,
        dropout=DROPOUT_RATE,
        num_encoder_heads=NUM_HEADS,
        num_decoder_heads=NUM_HEADS,
        encoder_ffn_multiplier=FFN_MULTIPLIER,
        decoder_ffn_multiplier=FFN_MULTIPLIER,
        num_encoders=NUM_ENCODERS,
        num_decoders=NUM_DECODERS,
        qk_norm=QK_NORM,
        norm=NORM,
    ).to(device=DEVICE)  # noqa: NAR001
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    train_state = TrainState(
        model=model,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        # fit steps the scheduler once per epoch, so this decays to 0 by the last one.
        scheduler=CosineAnnealingLR(optimizer=optimizer, T_max=NUM_EPOCHS)
        if COSINE_ANNEALING
        else None,
    )
    fit(
        run_uuid=run_uuid,
        state=train_state,
        epoch_spec=epoch_spec,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS,
        val_epoch_list=list(range(1, NUM_EPOCHS + 1)),  # noqa: NAR001
        train_seed=train_seed,
        val_seed=VAL_SEED,
        test_seed=TEST_SEED,
        wandb_run=wandb_run,
    )
    plot_attention_probes(
        probe_dir=ATTENTION_PROBES_PATH / run_uuid, vocab=src_vocab, tgt_vocab=tgt_vocab
    )


def main():
    """Train once, or with `decode`, score a bundle on a split; see the module docstring."""
    parser = argparse.ArgumentParser(description="Train or decode a Multi30kTranslator.")
    parser.add_argument("command", nargs="?", default="train", choices=["train", "decode"])  # noqa: NAR001
    parser.add_argument("--bundle", type=Path, default=BUNDLE_PATH)  # noqa: NAR001
    parser.add_argument("--split", default="test", choices=["val", "test"])  # noqa: NAR001
    parser.add_argument("--strategy", default="greedy", choices=["greedy", "beam"])  # noqa: NAR001
    parser.add_argument("--beam-size", type=int, default=4)  # noqa: NAR001
    parser.add_argument("--alpha", type=float, default=0.0)  # noqa: NAR001
    parser.add_argument("--cache", default="kv", choices=["kv", "kv_self", "none"])  # noqa: NAR001
    parser.add_argument("--max-len", type=int, default=DECODE_MAX_LEN)  # noqa: NAR001
    parser.add_argument("--out-dir", type=Path, default=DECODING_RESULTS_PATH)  # noqa: NAR001
    args = parser.parse_args()
    if args.command == "train":
        train()
        return
    if args.strategy == "greedy":
        strategy: Strategy = Greedy(eos_id=EOS_ID, pad_id=PAD_ID)
        label = "greedy"
    else:
        strategy = BeamSearch(
            beam_size=args.beam_size, eos_id=EOS_ID, pad_id=PAD_ID, alpha=args.alpha
        )
        label = f"beam{args.beam_size}_alpha{args.alpha}"
    row = evaluate(
        bundle_path=args.bundle,
        split=args.split,
        strategy=strategy,
        label=label,
        use_cache=args.cache != "none",
        cache_cross=args.cache == "kv",
        max_len=args.max_len,
        out_dir=args.out_dir,
    )
    print(row)  # noqa: NAR001


if __name__ == "__main__":
    main()
