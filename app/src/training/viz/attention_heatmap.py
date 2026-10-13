"""
attention_heatmap.py
Plot per-head attention weights as a grid of heatmaps, one row per sentence.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import torch
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
from loguru import logger
from training.common.constants import PAD_ID
from training.transformer.constants import B, H, L, S

MAX_TOKENS = 24
# For an encoder-decoder capture, each kind's (query side, key side); side 0 is the
# source ids and side 1 the target input ids, in the order the input stacks them.
ENCODER_DECODER_SIDES = {"encoder": (0, 0), "decoder_self": (1, 1), "decoder_cross": (1, 0)}


def _token_labels(ids: list[int], vocab: dict[int, str]) -> list[str]:
    """Tick labels for `ids`, with spaces shown as '·' so word boundaries stay visible."""
    return [vocab[i].replace(" ", "·") for i in ids]  # noqa: NAR001


@jaxtyped(typechecker=beartype)
def plot_attention_heads(
    weights: Float[torch.Tensor, f"{B} {H} {L} {S}"],
    token_ids: Int[torch.Tensor, f"{B} {L}"],
    vocab: dict[int, str],
    title: str,
    path: Path,
    max_tokens: int = MAX_TOKENS,
    key_ids: Optional[Int[torch.Tensor, f"{B} {S}"]] = None,
    key_vocab: Optional[dict[int, str]] = None,
) -> None:
    """Save a grid of attention heatmaps: rows are sentences, columns are heads.

    Within a cell, rows are query tokens and columns are the key tokens they attend
    to. Every cell shares one colour scale so heads can be compared. Pads are
    dropped and each sentence is cropped to its first `max_tokens` tokens to keep
    the labels legible, so a cropped row no longer sums to 1.

    Args:
        weights: Detached attention weights from one layer, one batch row per
            sentence, as `save_attention_maps` stores them.
        token_ids: The query ids those weights came from, right-padded with PAD_ID.
        vocab: Maps query ids to token strings, as returned by bpe.train.
        key_ids: The key ids, when they differ from the queries (cross-attention).
            Defaults to `token_ids`, the self-attention case.
        key_vocab: Maps key ids to token strings, when keys come from another
            tokenizer. Defaults to `vocab`.
        title: Figure title, e.g. which layer the weights came from.
        path: Where to write the PNG; parent directories are created.
        max_tokens: Keep at most this many leading tokens per sentence.
    """
    key_ids = token_ids if key_ids is None else key_ids
    key_vocab = vocab if key_vocab is None else key_vocab
    weights = weights.float().cpu()
    query_lengths = (token_ids != PAD_ID).sum(dim=1).clamp(max=max_tokens).tolist()
    key_lengths = (key_ids != PAD_ID).sum(dim=1).clamp(max=max_tokens).tolist()
    lengths = list(zip(query_lengths, key_lengths))  # noqa: NAR001
    n_sentences, n_heads = weights.shape[0], weights.shape[1]
    vmax = max(  # noqa: NAR001
        weights[s, :, :n_q, :n_k].max().item()
        for s, (n_q, n_k) in enumerate(lengths)  # noqa: NAR001
    )
    fig, axes = plt.subplots(
        nrows=n_sentences,
        ncols=n_heads,
        figsize=(2.4 * n_heads, 2.8 * n_sentences),
        squeeze=False,
        layout="constrained",
    )
    for s, (n_q, n_k) in enumerate(lengths):  # noqa: NAR001
        query_labels = _token_labels(ids=token_ids[s, :n_q].tolist(), vocab=vocab)
        key_labels = _token_labels(ids=key_ids[s, :n_k].tolist(), vocab=key_vocab)
        for head in range(n_heads):  # noqa: NAR001
            ax = axes[s][head]
            # aspect="auto" fills the axes box that constrained layout reserved, so
            # tick labels can't spill into the row below.
            ax.imshow(
                X=weights[s, head, :n_q, :n_k].numpy(),
                cmap="Blues",
                vmin=0.0,
                vmax=vmax,
                aspect="auto",
            )
            ax.set_xticks(ticks=range(n_k), labels=key_labels, rotation=90, fontsize=5)  # noqa: NAR001
            ax.set_yticks(ticks=range(n_q), labels=query_labels, fontsize=5)  # noqa: NAR001
            ax.tick_params(labelleft=head == 0)
            ax.set_title(label=f"sentence {s}, head {head}", fontsize=7)
    fig.colorbar(mappable=axes[0][0].images[0], ax=axes, shrink=0.6, label="attention weight")
    fig.suptitle(t=title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fname=path, dpi=150)
    plt.close(fig=fig)
    logger.info("Saved plot to {}", path)  # noqa: NAR001


def plot_attention_probes(
    probe_dir: Path, vocab: dict[int, str], tgt_vocab: Optional[dict[int, str]] = None
) -> None:
    """Plot every capture `save_attention_maps` wrote to `probe_dir`.

    Each `epoch_<NNN>.pt` file holds `{"token_ids": ..., "weights": {kind:
    [(n, h, L, S) per block]}}`. Writes one `epoch_<NNN>_<kind>_block_<i>.png` per
    kind and block beside it.

    Args:
        probe_dir: Directory the probe wrote to.
        vocab: Maps ids to token strings, as returned by bpe.train. For an
            encoder-decoder capture, the source vocab.
        tgt_vocab: The target vocab of an encoder-decoder capture. When set,
            `token_ids` must be `(n, 2, L)` with the source ids at index 0 and the
            target input ids at index 1, and each kind's axes are labelled from the
            sides in `ENCODER_DECODER_SIDES`. When None, `token_ids` is `(n, L)` and
            labels both axes of every kind.
    """
    for capture_path in sorted(probe_dir.glob(pattern="*.pt")):  # noqa: NAR001
        capture = torch.load(f=capture_path)
        token_ids = capture["token_ids"]
        for kind, blocks in capture["weights"].items():
            if tgt_vocab is None:
                query_ids, query_vocab, key_ids, key_vocab = token_ids, vocab, None, None
            else:
                query_side, key_side = ENCODER_DECODER_SIDES[kind]
                side_vocabs = (vocab, tgt_vocab)
                query_ids, query_vocab = token_ids[:, query_side], side_vocabs[query_side]
                key_ids, key_vocab = token_ids[:, key_side], side_vocabs[key_side]
            for block_index, weights in enumerate(blocks):  # noqa: NAR001
                plot_attention_heads(
                    weights=weights,
                    token_ids=query_ids,
                    vocab=query_vocab,
                    title=f"{capture_path.stem}, {kind} block {block_index}",
                    path=probe_dir / f"{capture_path.stem}_{kind}_block_{block_index}.png",
                    key_ids=key_ids,
                    key_vocab=key_vocab,
                )
