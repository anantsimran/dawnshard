"""
attention_heatmap.py
Plot a run's attention captures: what each head does and when it started doing it, and
one sentence's weights in every head.
"""

from pathlib import Path
from typing import NamedTuple, Optional

import matplotlib.pyplot as plt
import torch
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
from loguru import logger
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from training.preprocess.constants import PAD_ID
from training.transformer.constants import B, H, L, S
from training.transformer.probes import AttentionCapture, load_attention_captures

MAX_TOKENS = 24
# For an encoder-decoder capture, each kind's (query side, key side); side 0 is the
# source ids and side 1 the target input ids, in the order the input stacks them.
ENCODER_DECODER_SIDES = {"encoder": (0, 0), "decoder_self": (1, 1), "decoder_cross": (1, 0)}
# Each statistic `head_stats` computes, mapped to the panel title that says how to read it
# and the least its colour scale reaches. Shares get 0.5, so a share near 0 in every head
# is drawn pale instead of stretched across the whole scale.
HEAD_STATS = {
    "entropy": ("entropy (0 = one key, 1 = spread evenly)", 1.0),
    "top_key": ("share on the most-attended key", 0.5),
    "distance": ("mean distance |i - j| in tokens", 0.0),
    "previous": ("share on the previous token", 0.5),
}
# These compare query and key positions, so they need queries and keys from one sequence.
SELF_ATTENTION_STATS = ("distance", "previous")


class _Sides(NamedTuple):
    """The ids and vocab on each axis of one capture's attention."""

    query_ids: Int[torch.Tensor, f"{B} {L}"]
    query_vocab: dict[int, str]
    key_ids: Int[torch.Tensor, f"{B} {S}"]
    key_vocab: dict[int, str]
    self_attention: bool


def _sides(
    capture: AttentionCapture, vocab: dict[int, str], tgt_vocab: Optional[dict[int, str]]
) -> _Sides:
    """Pick `capture`'s query and key ids and vocabs, as `plot_attention_probes` describes."""
    if tgt_vocab is None:
        return _Sides(
            query_ids=capture.token_ids,
            query_vocab=vocab,
            key_ids=capture.token_ids,
            key_vocab=vocab,
            self_attention=True,
        )
    query_side, key_side = ENCODER_DECODER_SIDES[capture.kind]
    side_vocabs = (vocab, tgt_vocab)
    return _Sides(
        query_ids=capture.token_ids[:, query_side],
        query_vocab=side_vocabs[query_side],
        key_ids=capture.token_ids[:, key_side],
        key_vocab=side_vocabs[key_side],
        self_attention=query_side == key_side,
    )


def _token_labels(ids: list[int], vocab: dict[int, str]) -> list[str]:
    """Tick labels for `ids`, with spaces shown as '·' so word boundaries stay visible."""
    return [vocab[i].replace(" ", "·") for i in ids]  # noqa: NAR001


def _draw_heatmap(
    ax: Axes,
    matrix: Float[torch.Tensor, f"{L} {S}"],
    query_labels: list[str],
    key_labels: list[str],
    vmax: float,
    show_query_labels: bool,
) -> AxesImage:
    """Draw `matrix` on `ax`, queries as rows and keys as columns, labelled with tokens."""
    # aspect="auto" fills the axes box that constrained layout reserved, so
    # tick labels can't spill into the row below.
    image = ax.imshow(X=matrix.numpy(), cmap="Blues", vmin=0.0, vmax=vmax, aspect="auto")
    ax.set_xticks(
        ticks=range(len(key_labels)),  # noqa: NAR001
        labels=key_labels,
        rotation=90,
        fontsize=5,
    )
    ax.set_yticks(ticks=range(len(query_labels)), labels=query_labels, fontsize=5)  # noqa: NAR001
    ax.tick_params(labelleft=show_query_labels)
    return image


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
        weights: Detached attention weights from one block, one batch row per
            sentence, as an `AttentionCapture` holds them.
        token_ids: The query ids those weights came from, right-padded with PAD_ID.
        vocab: Maps query ids to token strings, as `Tokenizer.vocab` does.
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
            _draw_heatmap(
                ax=ax,
                matrix=weights[s, head, :n_q, :n_k],
                query_labels=query_labels,
                key_labels=key_labels,
                vmax=vmax,
                show_query_labels=head == 0,
            )
            ax.set_title(label=f"sentence {s}, head {head}", fontsize=7)
    fig.colorbar(mappable=axes[0][0].images[0], ax=axes, shrink=0.6, label="attention weight")
    fig.suptitle(t=title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fname=path, dpi=150)
    plt.close(fig=fig)
    logger.info("Saved plot to {}", path)  # noqa: NAR001


@jaxtyped(typechecker=beartype)
def head_stats(
    weights: Float[torch.Tensor, f"{B} {H} {L} {S}"],
    query_ids: Int[torch.Tensor, f"{B} {L}"],
    key_ids: Int[torch.Tensor, f"{B} {S}"],
    self_attention: bool,
) -> dict[str, Float[torch.Tensor, H]]:
    """Reduce one block's weights to a few numbers per head that say what the head does.

    Pad queries and pad keys are dropped. A key counts as visible to a query when any
    head gives it nonzero weight: a masked key's weight is exactly 0 and masks are
    shared across heads, so this covers padding and causal masks alike.

    - `entropy`: a query's entropy over its visible keys divided by the log of their
      count, so 0 means it attends to one key and 1 means it spreads evenly. Queries
      that see one key have no choice to make and are skipped.
    - `top_key`: per sentence, the share of all queries' attention that goes to the
      single most-attended key. Near 1 for a sink every query points at.
    - `distance`: how far, in tokens, a query looks: the attention-weighted mean of
      |i - j|.
    - `previous`: the weight a query puts on the token just before it, over the
      queries that have one.

    `top_key` is averaged over sentences, and the others over every real query in
    every sentence.

    Args:
        weights: One block's weights, as an `AttentionCapture` holds them.
        query_ids: Right-padded with PAD_ID.
        key_ids: Right-padded with PAD_ID; the same ids as `query_ids` for
            self-attention.
        self_attention: Whether queries and keys are one sequence. Only then are
            `distance` and `previous` computed, since they compare positions.

    Returns:
        Each statistic's name, a key of `HEAD_STATS`, mapped to its `(h,)` values.
    """
    names = [n for n in HEAD_STATS if self_attention or n not in SELF_ATTENTION_STATS]
    n_heads = weights.shape[1]
    totals = {name: torch.zeros(size=(n_heads,)) for name in names}
    counts = {name: 0 for name in names}
    for sentence in range(weights.shape[0]):  # noqa: NAR001
        n_q = int((query_ids[sentence] != PAD_ID).sum())  # noqa: NAR001
        n_k = int((key_ids[sentence] != PAD_ID).sum())  # noqa: NAR001
        attn = weights[sentence, :, :n_q, :n_k].float().cpu()
        visible = (attn > 0).any(dim=0).sum(dim=-1)
        has_choice = visible > 1
        entropy = -torch.special.xlogy(input=attn, other=attn).sum(dim=-1) / visible.log()
        totals["entropy"] += entropy[:, has_choice].sum(dim=-1)
        counts["entropy"] += int(has_choice.sum())  # noqa: NAR001
        totals["top_key"] += attn.sum(dim=1).max(dim=-1).values / n_q
        counts["top_key"] += 1
        if self_attention:
            gap = torch.arange(end=n_q)[:, None] - torch.arange(end=n_k)[None, :]
            totals["distance"] += (attn * gap.abs()).sum(dim=(1, 2))
            counts["distance"] += n_q
            totals["previous"] += attn.diagonal(offset=-1, dim1=1, dim2=2).sum(dim=-1)
            counts["previous"] += n_q - 1
    return {name: totals[name] / counts[name] for name in names}


def _plot_head_summary(
    stats: dict[str, torch.Tensor],
    epochs: list[int],
    title: str,
    path: Path,
) -> None:
    """Save one panel per statistic, two to a row, with a row per head and a column per epoch.

    Rows are grouped by block, top to bottom, with a line between blocks. Each panel
    runs its colour scale from 0 to its own maximum, or to the statistic's least top in
    `HEAD_STATS` if that is higher, so a small share still shows which heads stand out
    without noise near 0 filling the scale. The colour bar gives the values.

    Args:
        stats: Each statistic mapped to its `(epochs, blocks, h)` values.
        epochs: The epoch each first-axis entry of `stats` came from.
    """
    n_epochs, n_blocks, n_heads = next(iter(stats.values())).shape  # noqa: NAR001
    n_rows = n_blocks * n_heads
    row_labels = [
        f"block {b} head {h}"
        for b in range(n_blocks)  # noqa: NAR001
        for h in range(n_heads)  # noqa: NAR001
    ]
    n_cols = min(2, len(stats))  # noqa: NAR001
    n_panel_rows = -(-len(stats) // n_cols)  # noqa: NAR001
    fig, axes = plt.subplots(
        nrows=n_panel_rows,
        ncols=n_cols,
        figsize=(n_cols * (2.5 + 0.12 * n_epochs), n_panel_rows * (1.2 + 0.22 * n_rows)),
        squeeze=False,
        layout="constrained",
    )
    for index, (ax, (name, values)) in enumerate(zip(axes.flat, stats.items())):  # noqa: NAR001
        panel_title, least_top = HEAD_STATS[name]
        image = ax.imshow(
            X=values.reshape(shape=(n_epochs, n_rows)).T.numpy(),
            cmap="Blues",
            vmin=0.0,
            vmax=max(least_top, values.max().item()),  # noqa: NAR001
            aspect="auto",
        )
        ax.set_xticks(ticks=range(n_epochs), labels=epochs, rotation=90, fontsize=6)  # noqa: NAR001
        ax.set_yticks(ticks=range(n_rows), labels=row_labels, fontsize=6)  # noqa: NAR001
        ax.tick_params(labelleft=index % n_cols == 0)
        for block in range(1, n_blocks):  # noqa: NAR001
            ax.axhline(y=block * n_heads - 0.5, color="black", linewidth=0.8)
        ax.set_xlabel(xlabel="epoch")
        ax.set_title(label=panel_title, fontsize=8)
        fig.colorbar(mappable=image, ax=ax, shrink=0.8)
    fig.suptitle(t=title)
    fig.savefig(fname=path, dpi=150)
    plt.close(fig=fig)
    logger.info("Saved plot to {}", path)  # noqa: NAR001


def _plot_sentence(
    captures: list[AttentionCapture],
    sides: list[_Sides],
    stats: dict[str, torch.Tensor],
    title: str,
    path: Path,
    max_tokens: int = MAX_TOKENS,
) -> None:
    """Save the first saved sentence's weights, with a row per block and a column per head.

    Each panel runs its colour scale from 0 to its own maximum, so a head that spreads
    its attention stays visible beside one that points at a single key. Compare heads
    by their titles instead, which carry `head_stats` over every saved sentence. The
    sentence is cropped to its first `max_tokens` tokens, so a cropped row no longer
    sums to 1.

    Args:
        captures: One epoch's captures of one kind, in block order.
        sides: Each capture's `_sides`, in the same order.
        stats: Each statistic mapped to its `(blocks, h)` values.
    """
    n_heads = captures[0].weights.shape[1]
    fig, axes = plt.subplots(
        nrows=len(captures),  # noqa: NAR001
        ncols=n_heads,
        figsize=(3.0 * n_heads, 3.2 * len(captures)),  # noqa: NAR001
        squeeze=False,
        layout="constrained",
    )
    for block, (capture, side) in enumerate(zip(captures, sides)):  # noqa: NAR001
        n_q = min(int((side.query_ids[0] != PAD_ID).sum()), max_tokens)  # noqa: NAR001
        n_k = min(int((side.key_ids[0] != PAD_ID).sum()), max_tokens)  # noqa: NAR001
        query_labels = _token_labels(ids=side.query_ids[0, :n_q].tolist(), vocab=side.query_vocab)
        key_labels = _token_labels(ids=side.key_ids[0, :n_k].tolist(), vocab=side.key_vocab)
        for head in range(n_heads):  # noqa: NAR001
            matrix = capture.weights[0, head, :n_q, :n_k].float().cpu()
            ax = axes[block][head]
            _draw_heatmap(
                ax=ax,
                matrix=matrix,
                query_labels=query_labels,
                key_labels=key_labels,
                vmax=matrix.max().item(),
                show_query_labels=head == 0,
            )
            numbers = ", ".join(  # noqa: NAR001
                f"{name} {values[block, head]:.2f}" for name, values in stats.items()
            )
            ax.set_title(label=f"block {block} head {head}\n{numbers}", fontsize=6)
    fig.suptitle(t=title)
    fig.savefig(fname=path, dpi=150)
    plt.close(fig=fig)
    logger.info("Saved plot to {}", path)  # noqa: NAR001


def plot_attention_probes(
    probe_dir: Path, vocab: dict[int, str], tgt_vocab: Optional[dict[int, str]] = None
) -> None:
    """Plot the captures `save_attention_maps` wrote to `probe_dir`, two figures per kind.

    For each kind of attention, writes into `probe_dir`:

    - `<kind>_heads.png`: every head's `head_stats` at every saved epoch, which says
      what each head does and when it started doing it.
    - `<kind>_sentence_0.png`: the first saved sentence's weights in every block and
      head at the last epoch, titled with those heads' statistics, to check the
      numbers against real tokens.

    Every epoch must hold a capture for every block, as `save_attention_maps` writes.

    Args:
        probe_dir: Directory the probe wrote to.
        vocab: Maps ids to token strings, as `Tokenizer.vocab` does. For an
            encoder-decoder capture, the source vocab.
        tgt_vocab: The target vocab of an encoder-decoder capture. When set,
            `token_ids` must be `(n, 2, L)` with the source ids at index 0 and the
            target input ids at index 1, and each kind's axes are labelled from the
            sides in `ENCODER_DECODER_SIDES`. When None, `token_ids` is `(n, L)` and
            labels both axes of every kind.
    """
    captures = load_attention_captures(probe_dir=probe_dir)
    for kind in sorted({c.kind for c in captures}):  # noqa: NAR001
        of_kind = [c for c in captures if c.kind == kind]
        sides = [_sides(capture=c, vocab=vocab, tgt_vocab=tgt_vocab) for c in of_kind]
        epochs = sorted({c.epoch for c in of_kind})  # noqa: NAR001
        n_blocks = len(of_kind) // len(epochs)  # noqa: NAR001
        per_capture = [
            head_stats(
                weights=c.weights,
                query_ids=side.query_ids,
                key_ids=side.key_ids,
                self_attention=side.self_attention,
            )
            for c, side in zip(of_kind, sides)  # noqa: NAR001
        ]
        # Captures come sorted by epoch, then block, so stacking them is epoch-major.
        stats = {
            name: torch.stack(tensors=[s[name] for s in per_capture]).reshape(
                shape=(len(epochs), n_blocks, -1)  # noqa: NAR001
            )
            for name in per_capture[0]
        }
        _plot_head_summary(
            stats=stats,
            epochs=epochs,
            title=f"{kind} attention: every head at every saved epoch",
            path=probe_dir / f"{kind}_heads.png",
        )
        _plot_sentence(
            captures=of_kind[-n_blocks:],
            sides=sides[-n_blocks:],
            stats={name: values[-1] for name, values in stats.items()},
            title=f"{kind} attention, epoch {epochs[-1]}, sentence 0 "
            "(each panel scaled to its own max)",
            path=probe_dir / f"{kind}_sentence_0.png",
        )
