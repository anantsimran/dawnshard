import pytest
import torch
from torch import nn
from training.preprocess.constants import PAD_ID
from training.transformer.probes import AttentionProbePerEpoch, save_attention_maps
from training.viz.attention_heatmap import head_stats, plot_attention_probes

VOCAB = {i: f"t{i}" for i in range(PAD_ID + 1)}  # noqa: NAR001


def _uniform(length):
    return torch.full(size=(length, length), fill_value=1 / length)


def _previous_token(length):
    # Row 0 has no previous token, so it attends to itself.
    weights = torch.zeros(size=(length, length))
    weights[0, 0] = 1.0
    for i in range(1, length):  # noqa: NAR001
        weights[i, i - 1] = 1.0
    return weights


def test_uniform_head_and_previous_token_head():
    # Head 0 keeps every key visible, so head 1's exact zeros aren't read as masked.
    weights = torch.stack(tensors=[_uniform(length=4), _previous_token(length=4)])[None]
    ids = torch.tensor(data=[[1, 2, 3, 4]])

    stats = head_stats(weights=weights, query_ids=ids, key_ids=ids, self_attention=True)

    assert torch.allclose(input=stats["entropy"], other=torch.tensor(data=[1.0, 0.0]))
    assert torch.allclose(input=stats["top_key"], other=torch.tensor(data=[0.25, 0.5]))
    # Uniform: row mean |i - j| is 1.5, 1, 1, 1.5. Previous: 0 on row 0, then 1.
    assert torch.allclose(input=stats["distance"], other=torch.tensor(data=[1.25, 0.75]))
    assert torch.allclose(input=stats["previous"], other=torch.tensor(data=[0.25, 1.0]))


def test_causal_uniform_head_has_full_entropy():
    causal = torch.tril(input=torch.ones(size=(4, 4)))
    weights = (causal / causal.sum(dim=-1, keepdim=True))[None, None]
    ids = torch.tensor(data=[[1, 2, 3, 4]])

    stats = head_stats(weights=weights, query_ids=ids, key_ids=ids, self_attention=True)

    # Row 0 sees one key and is skipped; every other row spreads evenly over what it sees.
    assert stats["entropy"].item() == pytest.approx(expected=1.0)
    assert stats["previous"].item() == pytest.approx(expected=(1 / 2 + 1 / 3 + 1 / 4) / 3)


def test_pad_queries_and_keys_are_ignored():
    torch.manual_seed(seed=0)
    real = torch.softmax(input=torch.randn(size=(1, 2, 3, 3)), dim=-1)
    padded = torch.zeros(size=(1, 2, 5, 5))
    padded[..., :3, :3] = real
    # Pad query rows still attend to the real keys; their weights must not count.
    padded[..., 3:, :3] = 1 / 3
    ids = torch.tensor(data=[[1, 2, 3]])
    padded_ids = torch.tensor(data=[[1, 2, 3, PAD_ID, PAD_ID]])

    expected = head_stats(weights=real, query_ids=ids, key_ids=ids, self_attention=True)
    stats = head_stats(
        weights=padded, query_ids=padded_ids, key_ids=padded_ids, self_attention=True
    )

    for name, values in expected.items():
        assert torch.allclose(input=stats[name], other=values), name


def test_cross_attention_skips_position_statistics():
    weights = torch.full(size=(1, 1, 2, 3), fill_value=1 / 3)

    stats = head_stats(
        weights=weights,
        query_ids=torch.tensor(data=[[1, 2]]),
        key_ids=torch.tensor(data=[[1, 2, 3]]),
        self_attention=False,
    )

    assert set(stats) == {"entropy", "top_key"}  # noqa: NAR001


class _ModelWithMaps(nn.Module):
    def __init__(self, attention_maps):
        super().__init__()
        self.attention_maps = attention_maps


def _attention(query_ids, key_ids, causal=False):
    scores = torch.randn(size=(query_ids.shape[0], 2, query_ids.shape[1], key_ids.shape[1]))
    keep = (key_ids != PAD_ID)[:, None, None, :]
    if causal:
        keep = keep & torch.tril(input=torch.ones(size=scores.shape[-2:], dtype=torch.bool))
    return torch.softmax(input=scores.masked_fill(mask=~keep, value=-torch.inf), dim=-1)


def _save_two_epochs(out_dir, input_tensor, maps_for_epoch):
    for epoch in (1, 2):
        save_attention_maps(
            model=_ModelWithMaps(attention_maps=AttentionProbePerEpoch(map=maps_for_epoch())),
            input_tensor=input_tensor,
            target_tensor=input_tensor,
            predicted=input_tensor,
            epoch=epoch,
            batch_index=0,
            seed=None,
            out_dir=out_dir,
            num_sentences=2,
        )


def _pngs(directory):
    return sorted(str(p.relative_to(directory)) for p in directory.rglob(pattern="*.png"))  # noqa: NAR001


def test_encoder_probe_writes_a_head_summary_and_a_sentence_figure(tmp_path):
    torch.manual_seed(seed=0)
    ids = torch.tensor(data=[[1, 2, 3, PAD_ID], [4, 5, 6, 7], [8, 9, PAD_ID, PAD_ID]])
    _save_two_epochs(
        out_dir=tmp_path,
        input_tensor=ids,
        maps_for_epoch=lambda: {"encoder": [_attention(query_ids=ids, key_ids=ids)] * 2},
    )

    plot_attention_probes(probe_dir=tmp_path, vocab=VOCAB)

    assert _pngs(directory=tmp_path) == ["encoder_heads.png", "encoder_sentence_0.png"]


def test_encoder_decoder_probe_writes_two_figures_per_kind(tmp_path):
    torch.manual_seed(seed=0)
    src = torch.tensor(data=[[1, 2, 3, PAD_ID], [4, 5, 6, 7]])
    tgt = torch.tensor(data=[[9, 8, PAD_ID, PAD_ID], [7, 6, 5, PAD_ID]])
    _save_two_epochs(
        out_dir=tmp_path,
        input_tensor=torch.stack(tensors=[src, tgt], dim=1),
        maps_for_epoch=lambda: {
            "encoder": [_attention(query_ids=src, key_ids=src)],
            "decoder_self": [_attention(query_ids=tgt, key_ids=tgt, causal=True)],
            "decoder_cross": [_attention(query_ids=tgt, key_ids=src)],
        },
    )

    plot_attention_probes(probe_dir=tmp_path, vocab=VOCAB, tgt_vocab=VOCAB)

    assert _pngs(directory=tmp_path) == [
        f"{kind}_{figure}.png"
        for kind in ("decoder_cross", "decoder_self", "encoder")
        for figure in ("heads", "sentence_0")
    ]
