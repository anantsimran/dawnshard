import math

import pytest
import torch
from training.transformer.mask import padding_keep_mask
from training.transformer.modules import MultiHeadAttentionLayer, SinusoidalEmbedding


def test_output_preserves_d_model():
    torch.manual_seed(seed=0)
    layer = MultiHeadAttentionLayer(d_model=64, h=8, qk_norm=True)
    x = torch.randn(2, 10, 64)  # noqa: NAR001
    assert layer(batch=x, pad_mask=None)[0].shape == (2, 10, 64)


def test_qk_norm_makes_attention_scale_invariant():
    torch.manual_seed(seed=0)
    layer = MultiHeadAttentionLayer(d_model=64, h=8, qk_norm=True)
    x = torch.randn(2, 10, 64)  # noqa: NAR001
    torch.testing.assert_close(
        actual=layer(batch=10.0 * x, pad_mask=None)[0],
        expected=10.0 * layer(batch=x, pad_mask=None)[0],
    )


def test_without_qk_norm_attention_is_not_scale_invariant():
    torch.manual_seed(seed=0)
    layer = MultiHeadAttentionLayer(d_model=64, h=8, qk_norm=False)
    x = torch.randn(2, 10, 64)  # noqa: NAR001
    assert not torch.allclose(
        input=layer(batch=10.0 * x, pad_mask=None)[0],
        other=10.0 * layer(batch=x, pad_mask=None)[0],
    )


def test_pad_mask_hides_padded_positions_from_real_tokens():
    torch.manual_seed(seed=0)
    layer = MultiHeadAttentionLayer(d_model=64, h=8, qk_norm=True)
    token_ids = torch.tensor([[5, 7, 9, 0]])  # noqa: NAR001
    mask = padding_keep_mask(token_ids=token_ids, pad_id=0)
    x = torch.randn(1, 4, 64)  # noqa: NAR001
    x_other_pad = x.clone()
    x_other_pad[:, 3] = torch.randn(64)  # noqa: NAR001
    torch.testing.assert_close(
        actual=layer(batch=x_other_pad, pad_mask=mask)[0][:, :3],
        expected=layer(batch=x, pad_mask=mask)[0][:, :3],
    )


def test_returns_attention_weights_per_head():
    torch.manual_seed(seed=0)
    layer = MultiHeadAttentionLayer(d_model=64, h=8, qk_norm=True)
    x = torch.randn(2, 10, 64)  # noqa: NAR001
    _, weights = layer(batch=x, pad_mask=None)
    assert weights.shape == (2, 8, 10, 10)
    torch.testing.assert_close(actual=weights.sum(dim=-1), expected=torch.ones(2, 8, 10))  # noqa: NAR001


def test_position_table_matches_the_closed_form():
    layer = SinusoidalEmbedding(seq_length=16, d_model=8)
    assert layer.pe.shape == (16, 8)
    angle = 3 * 10_000 ** (-2 / 8)
    torch.testing.assert_close(
        actual=layer.pe[3, 2],
        expected=torch.tensor(math.sin(angle)),  # noqa: NAR001
    )
    torch.testing.assert_close(
        actual=layer.pe[3, 3],
        expected=torch.tensor(math.cos(angle)),  # noqa: NAR001
    )


def test_first_position_is_sin_zero_and_cos_zero():
    layer = SinusoidalEmbedding(seq_length=16, d_model=8)
    torch.testing.assert_close(
        actual=layer.pe[0, 0::2],
        expected=torch.zeros(4),  # noqa: NAR001
    )
    torch.testing.assert_close(
        actual=layer.pe[0, 1::2],
        expected=torch.ones(4),  # noqa: NAR001
    )


def test_each_feature_pair_lies_on_the_unit_circle():
    layer = SinusoidalEmbedding(seq_length=16, d_model=8)
    radius = layer.pe[:, 0::2] ** 2 + layer.pe[:, 1::2] ** 2
    torch.testing.assert_close(
        actual=radius,
        expected=torch.ones(16, 4),  # noqa: NAR001
    )


def test_every_position_gets_a_different_encoding():
    layer = SinusoidalEmbedding(seq_length=16, d_model=8)
    distances = torch.cdist(x1=layer.pe, x2=layer.pe)
    assert (distances + torch.eye(16) > 1e-3).all()  # noqa: NAR001


def test_forward_adds_the_table_to_every_row_of_the_batch():
    layer = SinusoidalEmbedding(seq_length=16, d_model=8)
    x = torch.randn(3, 10, 8)  # noqa: NAR001
    torch.testing.assert_close(
        actual=layer(x) - x,  # noqa: NAR001
        expected=layer.pe[:10].expand(3, 10, 8),  # noqa: NAR001
    )


def test_forward_rejects_sequences_longer_than_the_table():
    layer = SinusoidalEmbedding(seq_length=4, d_model=8)
    with pytest.raises(ValueError):
        layer(torch.zeros(1, 5, 8))  # noqa: NAR001


def test_table_is_not_a_parameter_and_stays_out_of_the_state_dict():
    layer = SinusoidalEmbedding(seq_length=16, d_model=8)
    assert list(layer.parameters()) == []  # noqa: NAR001
    assert "pe" not in layer.state_dict()
