import torch
from training.transformer.mask import padding_keep_mask
from training.transformer.modules import MultiHeadAttentionLayer


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
