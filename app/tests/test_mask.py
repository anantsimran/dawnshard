import torch
import torch.nn.functional as F
from training.transformer.functions import attention
from training.transformer.mask import causal_keep_mask, pad_masked_mean, padding_keep_mask

PAD = 0


def test_padding_keep_mask_values_and_shape():
    token_ids = torch.tensor([[5, 7, 9, 0], [3, 4, 0, 0]])  # noqa: NAR001
    mask = padding_keep_mask(token_ids=token_ids, pad_id=PAD)
    expected = torch.tensor(  # noqa: NAR001
        [[True, True, True, False], [True, True, False, False]]
    )[:, None, None, :]
    assert mask.shape == (2, 1, 1, 4)
    assert torch.equal(mask, expected)  # noqa: NAR001


def test_causal_keep_mask_is_lower_triangular():
    mask = causal_keep_mask(seq_len=3)
    expected = torch.tensor(  # noqa: NAR001
        [[True, False, False], [True, True, False], [True, True, True]]
    )[None, None]
    assert mask.shape == (1, 1, 3, 3)
    assert torch.equal(mask, expected)  # noqa: NAR001


def test_causal_mask_matches_torch_reference():
    torch.manual_seed(seed=0)
    q = torch.randn(2, 4, 8, 16)  # noqa: NAR001
    k = torch.randn(2, 4, 8, 16)  # noqa: NAR001
    v = torch.randn(2, 4, 8, 32)  # noqa: NAR001
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # noqa: NAR001
    actual, _ = attention(q=q, k=k, v=v, mask=causal_keep_mask(seq_len=8))
    torch.testing.assert_close(actual=actual, expected=expected)


def test_padded_keys_get_no_weight():
    torch.manual_seed(seed=0)
    token_ids = torch.tensor([[5, 7, 9, 0], [3, 4, 0, 0]])  # noqa: NAR001
    q = torch.randn(2, 2, 4, 8)  # noqa: NAR001
    k = torch.randn(2, 2, 4, 8)  # noqa: NAR001
    v = torch.eye(4).expand(2, 2, 4, 4)  # noqa: NAR001
    mask = padding_keep_mask(token_ids=token_ids, pad_id=PAD)
    _, weights = attention(q=q, k=k, v=v, mask=mask)
    assert torch.all(weights[0, ..., 3] == 0)  # noqa: NAR001
    assert torch.all(weights[1, ..., 2:] == 0)  # noqa: NAR001
    assert not weights.isnan().any()


def test_combined_mask_broadcasts_and_hides_future_and_pad():
    token_ids = torch.tensor([[5, 7, 9, 0]])  # noqa: NAR001
    mask = padding_keep_mask(token_ids=token_ids, pad_id=PAD) & causal_keep_mask(seq_len=4)
    assert mask.shape == (1, 1, 4, 4)
    torch.manual_seed(seed=0)
    q = torch.randn(1, 2, 4, 8)  # noqa: NAR001
    k = torch.randn(1, 2, 4, 8)  # noqa: NAR001
    v = torch.eye(4).expand(1, 2, 4, 4)  # noqa: NAR001
    _, weights = attention(q=q, k=k, v=v, mask=mask)
    assert torch.all(weights.triu(diagonal=1) == 0)  # noqa: NAR001
    assert torch.all(weights[..., 3] == 0)  # noqa: NAR001


def test_pad_masked_mean_averages_only_real_positions():
    attn = torch.tensor(  # noqa: NAR001
        [[[1.0, 2.0], [3.0, 4.0], [100.0, -100.0]]]
    )
    mask = torch.tensor([[True, True, False]])  # noqa: NAR001
    torch.testing.assert_close(
        actual=pad_masked_mean(attn=attn, mask=mask),
        expected=torch.tensor([[2.0, 3.0]]),  # noqa: NAR001
    )


def test_pad_masked_mean_all_padding_row_is_zeros_not_nan():
    torch.manual_seed(seed=0)
    attn = torch.randn(2, 3, 4)  # noqa: NAR001
    mask = torch.tensor([[True, False, False], [False, False, False]])  # noqa: NAR001
    pooled = pad_masked_mean(attn=attn, mask=mask)
    assert not pooled.isnan().any()
    torch.testing.assert_close(actual=pooled[0], expected=attn[0, 0])
    torch.testing.assert_close(actual=pooled[1], expected=torch.zeros(4))  # noqa: NAR001
