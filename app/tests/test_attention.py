import torch
import torch.nn.functional as F

from training.transformer.functions import attention


def test_matches_torch_reference():
    torch.manual_seed(seed=0)
    q = torch.randn(2, 4, 8, 16)  # noqa: NAR001
    k = torch.randn(2, 4, 8, 16)  # noqa: NAR001
    v = torch.randn(2, 4, 8, 32)  # noqa: NAR001
    expected = F.scaled_dot_product_attention(q, k, v)  # noqa: NAR001
    actual, _ = attention(q=q, k=k, v=v, mask=None)
    torch.testing.assert_close(actual=actual, expected=expected)


def test_attention_weights_sum_to_one():
    torch.manual_seed(seed=0)
    q = torch.randn(2, 4, 8, 16)  # noqa: NAR001
    k = torch.randn(2, 4, 8, 16)  # noqa: NAR001
    v = torch.eye(8).expand(2, 4, 8, 8)  # noqa: NAR001
    _, weights = attention(q=q, k=k, v=v, mask=None)
    torch.testing.assert_close(
        actual=weights.sum(dim=-1),
        expected=torch.ones(2, 4, 8),  # noqa: NAR001
    )
