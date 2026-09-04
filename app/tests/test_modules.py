import torch
from transformer.modules import MultiHeadAttentionLayer


def test_output_preserves_d_model():
    torch.manual_seed(seed=0)
    layer = MultiHeadAttentionLayer(d_model=64, h=8)
    x = torch.randn(2, 10, 64)  # noqa: NAR001
    assert layer(x).shape == (2, 10, 64)  # noqa: NAR001
