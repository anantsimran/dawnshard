import pytest
import torch
from training.transformer.kv_cache import KVCache, kv_cache_bytes


def _cache(cache_cross: bool = True) -> KVCache:
    return KVCache(
        num_layers=3,
        batch_size=2,
        h=4,
        d_k=8,
        max_len=5,
        dtype=torch.float32,
        device=torch.device("cpu"),  # noqa: NAR001
        cache_cross=cache_cross,
    )


def test_measured_bytes_equal_the_formula():
    cache = _cache()
    self_bytes = kv_cache_bytes(
        num_layers=3, batch_size=2, h=4, d_k=8, positions=5, dtype=torch.float32
    )
    assert cache.nbytes() == self_bytes
    for layer in range(3):  # noqa: NAR001
        cross = torch.randn(2, 4, 7, 8)  # noqa: NAR001
        cache.store_cross(layer=layer, kv=(cross, cross.clone()))
    cross_bytes = kv_cache_bytes(
        num_layers=3, batch_size=2, h=4, d_k=8, positions=7, dtype=torch.float32
    )
    assert cache.nbytes() == self_bytes + cross_bytes


def test_bytes_per_token_matches_the_llama_example():
    # 32 layers, 8 KV heads, head dim 128, bf16: 128 KiB per token.
    assert kv_cache_bytes(
        num_layers=32, batch_size=1, h=8, d_k=128, positions=1, dtype=torch.bfloat16
    ) == 128 * 1024


def test_append_returns_every_cached_position():
    cache = _cache()
    first = torch.randn(2, 4, 2, 8)  # noqa: NAR001
    second = torch.randn(2, 4, 1, 8)  # noqa: NAR001
    cache.append(layer=0, kv=(first, first))
    keys, values = cache.append(layer=0, kv=(second, second))
    torch.testing.assert_close(actual=keys, expected=torch.cat(tensors=[first, second], dim=2))
    assert cache.lengths == [3, 0, 0]
    assert cache.length == 0


def test_append_past_max_len_raises():
    cache = _cache()
    too_long = torch.randn(2, 4, 6, 8)  # noqa: NAR001
    with pytest.raises(expected_exception=ValueError):
        cache.append(layer=0, kv=(too_long, too_long))


def test_reorder_gathers_rows_in_every_tensor():
    cache = _cache()
    keys = torch.randn(2, 4, 1, 8)  # noqa: NAR001
    cross = torch.randn(2, 4, 7, 8)  # noqa: NAR001
    cache.append(layer=0, kv=(keys, keys))
    cache.store_cross(layer=0, kv=(cross, cross))
    cache.reorder(index=torch.tensor([1, 1]))  # noqa: NAR001
    torch.testing.assert_close(actual=cache.keys[0, 0, :, :1], expected=keys[1])
    torch.testing.assert_close(actual=cache.keys[0, 1, :, :1], expected=keys[1])
    stored = cache.cross(layer=0)
    assert stored is not None
    torch.testing.assert_close(actual=stored[0][0], expected=cross[1])


def test_cross_is_not_kept_when_cache_cross_is_off():
    cache = _cache(cache_cross=False)
    cross = torch.randn(2, 4, 7, 8)  # noqa: NAR001
    cache.store_cross(layer=0, kv=(cross, cross))
    assert cache.cross(layer=0) is None
