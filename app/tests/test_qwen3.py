import pytest
import torch
import torch.nn.functional as F
from training.model.qwen3 import (
    QWEN3_0_6B,
    QWEN3_1_7B,
    QWEN3_TINY,
    Qwen3,
    Qwen3Attention,
    apply_rope,
    count_parameters,
)


@pytest.mark.parametrize(("config", "expected"), [(QWEN3_0_6B, 0.596e9), (QWEN3_1_7B, 1.72e9)])
def test_official_configs_have_the_published_parameter_counts(config, expected):
    with torch.device("meta"):
        model = Qwen3(config=config)
    measured = sum(p.numel() for p in model.parameters())
    assert measured == count_parameters(config=config)
    assert abs(measured - expected) / expected < 0.005


def test_output_shape_is_flat_logits():
    model = Qwen3(config=QWEN3_TINY)
    ids = torch.randint(low=0, high=QWEN3_TINY.vocab_size, size=(2, 10))
    assert model(ids).shape == (20, QWEN3_TINY.vocab_size)
    assert model.hidden_states(input_ids=ids).shape == (2, 10, QWEN3_TINY.hidden_size)


def test_a_later_token_never_changes_earlier_logits():
    torch.manual_seed(seed=0)
    model = Qwen3(config=QWEN3_TINY).eval()
    ids = torch.randint(low=0, high=QWEN3_TINY.vocab_size, size=(1, 12))
    changed = ids.clone()
    changed[0, 7] = (ids[0, 7] + 1) % QWEN3_TINY.vocab_size
    with torch.no_grad():
        before, after = model(ids).view(12, -1), model(changed).view(12, -1)
    torch.testing.assert_close(actual=after[:7], expected=before[:7])
    assert not torch.allclose(after[7:], before[7:])


def test_gqa_groups_consecutive_query_heads_on_one_kv_head():
    torch.manual_seed(seed=0)
    attn = Qwen3Attention(config=QWEN3_TINY).eval()
    x = torch.randn(2, 6, QWEN3_TINY.hidden_size)
    cos, sin = torch.ones(6, QWEN3_TINY.head_dim), torch.zeros(6, QWEN3_TINY.head_dim)
    group = QWEN3_TINY.num_attention_heads // QWEN3_TINY.num_key_value_heads
    with torch.no_grad():
        q = attn.q_norm(attn.q_proj(x).view(2, 6, -1, QWEN3_TINY.head_dim)).transpose(1, 2)
        k = attn.k_norm(attn.k_proj(x).view(2, 6, -1, QWEN3_TINY.head_dim)).transpose(1, 2)
        v = attn.v_proj(x).view(2, 6, -1, QWEN3_TINY.head_dim).transpose(1, 2)
        k, v = k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)
        manual = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        manual = attn.o_proj(manual.transpose(1, 2).reshape(2, 6, -1))
        torch.testing.assert_close(actual=attn(x=x, cos=cos, sin=sin), expected=manual)


def test_rope_is_a_rotation_and_relative():
    x = torch.randn(1, 1, 1, 8)
    angle = torch.tensor(0.3)
    cos, sin = torch.cos(angle).expand(1, 8), torch.sin(angle).expand(1, 8)
    rotated = apply_rope(x=x, cos=cos, sin=sin)
    torch.testing.assert_close(actual=rotated.norm(), expected=x.norm())


def test_small_model_overfits_one_batch():
    torch.manual_seed(seed=0)
    model = Qwen3(config=QWEN3_TINY)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=3e-3)
    ids = torch.randint(low=0, high=QWEN3_TINY.vocab_size, size=(4, 17))
    losses = []
    for _ in range(30):
        loss = F.cross_entropy(model(ids[:, :-1]), ids[:, 1:].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] / 2


def test_checkpointing_modes_give_the_same_gradients():
    ids = torch.randint(low=0, high=QWEN3_TINY.vocab_size, size=(2, 9))
    grads = []
    for mode in ("none", "selective", "full"):
        torch.manual_seed(seed=0)
        model = Qwen3(config=QWEN3_TINY, checkpointing=mode)
        F.cross_entropy(model(ids[:, :-1]), ids[:, 1:].reshape(-1)).backward()
        grads.append(model.layers[0].mlp.up_proj.weight.grad)
    torch.testing.assert_close(actual=grads[1], expected=grads[0])
    torch.testing.assert_close(actual=grads[2], expected=grads[0])


def test_hidden_output_feeds_a_head_that_reproduces_the_logits():
    torch.manual_seed(seed=0)
    model = Qwen3(config=QWEN3_TINY).eval()
    ids = torch.randint(low=0, high=QWEN3_TINY.vocab_size, size=(2, 5))
    logits = model(ids)
    model.output = "hidden"
    hidden = model(ids)
    assert hidden.shape == (10, QWEN3_TINY.hidden_size)
    torch.testing.assert_close(actual=hidden @ model.lm_head_weight.T, expected=logits)
