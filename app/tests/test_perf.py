import pytest
import torch
from torch.utils.flop_counter import FlopCounterMode
from training.model.qwen3 import QWEN3_0_6B, QWEN3_TINY, Qwen3, model_shape
from training.train.perf import (
    ModelShape,
    StepTimer,
    elementwise_intensity,
    flops_per_token,
    matmul_intensity,
    model_flops_utilization,
    predict_memory,
    static_bytes,
)

# Two layers, d=4, two heads of width 2 (one KV head), ff=8, vocab 10, tied head.
TOY = ModelShape(
    num_parameters=2 * (4 * 4 + 2 * 4 * 2 + 4 * 4 + 2 * 2 + 3 * 4 * 8 + 2 * 4) + 10 * 4 + 4,
    num_layers=2, hidden_size=4, num_heads=2, num_kv_heads=1, head_dim=2,
    intermediate_size=8, vocab_size=10, tied_head=True,
)


def test_flops_match_a_hand_count_on_a_two_layer_toy():
    # Per layer: q 16, k 8, v 8, o 16, norms 4+8, MLP 96 = 156; x2 = 312; embedding 40;
    # final norm 4. N = 356, all in matmuls because the head is tied.
    assert TOY.num_parameters == 356
    # 6N + 12 * layers * heads * head_dim * seq at seq 5: 2136 + 480.
    assert flops_per_token(shape=TOY, seq_len=5) == 2136 + 480


def test_untied_embedding_lookup_is_not_compute():
    untied = ModelShape(**{**TOY.__dict__, "tied_head": False})
    assert flops_per_token(shape=untied, seq_len=5) == 6 * (356 - 40) + 480


def test_qwen3_0_6b_matches_the_worked_example():
    assert flops_per_token(shape=model_shape(config=QWEN3_0_6B), seq_len=2048) == pytest.approx(4.99e9, rel=0.01)
    assert model_flops_utilization(tokens_per_s=9000, flops_per_token=4.99e9, peak_flops=165e12) == pytest.approx(0.272, abs=1e-3)


def test_6n_equals_counted_matmul_flops_on_a_real_model():
    # FlopCounterMode has no formula for the CPU SDPA kernel, so it counts only the
    # weight matmuls here; those must be exactly 6 FLOPs per weight per token. Norm
    # weights are in N but do elementwise work, so they are left out of the count.
    torch.manual_seed(seed=0)
    model = Qwen3(config=QWEN3_TINY)
    ids = torch.randint(low=0, high=QWEN3_TINY.vocab_size, size=(2, 32))
    with FlopCounterMode(display=False) as counter:
        model(ids).sum().backward()
    norms = sum(p.numel() for m in model.modules() if isinstance(m, torch.nn.RMSNorm) for p in m.parameters())
    matmul_weights = model_shape(config=QWEN3_TINY).num_parameters - norms
    assert counter.get_total_flops() == 6 * matmul_weights * ids.numel()


def test_static_memory_is_weights_grads_and_adam_moments():
    model = torch.nn.Linear(in_features=10, out_features=3)
    optimizer = torch.optim.Adam(params=model.parameters())
    model(torch.randn(4, 10)).sum().backward()
    optimizer.step()
    n = sum(p.numel() for p in model.parameters())
    steps = 2 * 4  # Adam's per-parameter step counters are scalar tensors
    assert static_bytes(model=model, optimizer=optimizer) == n * 4 * 4 + steps


def test_16_bytes_per_parameter_before_activations():
    prediction = predict_memory(shape=model_shape(config=QWEN3_0_6B), tokens=1)
    assert prediction.static == 596_049_920 * 16


def test_logits_term_matches_the_worked_example_and_chunking_drops_it():
    shape = model_shape(config=QWEN3_0_6B)
    assert predict_memory(shape=shape, tokens=4 * 2048).logits == 2 * 4 * 2048 * 151936 * 4
    assert predict_memory(shape=shape, tokens=4 * 2048, chunked_loss=True).logits == 0


def test_checkpointing_shrinks_activations_in_order():
    shape = model_shape(config=QWEN3_0_6B)
    acts = [predict_memory(shape=shape, tokens=2048, checkpointing=c).activations for c in ("none", "selective", "full")]
    assert acts[0] > acts[1] > acts[2]


def test_arithmetic_intensity_of_a_matmul_and_an_elementwise_op():
    assert matmul_intensity(m=4096, n=3072, k=1024, bytes_per_element=2) == pytest.approx(647, abs=1)
    assert elementwise_intensity(flops_per_element=4, tensors_moved=2, bytes_per_element=2) == 1.0


def test_step_timer_counts_whole_steps_after_warmup():
    timer = StepTimer(accumulation_steps=2, tokens_per_step=100, warmup_steps=1)
    timer.stamps = [0.0, 1.0, 1.5, 2.0]
    assert timer.tokens_per_s() == pytest.approx(200.0)
    timer(None, None, None, None, 1, 0, None)
    assert len(timer.stamps) == 4
