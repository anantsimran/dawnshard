import pytest
import torch
from training.model.qwen3 import QWEN3_TINY, Qwen3
from training.model.qwen3_serve import build_engine
from training.serve.sampler import SamplingParams
from training.serve.scheduler import SchedulerConfig


def _reference(prompts, max_tokens):
    torch.manual_seed(0)
    model = Qwen3(config=QWEN3_TINY).eval()
    outputs = []
    with torch.no_grad():
        for prompt in prompts:
            tokens = list(prompt)
            for _ in range(max_tokens):
                logits = model(torch.tensor([tokens])).view(len(tokens), -1)
                tokens.append(int(logits[-1].argmax()))
            outputs.append(tokens[len(prompt):])
    return outputs


PROMPTS = [list(range(10, 31)), list(range(10, 31)) + [5, 6], [7] * 5, list(range(100, 140))]


@pytest.mark.parametrize(
    ("scheduler", "prefix", "num_blocks"),
    [
        (SchedulerConfig(), False, 256),  # continuous batching
        (SchedulerConfig(chunk_size=7, max_tokens_per_step=12), True, 256),  # chunked prefill + prefix cache
        (SchedulerConfig(static_batching=True), True, 256),
        (SchedulerConfig(), False, 16),  # too few blocks for all four: forces preemption by recompute
    ],
)
def test_engine_greedy_equals_full_forward_greedy(scheduler, prefix, num_blocks):
    engine = build_engine(config=QWEN3_TINY, num_blocks=num_blocks, scheduler=scheduler, prefix_caching=prefix, block_size=4)
    params = SamplingParams(max_tokens=12, ignore_eos=True)
    reference = _reference(prompts=PROMPTS, max_tokens=12)
    assert engine.generate(prompts=PROMPTS, params=params) == reference
    assert all(count == 0 for count in engine.blocks.refcount)
    if num_blocks == 16:
        assert engine.scheduler.preemptions > 0
    if prefix:  # a second pass reuses the blocks the first one cached
        assert engine.generate(prompts=PROMPTS, params=params) == reference
        assert engine.blocks.hits > 0


def test_a_request_larger_than_the_pool_is_rejected_up_front():
    engine = build_engine(config=QWEN3_TINY, num_blocks=4, scheduler=SchedulerConfig(), prefix_caching=False, block_size=4)
    with pytest.raises(ValueError):
        engine.add_request(prompt=list(range(20)), params=SamplingParams(max_tokens=1))
