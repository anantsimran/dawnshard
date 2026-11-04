import torch
from training.serve.sampler import IncrementalDetokenizer, SamplingParams, StopChecker, sample


def test_multi_byte_utf8_split_across_tokens_detokenizes_whole():
    text = "über 日本"
    detok = IncrementalDetokenizer(token_bytes=lambda t: bytes([t]))
    pieces = [detok.push(token=b) for b in text.encode()]
    assert "".join(pieces) + detok.flush() == text
    assert pieces[0] == "" and pieces[1] == "ü"  # the first byte alone is held back


def test_stop_string_spanning_two_chunks_is_caught_and_cut():
    checker = StopChecker(stops=("STOP",))
    out = checker.feed(text="hello ST")
    out += checker.feed(text="OP and more")
    assert checker.stopped and out == "hello "


def test_text_without_a_stop_is_released_in_full():
    checker = StopChecker(stops=("XYZ",))
    out = checker.feed(text="abcXY") + checker.feed(text="w") + checker.flush()
    assert out == "abcXYw" and not checker.stopped


def test_greedy_top_k_and_seeded_sampling():
    logits = torch.tensor([0.0, 3.0, 1.0, 2.0])
    assert sample(logits=logits, params=SamplingParams(), generator=None) == 1
    params = SamplingParams(temperature=1.0, top_k=2)
    draws = {sample(logits=logits, params=params, generator=torch.Generator().manual_seed(s)) for s in range(50)}
    assert draws <= {1, 3}
    seeded = [sample(logits=logits, params=SamplingParams(temperature=1.0), generator=torch.Generator().manual_seed(7)) for _ in range(3)]
    assert len(set(seeded)) == 1


def test_top_p_keeps_the_token_that_crosses_p():
    logits = torch.tensor([10.0, 0.0, 0.0, 0.0])
    params = SamplingParams(temperature=1.0, top_p=0.5)
    assert {sample(logits=logits, params=params, generator=torch.Generator().manual_seed(s)) for s in range(30)} == {0}
