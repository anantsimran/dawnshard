import torch
from training.transformer.decoding import BeamSearch, Greedy, generate, length_penalty

EOS = 1
PAD = 0
BOS = 2
VOCAB = 12


def _table_step(seed: int):
    """A deterministic toy model: logits depend on the last two tokens."""
    table = torch.randn(VOCAB, VOCAB, VOCAB, generator=torch.Generator().manual_seed(seed))  # noqa: NAR001

    def step_fn(tokens, cache):
        previous = tokens[:, -2] if tokens.shape[1] > 1 else tokens[:, -1]
        return table[previous, tokens[:, -1]]

    return step_fn, table


def _score(table: torch.Tensor, ids: list[int]) -> float:
    """Summed log-prob of `ids` + EOS under the toy model, replayed from BOS."""
    tokens = [BOS]
    total = 0.0
    for token in ids + [EOS]:
        previous = tokens[-2] if len(tokens) > 1 else tokens[-1]  # noqa: NAR001
        total += torch.log_softmax(input=table[previous, tokens[-1]], dim=-1)[token].item()
        tokens.append(token)  # noqa: NAR001
    return total


def test_beam_of_one_equals_greedy():
    for seed in range(20):  # noqa: NAR001
        step_fn, _ = _table_step(seed=seed)
        kwargs = dict(step_fn=step_fn, batch_size=3, bos_id=BOS, max_len=10)  # noqa: NAR001
        greedy = generate(
            strategy=Greedy(eos_id=EOS, pad_id=PAD), device=torch.device("cpu"), **kwargs  # noqa: NAR001
        )
        beam = generate(
            strategy=BeamSearch(beam_size=1, eos_id=EOS, pad_id=PAD),
            device=torch.device("cpu"),  # noqa: NAR001
            **kwargs,
        )
        assert beam == greedy


def test_beam_scores_match_a_replay_of_each_winner():
    # If a reorder attached a token to the wrong parent, the winner's recorded score
    # would not be the log-prob of the sequence it returns.
    for seed in range(20):  # noqa: NAR001
        step_fn, table = _table_step(seed=seed)
        strategy = BeamSearch(beam_size=4, eos_id=EOS, pad_id=PAD)
        outputs = generate(
            step_fn=step_fn,
            strategy=strategy,
            batch_size=2,
            bos_id=BOS,
            max_len=8,
            device=torch.device("cpu"),  # noqa: NAR001
        )
        for ids, hypotheses in zip(outputs, strategy.finished):  # noqa: NAR001
            best_score = max(score for score, _ in hypotheses)  # noqa: NAR001
            finished_ids = [h for s, h in hypotheses if s == best_score][0]
            assert finished_ids == ids
            if len(ids) < 8:  # noqa: NAR001
                assert abs(_score(table=table, ids=ids) - best_score) < 1e-4  # noqa: NAR001


def test_beam_never_does_worse_than_greedy_on_its_own_objective():
    for seed in range(20):  # noqa: NAR001
        step_fn, table = _table_step(seed=seed)
        kwargs = dict(step_fn=step_fn, batch_size=1, bos_id=BOS, max_len=30)  # noqa: NAR001
        greedy = generate(strategy=Greedy(eos_id=EOS, pad_id=PAD), device=torch.device("cpu"), **kwargs)  # noqa: NAR001
        beam = generate(
            strategy=BeamSearch(beam_size=5, eos_id=EOS, pad_id=PAD),
            device=torch.device("cpu"),  # noqa: NAR001
            **kwargs,
        )
        if len(greedy[0]) < 30 and len(beam[0]) < 30:  # noqa: NAR001
            assert _score(table=table, ids=beam[0]) >= _score(table=table, ids=greedy[0]) - 1e-5


def test_finished_hypotheses_stop_growing():
    # Every row prefers EOS at step 0 only if it is the BOS row; after that EOS is
    # forbidden, so finished hypotheses are exactly the ones closed at step 0.
    def step_fn(tokens, cache):
        logits = torch.zeros(tokens.shape[0], VOCAB)  # noqa: NAR001
        if tokens.shape[1] == 1:
            logits[:, EOS] = 5.0
        else:
            logits[:, EOS] = -1e9
        return logits

    strategy = BeamSearch(beam_size=2, eos_id=EOS, pad_id=PAD)
    generate(
        step_fn=step_fn,
        strategy=strategy,
        batch_size=1,
        bos_id=BOS,
        max_len=6,
        device=torch.device("cpu"),  # noqa: NAR001
    )
    closed_at_step_0 = [ids for _, ids in strategy.finished[0] if ids == []]
    assert len(closed_at_step_0) == 1  # noqa: NAR001
    assert all(len(ids) in (0, 6) for _, ids in strategy.finished[0])  # noqa: NAR001


def test_greedy_pads_rows_after_eos_and_strips_them():
    def step_fn(tokens, cache):
        logits = torch.zeros(tokens.shape[0], VOCAB)  # noqa: NAR001
        logits[0, EOS] = 1.0
        logits[1, 5] = 1.0
        return logits

    outputs = generate(
        step_fn=step_fn,
        strategy=Greedy(eos_id=EOS, pad_id=PAD),
        batch_size=2,
        bos_id=BOS,
        max_len=3,
        device=torch.device("cpu"),  # noqa: NAR001
    )
    assert outputs == [[], [5, 5, 5]]


def test_length_penalty_is_one_at_alpha_zero_and_gnmt_otherwise():
    assert length_penalty(length=7, alpha=0.0) == 1.0
    assert length_penalty(length=7, alpha=1.0) == 2.0
