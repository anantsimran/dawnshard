import pytest
from training.experiments.serving_platform import (
    DOLLARS_PER_REPLICA_HOUR,
    make_trace,
    random_vs_two_choices,
)


def test_the_trace_is_reproducible_from_its_seed():
    assert make_trace(seed=4) == make_trace(seed=4)
    assert make_trace(seed=4) != make_trace(seed=5)


def test_two_choices_beats_one_random_choice():
    result = random_vs_two_choices()
    assert result["max_two_choices"] < result["max_one_choice"]
    assert result["max_two_choices"] <= result["mean"] + 3


def test_dollars_per_million_tokens_arithmetic():
    tokens_per_s = 1500
    dollars = DOLLARS_PER_REPLICA_HOUR / (tokens_per_s * 3600 / 1e6)
    assert dollars == pytest.approx(0.1309, abs=1e-4)
