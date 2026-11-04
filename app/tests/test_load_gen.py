import pytest
from training.serve.load_gen import RequestRecord, make_trace, summarize


def test_ttft_itl_and_goodput_from_a_synthetic_timestamp_log():
    records = [
        RequestRecord(arrival=0.0, token_times=[0.2, 0.25, 0.3]),  # meets a 0.5 s / 0.1 s SLO
        RequestRecord(arrival=1.0, token_times=[2.0, 2.5]),  # TTFT 1.0 s: misses
    ]
    summary = summarize(records=records, duration=10.0, ttft_slo=0.5, itl_slo=0.1)
    assert summary["ttft_p50_s"] == pytest.approx(0.6)
    assert summary["itl_p99_s"] == pytest.approx(0.5, abs=0.02)
    assert summary["output_tokens_per_s"] == 0.5
    assert summary["goodput_rps"] == 0.1


def test_the_trace_is_reproducible_from_its_seed_and_shares_prefixes():
    a = make_trace(seed=3, num_requests=50, rate=5.0, shared_prefix_share=0.5, num_prefixes=2, prefix_words=10)
    assert a == make_trace(seed=3, num_requests=50, rate=5.0, shared_prefix_share=0.5, num_prefixes=2, prefix_words=10)
    assert a != make_trace(seed=4, num_requests=50, rate=5.0, shared_prefix_share=0.5, num_prefixes=2, prefix_words=10)
    assert sum(r.prompt.startswith("System") for r in a) > 10
    assert all(b.arrival > x.arrival for x, b in zip(a, a[1:]))
