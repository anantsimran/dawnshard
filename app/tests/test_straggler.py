from training.train.straggler import StragglerDetector


def test_flags_only_the_slow_rank_while_the_others_wait():
    detector = StragglerDetector(num_ranks=4)
    for _ in range(20):
        detector.record(compute_s=[0.10, 0.10, 0.16, 0.11], wait_s=[0.06, 0.06, 0.001, 0.05])
    assert detector.stragglers() == [2]


def test_flags_nothing_on_a_balanced_step():
    detector = StragglerDetector(num_ranks=4)
    for _ in range(20):
        detector.record(compute_s=[0.10, 0.11, 0.10, 0.105], wait_s=[0.01, 0.0, 0.01, 0.005])
    assert detector.stragglers() == []


def test_a_busier_rank_nobody_waits_for_is_not_a_straggler():
    detector = StragglerDetector(num_ranks=3)
    for _ in range(20):
        detector.record(compute_s=[0.10, 0.10, 0.15], wait_s=[0.001, 0.001, 0.001])
    assert detector.stragglers() == []


def test_the_window_forgets_old_steps():
    detector = StragglerDetector(num_ranks=3, window=5)
    for _ in range(10):
        detector.record(compute_s=[0.1, 0.3, 0.1], wait_s=[0.2, 0.0, 0.2])
    for _ in range(5):
        detector.record(compute_s=[0.1, 0.1, 0.1], wait_s=[0.0, 0.0, 0.0])
    assert detector.stragglers() == []
