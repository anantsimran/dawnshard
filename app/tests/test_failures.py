import pytest
from training.env.failures import (
    EpisodeLedger,
    FailureClass,
    InfraError,
    ModelServerError,
    classify,
    run_with_retries,
)
from training.env.sandbox import RunResult


def _result(**kwargs) -> RunResult:
    return RunResult(**{"exit_code": 0, "stdout": "", "stderr": "", "duration_s": 0.0, **kwargs})


@pytest.mark.parametrize(
    ("fixture", "expected"),
    [
        (_result(), FailureClass.NONE),
        (_result(exit_code=1), FailureClass.TASK),  # genuine failure
        (_result(exit_code=137, oom_killed=True), FailureClass.TASK),  # sandbox OOM
        (_result(exit_code=137, timed_out=True), FailureClass.TASK),  # program's own timeout
        (_result(exit_code=137), FailureClass.INFRA),  # killed from outside: host OOM
        (_result(exit_code=-1, start_failed=True), FailureClass.INFRA),
    ],
)
def test_fixtures_map_to_the_right_class(fixture, expected):
    assert classify(result=fixture) == expected


def test_model_server_errors_are_infra():
    assert isinstance(ModelServerError(status=503), InfraError)


def test_a_retry_with_the_same_episode_id_is_counted_once():
    ledger = EpisodeLedger.empty()
    outcomes = iter([ModelServerError(status=503), 1.0])

    def episode() -> float:
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    assert run_with_retries(episode_id="e1", episode=episode, ledger=ledger) == 1.0
    assert ledger.rewards == {"e1": 1.0}
    assert ledger.attempts == {"e1": 2}


def test_repeated_infra_failure_drops_the_episode_unscored():
    ledger = EpisodeLedger.empty()

    def episode() -> float:
        raise InfraError("sandbox did not start")

    assert run_with_retries(episode_id="e1", episode=episode, ledger=ledger, max_attempts=3) is None
    assert ledger.rewards == {}
    assert ledger.dropped == {"e1"}
