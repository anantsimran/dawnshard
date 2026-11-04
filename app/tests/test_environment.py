import json

import pytest
from training.env.environment import Environment, EpisodeState, Task, replay
from training.env.failures import InfraError
from training.env.sandbox import Limits, RunResult, SubprocessSandbox, WarmPool
from training.env.tools import RUN_PYTHON, SUBMIT
from training.env.verify import Verifier

TASK = Task(task_id="add", prompt="Write add(a, b).", tests="assert add(1, 2) == 3")


def _call(name: str, code: str) -> dict:
    return {"id": "c", "name": name, "arguments": json.dumps({"code": code})}


@pytest.fixture
def make_env(tmp_path):
    verifier = Verifier(tests_dir=tmp_path, sandbox_factory=SubprocessSandbox, limits=Limits())
    verifier.register(task_id=TASK.task_id, tests=TASK.tests)
    pool = WarmPool(factory=SubprocessSandbox, size=1)

    def make(max_turns: int = 4) -> Environment:
        return Environment(
            tasks={TASK.task_id: TASK},
            acquire=pool.acquire,
            release=pool.release,
            verifier=verifier,
            limits=Limits(),
            max_turns=max_turns,
        )

    yield make
    pool.close()


def test_run_python_then_submit(make_env):
    env = make_env()
    assert "add(a, b)" in env.reset(task_id="add", seed=0, episode_id="e")
    step = env.step(tool_call=_call(name=RUN_PYTHON, code="print(2 + 2)"))
    assert ("exit code 0", "4") == (step.observation.splitlines()[0], step.observation.split("\n")[2])
    assert not step.done
    final = env.step(tool_call=_call(name=SUBMIT, code="def add(a, b):\n    return a + b"))
    assert (final.reward, final.done) == (1.0, True)


def test_malformed_call_is_an_observation(make_env):
    env = make_env()
    env.reset(task_id="add", seed=0, episode_id="e")
    step = env.step(tool_call="{broken")
    assert step.observation.startswith("error:")
    assert step.reward == 0.0


def test_turn_limit_ends_the_episode_unscored(make_env):
    env = make_env(max_turns=2)
    env.reset(task_id="add", seed=0, episode_id="e")
    env.step(tool_call=_call(name=RUN_PYTHON, code="pass"))
    assert env.step(tool_call=_call(name=RUN_PYTHON, code="pass")).done


def test_episode_state_round_trips_and_replays(make_env):
    env = make_env()
    env.reset(task_id="add", seed=7, episode_id="e")
    env.step(tool_call="{broken")
    env.step(tool_call=_call(name=RUN_PYTHON, code="print('hi')"))
    env.step(tool_call=_call(name=SUBMIT, code="def add(a, b):\n    return a + b"))
    state = EpisodeState.from_json(text=env.state.to_json())
    assert state == env.state
    assert replay(state=state, env=make_env()) == state.rewards == [0.0, 0.0, 1.0]


def test_a_sandbox_infra_failure_raises_instead_of_scoring(tmp_path):
    class Broken(SubprocessSandbox):
        def run(self, files, command, limits, stdin=""):
            return RunResult(exit_code=-1, stdout="", stderr="", duration_s=0.0, start_failed=True)

    verifier = Verifier(tests_dir=tmp_path, sandbox_factory=SubprocessSandbox, limits=Limits())
    verifier.register(task_id="add", tests=TASK.tests)
    env = Environment(
        tasks={"add": TASK}, acquire=Broken, release=lambda sandbox: None,
        verifier=verifier, limits=Limits(),
    )
    env.reset(task_id="add", seed=0, episode_id="e")
    with pytest.raises(InfraError):
        env.step(tool_call=_call(name=RUN_PYTHON, code="pass"))
