"""A Gymnasium-shaped environment for an LLM agent that writes code.

`reset(task_id, seed, episode_id)` returns the first observation; `step(tool_call)`
returns `StepResult(observation, reward, done, info)`. Everything an episode did lives
in `EpisodeState`, which round-trips through JSON, so an episode can be saved, retried
under the same id, replayed, or shipped to a trainer.

The environment holds one sandbox per episode for `run_python`, acquired on `reset`
and released when the episode ends. `submit` goes to the `Verifier`, which uses its
own fresh sandbox. An infrastructure failure raises `InfraError` instead of producing
an observation, so it can never be scored.
"""

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from training.env.failures import FailureClass, InfraError, classify
from training.env.sandbox import Limits, RunResult, Sandbox
from training.env.tools import RUN_PYTHON, ToolError, parse_tool_call
from training.env.verify import Verifier

INSTRUCTIONS = (
    "Solve the task in Python. Use run_python to test code; its output comes back to "
    "you. Call submit with the complete solution when you are done."
)


@dataclass(frozen=True)
class Task:
    """A prompt for the agent and the tests the verifier runs."""

    task_id: str
    prompt: str
    tests: str


@dataclass
class EpisodeState:
    """Everything an episode did, in order. Serializable for replay and retry.

    `actions` holds the raw tool calls exactly as the agent sent them, so `replay`
    feeds the same bytes back, malformed ones included.
    """

    episode_id: str
    task_id: str
    seed: int
    messages: list[dict[str, Any]] = field(default_factory=list)
    actions: list[Any] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    done: bool = False

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(obj=asdict(obj=self))

    @classmethod
    def from_json(cls, text: str) -> "EpisodeState":
        """Rebuild from `to_json` output."""
        return cls(**json.loads(s=text))


@dataclass(frozen=True)
class StepResult:
    """One step's observation, reward, and whether the episode is over."""

    observation: str
    reward: float
    done: bool
    info: dict[str, Any]


def format_run(result: RunResult) -> str:
    """Render a sandbox run as the observation the agent reads."""
    status = "timed out" if result.timed_out else f"exit code {result.exit_code}"
    return f"{status}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"


class Environment:
    """Coding tasks behind `reset` / `step`, one sandbox per episode."""

    def __init__(
        self,
        tasks: dict[str, Task],
        acquire: Callable[[], Sandbox],
        release: Callable[[Sandbox], None],
        verifier: Verifier,
        limits: Limits,
        max_turns: int = 8,
    ) -> None:
        """Set up; no sandbox is held until `reset`.

        Args:
            acquire: Returns a started sandbox, such as `WarmPool.acquire`.
            release: Takes it back when the episode ends, such as `WarmPool.release`.
            max_turns: Tool calls before the episode ends unsubmitted, reward 0.
        """
        self.tasks = tasks
        self.acquire = acquire
        self.release = release
        self.verifier = verifier
        self.limits = limits
        self.max_turns = max_turns
        self.sandbox: Optional[Sandbox] = None
        self.state = EpisodeState(episode_id="", task_id="", seed=0)

    def reset(self, task_id: str, seed: int, episode_id: str) -> str:
        """Start an episode and return the first observation.

        Side effects:
            Releases any sandbox still held, acquires a new one, and replaces `state`.
        """
        self._release()
        self.sandbox = self.acquire()
        observation = f"{INSTRUCTIONS}\n\n{self.tasks[task_id].prompt}"
        self.state = EpisodeState(episode_id=episode_id, task_id=task_id, seed=seed)
        self.state.messages.append({"role": "user", "content": observation})  # noqa: NAR001
        return observation

    def step(self, tool_call: Any) -> StepResult:
        """Apply one tool call; see the module docstring.

        Args:
            tool_call: `{"id", "name", "arguments"}` as the agent sent it, or anything
                malformed, which becomes an error observation.

        Raises:
            InfraError: If a sandbox failed; the episode should be retried, not scored.

        Side effects:
            Appends to `state`; releases the sandbox when the episode ends.
        """
        self.state.actions.append(tool_call)  # noqa: NAR001
        call = parse_tool_call(raw=tool_call)
        reward, done, info = 0.0, False, {}
        if isinstance(call, ToolError):  # noqa: NAR001
            observation = f"error: {call.message}"
        elif call.name == RUN_PYTHON:
            assert self.sandbox is not None
            result = self.sandbox.run(
                files={"main.py": call.code},
                command=[self.sandbox.python, "main.py"],
                limits=self.limits,
            )
            if classify(result=result) == FailureClass.INFRA:
                raise InfraError(f"episode sandbox failed: {result.stderr[:200]}")  # noqa: NAR001
            observation = format_run(result=result)
        else:
            verdict = self.verifier.verify(task_id=self.state.task_id, code=call.code)
            reward, done, info = verdict.reward, True, {"verdict": verdict.reason}
            observation = f"submitted: {verdict.reason}"
        done = done or len(self.state.actions) >= self.max_turns  # noqa: NAR001
        self._record(tool_call=tool_call, observation=observation, reward=reward, done=done)
        return StepResult(observation=observation, reward=reward, done=done, info=info)

    def _record(self, tool_call: Any, observation: str, reward: float, done: bool) -> None:
        """Append the call and its observation as chat messages, and the reward.

        Side effects:
            Mutates `state`; releases the sandbox if `done`.
        """
        call = tool_call if isinstance(tool_call, dict) else {}  # noqa: NAR001
        call_id = str(call.get("id") or "call")  # noqa: NAR001
        name = str(call.get("name"))  # noqa: NAR001
        arguments = str(call.get("arguments", tool_call))  # noqa: NAR001
        function = {"name": name, "arguments": arguments}
        tool_calls = [{"id": call_id, "type": "function", "function": function}]
        self.state.messages.append({"role": "assistant", "tool_calls": tool_calls})  # noqa: NAR001
        self.state.messages.append(  # noqa: NAR001
            {"role": "tool", "tool_call_id": call_id, "content": observation}
        )  # noqa: NAR001
        self.state.rewards.append(reward)  # noqa: NAR001
        self.state.done = done
        if done:
            self._release()

    def _release(self) -> None:
        """Hand the held sandbox back, if any."""
        if self.sandbox is not None:
            self.release(self.sandbox)  # noqa: NAR001
            self.sandbox = None


def replay(state: EpisodeState, env: Environment) -> list[float]:
    """Feed a saved episode's actions into `env` again and return the rewards.

    A deterministic environment returns exactly `state.rewards`; a difference means
    the episode depended on something outside its state.
    """
    env.reset(task_id=state.task_id, seed=state.seed, episode_id=state.episode_id)
    return [env.step(tool_call=action).reward for action in state.actions]
