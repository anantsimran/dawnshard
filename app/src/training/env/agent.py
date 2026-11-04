"""The agent loop: ask the model for a tool call, step the environment, repeat.

`chat_completion` talks to any OpenAI-compatible `/v1/chat/completions` endpoint
(vLLM's server, for one) with the standard library, so the package adds no client
dependency. HTTP 5xx and connection failures raise `ModelServerError`, which is an
`InfraError`: the retry logic treats a dead model server like a dead sandbox.
"""

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from training.env.environment import Environment, EpisodeState
from training.env.failures import ModelServerError
from training.env.tools import TOOL_SCHEMAS

type Complete = Callable[[list[dict[str, Any]], int], dict[str, Any]]


def chat_completion(
    base_url: str, model: str, messages: list[dict[str, Any]], seed: int, timeout_s: float = 120.0
) -> dict[str, Any]:
    """Return the assistant message for `messages`, with tools offered, temperature 0.

    Raises:
        ModelServerError: On an HTTP error or no connection.
    """
    body = {
        "model": model,
        "messages": messages,
        "tools": TOOL_SCHEMAS,
        "temperature": 0.0,
        "seed": seed,
    }
    request = urllib.request.Request(
        url=f"{base_url.rstrip('/')}/v1/chat/completions",  # noqa: NAR001
        data=json.dumps(obj=body).encode(),  # noqa: NAR001
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(url=request, timeout=timeout_s) as response:
            return json.loads(s=response.read())["choices"][0]["message"]  # noqa: NAR001
    except urllib.error.HTTPError as error:
        raise ModelServerError(status=error.code) from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise ModelServerError(status=None) from error


def first_tool_call(message: dict[str, Any]) -> dict[str, Any]:
    """Return the message's first tool call as `{"id", "name", "arguments"}`.

    A message with no tool call becomes a call to no tool, carrying the text, which
    the environment answers with an error observation.
    """
    calls = message.get("tool_calls") or []  # noqa: NAR001
    if not calls:
        return {"id": None, "name": None, "arguments": message.get("content")}  # noqa: NAR001
    return {"id": calls[0].get("id"), **calls[0]["function"]}  # noqa: NAR001


def run_episode(
    env: Environment, task_id: str, seed: int, episode_id: str, complete: Complete
) -> tuple[float, EpisodeState]:
    """Run one episode to `submit` or the turn limit; return `(reward, state)`.

    Args:
        complete: `(messages, seed) -> assistant message`, such as
            `chat_completion` with its URL and model bound.

    Raises:
        InfraError: From the environment or the model server; retry the episode.
    """
    env.reset(task_id=task_id, seed=seed, episode_id=episode_id)
    while True:
        message = complete(env.state.messages, seed)  # noqa: NAR001
        result = env.step(tool_call=first_tool_call(message=message))
        if result.done:
            return result.reward, env.state
