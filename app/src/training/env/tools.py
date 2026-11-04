"""Tool schemas the agent may call, and parsing of its calls.

Two tools: `run_python` executes code in the episode's sandbox and returns its output;
`submit` hands final code to the verifier and ends the episode. Schemas follow the
OpenAI function-calling format, which vLLM's OpenAI-compatible server accepts.

A malformed call is the model's mistake, not the harness's, so `parse_tool_call`
returns a `ToolError` that the environment turns into an observation the model can
read and fix. It never raises.
"""

import json
from dataclasses import dataclass
from typing import Any

RUN_PYTHON = "run_python"
SUBMIT = "submit"

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "Python source."}},
                "required": ["code"],
            },
        },
    }
    for name, description in (
        (RUN_PYTHON, "Run Python code in a sandbox and return exit code, stdout, stderr."),
        (SUBMIT, "Submit the final solution. Ends the episode."),
    )
]


@dataclass(frozen=True)
class ToolCall:
    """A well-formed call: tool name and its `code` argument."""

    name: str
    code: str


@dataclass(frozen=True)
class ToolError:
    """Why a call could not be parsed; shown to the model as its observation."""

    message: str


def parse_tool_call(raw: str | dict[str, Any]) -> ToolCall | ToolError:
    """Parse one OpenAI-style tool call: `{"name": ..., "arguments": "<json>"}`.

    Accepts the call as a dict or as its JSON text, with `arguments` either a JSON
    string (as the API sends it) or an already-parsed dict.
    """
    try:
        call = json.loads(s=raw) if isinstance(raw, str) else raw  # noqa: NAR001
        name = call.get("name") if isinstance(call, dict) else None  # noqa: NAR001
        arguments = call.get("arguments", {}) if isinstance(call, dict) else None  # noqa: NAR001
        if isinstance(arguments, str):  # noqa: NAR001
            arguments = json.loads(s=arguments)
    except json.JSONDecodeError as error:
        return ToolError(message=f"tool call is not valid JSON: {error}")
    if name not in (RUN_PYTHON, SUBMIT):
        return ToolError(message=f"unknown tool {name!r}; use {RUN_PYTHON} or {SUBMIT}")
    if not isinstance(arguments, dict) or not isinstance(arguments.get("code"), str):  # noqa: NAR001
        return ToolError(message=f"{name} needs a string argument 'code'")
    return ToolCall(name=name, code=arguments["code"])
