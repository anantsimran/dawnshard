import json

from training.env.tools import RUN_PYTHON, SUBMIT, ToolCall, ToolError, parse_tool_call


def test_parses_a_call_with_json_string_arguments():
    call = parse_tool_call(raw={"name": RUN_PYTHON, "arguments": json.dumps({"code": "print(1)"})})
    assert call == ToolCall(name=RUN_PYTHON, code="print(1)")


def test_parses_a_call_given_as_json_text():
    raw = json.dumps({"name": SUBMIT, "arguments": {"code": "x = 1"}})
    assert parse_tool_call(raw=raw) == ToolCall(name=SUBMIT, code="x = 1")


def test_malformed_json_becomes_an_error_not_an_exception():
    assert isinstance(parse_tool_call(raw="{not json"), ToolError)
    assert isinstance(parse_tool_call(raw={"name": RUN_PYTHON, "arguments": "{oops"}), ToolError)


def test_unknown_tool_and_missing_code_are_errors():
    assert isinstance(parse_tool_call(raw={"name": "rm_rf", "arguments": {"code": ""}}), ToolError)
    assert isinstance(parse_tool_call(raw={"name": SUBMIT, "arguments": {}}), ToolError)
    assert isinstance(parse_tool_call(raw=["not", "a", "dict"]), ToolError)
