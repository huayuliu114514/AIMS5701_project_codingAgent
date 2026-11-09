"""C0 loop tests with a fake environment and scripted Responses API results."""
from __future__ import annotations

import json
from dataclasses import dataclass

import pytest
import requests

from sweagent.agents import c0_agent
from sweagent.model import ContextLimitError

CONTINUE = c0_agent.CONTINUE


@dataclass
class Result:
    output: str
    returncode: int
    timed_out: bool = False


class FakeEnvironment:
    workdir = "/workspace/repo"
    command_timeout = 60.0

    def __init__(self, results=()):
        self.results = list(results)
        self.commands = []

    def execute(self, command):
        self.commands.append(command)
        value = self.results.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value


@pytest.fixture
def scripted(monkeypatch):
    holder = {"script": []}

    class FakeResponsesModel:
        def __init__(self, config):
            self.config = config
            self.calls = []
            holder["model"] = self

        def complete(self, history, tools):
            self.calls.append((
                json.loads(json.dumps(history)),
                json.loads(json.dumps(tools)),
            ))
            value = holder["script"].pop(0)
            if isinstance(value, BaseException):
                raise value
            return value

    monkeypatch.setattr(c0_agent, "ResponsesModel", FakeResponsesModel)
    return holder


def run_agent(tmp_path, scripted, environment, config=None):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    full_config = {"model": "test-model", **(config or {})}
    usage = c0_agent.run_agent("Fix the bug.", environment, full_config, log_dir)
    return usage, log_dir


def events(log_dir):
    lines = (log_dir / "trajectory.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def response(*output, input_tokens=10, output_tokens=5):
    return {
        "id": "resp_1",
        "object": "response",
        "status": "completed",
        "output": list(output),
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }


def reasoning_item():
    return {
        "type": "reasoning",
        "id": "rs_1",
        "summary": [],
        "encrypted_content": "opaque-reasoning",
    }


def message_item(text="I will inspect the repository."):
    return {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def call_item(name="bash", arguments='{"command":"ls"}', call_id="call_1"):
    return {
        "type": "function_call",
        "id": f"fc_{call_id}",
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
        "status": "completed",
    }


def finish_item(call_id="call_finish"):
    return call_item("finish", "{}", call_id)


def test_full_replay_preserves_reasoning_and_tool_outputs(tmp_path, scripted):
    reasoning = reasoning_item()
    message = message_item()
    call = call_item("bash", '{"command":"git status"}', "call_1")
    finish = finish_item("call_2")
    first_response = response(reasoning, message, call)
    second_response = response(finish)
    scripted["script"].extend([first_response, second_response])
    env = FakeEnvironment([Result("On branch main\n", 0)])

    usage, log_dir = run_agent(tmp_path, scripted, env, {"api_key": "sk-never-log-this"})

    assert set(usage) == {
        "model_requests", "bash_calls", "input_tokens", "output_tokens",
        "model_seconds", "bash_seconds", "total_seconds", "stop_reason",
    }
    assert usage["model_requests"] == 2
    assert usage["bash_calls"] == 1
    assert usage["input_tokens"] == 20
    assert usage["output_tokens"] == 10
    assert usage["model_seconds"] >= 0
    assert usage["bash_seconds"] >= 0
    assert usage["total_seconds"] >= usage["model_seconds"]
    assert usage["stop_reason"] == "finish"
    assert env.commands == ["git status"]

    model = scripted["model"]
    assert model.config["api_key"] == "sk-never-log-this"
    first_history, tools = model.calls[0]
    assert [tool["name"] for tool in tools] == ["bash", "finish"]
    assert tools == c0_agent.TOOLS
    assert tools[0]["parameters"]["required"] == ["command"]
    assert tools[0]["parameters"]["properties"]["command"]["type"] == "string"
    assert tools[1]["parameters"] == {
        "type": "object", "properties": {}, "required": [],
        "additionalProperties": False,
    }
    assert all(tool["strict"] is True for tool in tools)
    prompt = first_history[0]["content"][0]["text"]
    assert "Fix the bug." in prompt
    assert FakeEnvironment.workdir in prompt
    assert "fresh Bash shell" in prompt
    assert "60" in prompt
    assert "20000" in prompt
    assert "bash(command)" in prompt
    assert "finish()" in prompt

    second_history = model.calls[1][0]
    assert model.calls[1][1] == tools
    assert second_history[0] == first_history[0]
    assert second_history[1:4] == [reasoning, message, call]
    output_item = second_history[4]
    assert output_item["type"] == "function_call_output"
    assert output_item["call_id"] == "call_1"
    assert "On branch main" in output_item["output"]

    history = json.loads((log_dir / "history.json").read_text(encoding="utf-8"))
    assert history == second_history + [finish]
    assert json.loads((log_dir / "agent_usage.json").read_text(encoding="utf-8")) == usage

    log = events(log_dir)
    assert [event["type"] for event in log] == [
        "start", "request", "response", "tool_result",
        "request", "response", "finish", "stop",
    ]
    assert log[0]["prompt"] == prompt
    assert "api_key" not in log[0]["config"]
    assert log[0]["tools"] == tools
    assert log[2]["response"] == first_response
    assert log[3]["command"] == "git status"
    assert log[3]["exit_code"] == 0
    assert log[3]["timed_out"] is False
    assert log[3]["full_output"] == "On branch main\n"
    assert log[3]["observation"] == output_item["output"]
    assert log[5]["response"] == second_response
    assert log[7]["stop_reason"] == "finish"
    assert log[7]["usage"] == usage

    for name in ("trajectory.jsonl", "history.json", "agent_usage.json"):
        assert "sk-never-log-this" not in (log_dir / name).read_text(encoding="utf-8")


def test_finish_stops_unconditionally_without_bash(tmp_path, scripted):
    scripted["script"].append(response(finish_item()))
    env = FakeEnvironment()

    usage, _ = run_agent(tmp_path, scripted, env)

    assert usage["stop_reason"] == "finish"
    assert usage["model_requests"] == 1
    assert usage["bash_calls"] == 0
    assert env.commands == []


def test_no_tool_call_appends_continue_reminder(tmp_path, scripted):
    scripted["script"].extend([response(message_item("Thinking.")), response(finish_item())])
    env = FakeEnvironment()

    usage, log_dir = run_agent(tmp_path, scripted, env)

    assert usage["model_requests"] == 2
    assert usage["bash_calls"] == 0
    assert usage["stop_reason"] == "finish"
    second_history = scripted["model"].calls[1][0]
    assert second_history[-1] == {
        "role": "user",
        "content": [{"type": "input_text", "text": CONTINUE}],
    }
    log = events(log_dir)
    assert [event["type"] for event in log] == [
        "start", "request", "response", "continuation",
        "request", "response", "finish", "stop",
    ]
    assert log[3]["text"] == CONTINUE


def test_nonzero_bash_exit_is_reported_and_loop_continues(tmp_path, scripted):
    scripted["script"].extend([
        response(call_item("bash", '{"command":"pytest -q"}', "call_1")),
        response(finish_item()),
    ])
    env = FakeEnvironment([Result("1 failed\n", 1)])

    usage, _ = run_agent(tmp_path, scripted, env)

    assert usage["bash_calls"] == 1
    assert usage["stop_reason"] == "finish"
    observation = scripted["model"].calls[1][0][-1]["output"]
    assert "exit_code: 1" in observation
    assert "timed_out: false" in observation
    assert "1 failed" in observation


def test_observation_caps_output_but_log_keeps_full_output(tmp_path, scripted):
    full_output = "0123456789ABCDEFGHIJ"
    scripted["script"].extend([
        response(call_item("bash", '{"command":"big"}', "call_1")),
        response(finish_item()),
    ])
    env = FakeEnvironment([Result(full_output, 3)])

    usage, log_dir = run_agent(tmp_path, scripted, env, {"max_observation_chars": 10})

    assert usage["bash_calls"] == 1
    observation = scripted["model"].calls[1][0][-1]["output"]
    assert "01234" in observation
    assert "FGHIJ" in observation
    assert "56789ABCDE" not in observation
    assert "exit_code: 3" in observation
    assert "timed_out: false" in observation
    tool_event = [e for e in events(log_dir) if e["type"] == "tool_result"][0]
    assert tool_event["full_output"] == full_output
    assert tool_event["observation"] == observation


def test_timeout_observation_continues(tmp_path, scripted):
    scripted["script"].extend([
        response(call_item("bash", '{"command":"sleep 99"}', "call_1")),
        response(finish_item()),
    ])
    env = FakeEnvironment([Result("partial\n", 124, True)])

    usage, _ = run_agent(tmp_path, scripted, env)

    assert usage["bash_calls"] == 1
    assert usage["model_requests"] == 2
    assert usage["stop_reason"] == "finish"
    observation = scripted["model"].calls[1][0][-1]["output"]
    assert "exit_code: 124" in observation
    assert "timed_out: true" in observation
    assert "partial" in observation


@pytest.mark.parametrize(("name", "arguments", "fragment"), [
    ("bash", "{not json", "invalid JSON"),
    ("bash", '{"cmd":"ls"}', "string 'command'"),
    ("edit", '{"path":"x"}', "unknown tool"),
    ("finish", '{"force": true}', "empty arguments"),
])
def test_bad_tool_calls_are_rejected_without_execution(
    tmp_path, scripted, name, arguments, fragment
):
    bad_call = call_item(name, arguments, "call_bad")
    scripted["script"].extend([response(bad_call), response(finish_item())])
    env = FakeEnvironment()

    usage, log_dir = run_agent(tmp_path, scripted, env)

    assert env.commands == []
    assert usage["bash_calls"] == 0
    assert usage["stop_reason"] == "finish"
    output_item = scripted["model"].calls[1][0][-1]
    assert output_item["call_id"] == "call_bad"
    assert fragment in output_item["output"]
    tool_event = [e for e in events(log_dir) if e["type"] == "tool_result"][0]
    assert tool_event["executed"] is False
    assert tool_event["observation"] == output_item["output"]


def test_multiple_calls_return_protocol_errors_without_execution(tmp_path, scripted):
    first = call_item("bash", '{"command":"ls"}', "call_1")
    second = call_item("bash", '{"command":"pwd"}', "call_2")
    scripted["script"].extend([
        response(first, second),
        response(finish_item("call_3")),
    ])
    env = FakeEnvironment()

    usage, _ = run_agent(tmp_path, scripted, env)

    assert env.commands == []
    assert usage["bash_calls"] == 0
    assert usage["stop_reason"] == "finish"
    next_history = scripted["model"].calls[1][0]
    outputs = [item for item in next_history if item.get("type") == "function_call_output"]
    assert [item["call_id"] for item in outputs] == ["call_1", "call_2"]
    assert all("more than one tool call" in item["output"] for item in outputs)


def test_context_limit_stops_and_returns_usage(tmp_path, scripted):
    body = {"error": {"message": "too long", "code": "context_length_exceeded"}}
    scripted["script"].append(ContextLimitError("too long", body))
    env = FakeEnvironment()

    usage, log_dir = run_agent(tmp_path, scripted, env)

    assert usage["stop_reason"] == "context_limit"
    assert usage["model_requests"] == 1
    assert usage["bash_calls"] == 0
    assert usage["input_tokens"] == 0
    assert usage["output_tokens"] == 0
    assert usage["model_seconds"] >= 0
    assert usage["bash_seconds"] == 0
    assert json.loads((log_dir / "agent_usage.json").read_text(encoding="utf-8")) == usage
    assert json.loads((log_dir / "history.json").read_text(encoding="utf-8")) == (
        scripted["model"].calls[0][0]
    )
    log = events(log_dir)
    assert [event["type"] for event in log] == ["start", "request", "model_error", "stop"]
    assert log[2]["body"] == body
    assert log[3]["stop_reason"] == "context_limit"


def test_unrelated_api_error_propagates_without_retry(tmp_path, scripted):
    scripted["script"].append(requests.HTTPError("500 server error"))
    env = FakeEnvironment()

    with pytest.raises(requests.HTTPError):
        run_agent(tmp_path, scripted, env)

    assert len(scripted["model"].calls) == 1
    logged = json.loads((tmp_path / "logs" / "agent_usage.json").read_text())
    assert logged["stop_reason"] == "error"
    assert logged["model_requests"] == 1
    assert (tmp_path / "logs" / "history.json").exists()


def test_missing_usage_is_not_silently_reported_as_zero(tmp_path, scripted):
    scripted["script"].append({"output": [finish_item()]})
    with pytest.raises(KeyError, match="usage"):
        run_agent(tmp_path, scripted, FakeEnvironment())
    assert json.loads((tmp_path / "logs" / "agent_usage.json").read_text())["stop_reason"] == "error"


@pytest.mark.parametrize("phase", ["model", "bash"])
def test_keyboard_interrupt_returns_usage(tmp_path, scripted, phase):
    env = FakeEnvironment()
    if phase == "model":
        scripted["script"].append(KeyboardInterrupt())
    else:
        scripted["script"].append(
            response(call_item("bash", '{"command":"sleep 5"}', "call_1"))
        )
        env.results.append(KeyboardInterrupt())

    usage, log_dir = run_agent(tmp_path, scripted, env)

    assert usage["stop_reason"] == "interrupted"
    assert usage["model_requests"] == 1
    assert usage["bash_calls"] == (1 if phase == "bash" else 0)
    assert usage["model_seconds"] >= 0
    assert usage["bash_seconds"] >= 0
    assert json.loads((log_dir / "agent_usage.json").read_text(encoding="utf-8")) == usage
    assert json.loads((log_dir / "history.json").read_text(encoding="utf-8"))
    log = events(log_dir)
    assert log[-1]["type"] == "stop"
    assert log[-1]["stop_reason"] == "interrupted"
    assert log[-2]["type"] == "interrupted"
    assert log[-2]["phase"] == phase
