"""C0: a minimal bash-only agent with linear Responses API history."""
from __future__ import annotations

import json
import time
from pathlib import Path

from sweagent.model import ContextLimitError, ResponsesModel, logged_config

CONTINUE = "Continue, or call finish to submit the current changes."
DEFAULT_MAX_OBSERVATION_CHARS = 20000

TOOLS = [
    {
        "type": "function",
        "name": "bash",
        "description": "Run a Bash command in the repository.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The Bash command to run."}
            },
            "required": ["command"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "finish",
        "description": "Submit the current working tree as the final answer.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


def _initial_prompt(problem_statement: str, environment, limit: int) -> str:
    return f"""You are fixing a software issue in the repository at {environment.workdir}.

Resolve the issue by modifying the repository. Choose your own approach and any verification you find useful. The repository is prepared: its dependencies are installed and its environment is active. Each bash call starts a fresh Bash shell in that directory; no shell state (working directory, variables, aliases, history) persists between calls. Commands time out after {environment.command_timeout:g} seconds. Command output longer than {limit} characters is truncated in the middle; the full output stays in the runner log. The exit code and timeout status are always reported.

Make exactly one tool invocation per turn.

Tools:
- bash(command): run a Bash command in the repository.
- finish(): submit the current working tree and end the task. Call it when you are done, whether or not tests pass.

Issue:
{problem_statement}
"""


def _observation(result, limit: int) -> str:
    output = result.output
    if len(output) > limit:
        head = limit // 2
        tail = limit - head
        omitted = len(output) - limit
        output = (
            f"{output[:head]}\n[... {omitted} characters omitted ...]\n"
            f"{output[len(output) - tail:]}"
        )
    timed_out = "true" if result.timed_out else "false"
    return f"exit_code: {result.returncode}\ntimed_out: {timed_out}\noutput:\n{output}"


def _tool_action(call: dict):
    name = call.get("name")
    try:
        arguments = json.loads(call.get("arguments", ""))
    except (json.JSONDecodeError, TypeError):
        return None, "error: invalid JSON arguments"
    if not isinstance(arguments, dict):
        return None, "error: arguments must be a JSON object"
    if name == "bash":
        command = arguments.get("command")
        if not isinstance(command, str):
            return None, "error: bash requires a string 'command' argument"
        return ("bash", command), None
    if name == "finish":
        if arguments:
            return None, "error: finish requires an empty arguments object"
        return ("finish", None), None
    return None, f"error: unknown tool {name!r}"


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_agent(problem_statement: str, environment, config: dict, log_dir: Path) -> dict:
    started = time.monotonic()
    log_dir = Path(log_dir)
    model = ResponsesModel(config)
    limit = config.get("max_observation_chars", DEFAULT_MAX_OBSERVATION_CHARS)
    prompt = _initial_prompt(problem_statement, environment, limit)
    history = [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}]
    usage = {
        "model_requests": 0,
        "bash_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "model_seconds": 0.0,
        "bash_seconds": 0.0,
        "total_seconds": 0.0,
        "stop_reason": "error",
    }

    with (log_dir / "trajectory.jsonl").open("a", encoding="utf-8") as trajectory:
        def log(event: dict) -> None:
            trajectory.write(json.dumps(event, ensure_ascii=False) + "\n")
            trajectory.flush()

        def record_result(turn: int, call: dict, observation: str, **fields) -> None:
            history.append({
                "type": "function_call_output",
                "call_id": call["call_id"],
                "output": observation,
            })
            log({
                "type": "tool_result",
                "turn": turn,
                "call_id": call.get("call_id"),
                "name": call.get("name"),
                "arguments": call.get("arguments"),
                "observation": observation,
                **fields,
            })

        log({
            "type": "start",
            "model": config["model"],
            "config": logged_config(config),
            "prompt": prompt,
            "tools": TOOLS,
            "workdir": environment.workdir,
            "command_timeout": environment.command_timeout,
            "max_observation_chars": limit,
        })

        phase = "model"
        try:
            while True:
                phase = "model"
                usage["model_requests"] += 1
                turn = usage["model_requests"]
                log({"type": "request", "turn": turn})
                print(f"[c0] turn {turn}: request", flush=True)
                request_started = time.monotonic()
                try:
                    response = model.complete(history, TOOLS)
                finally:
                    usage["model_seconds"] += time.monotonic() - request_started

                phase = "response"
                log({"type": "response", "turn": turn, "response": response})
                reported = response["usage"]
                usage["input_tokens"] += reported["input_tokens"]
                usage["output_tokens"] += reported["output_tokens"]
                output = response["output"]
                history.extend(output)
                calls = [item for item in output if item["type"] == "function_call"]

                if len(calls) > 1:
                    observation = "error: more than one tool call in one response; no calls were executed"
                    for call in calls:
                        record_result(turn, call, observation, executed=False)
                    print(f"[c0] turn {turn}: protocol error", flush=True)
                    continue

                if not calls:
                    history.append({
                        "role": "user",
                        "content": [{"type": "input_text", "text": CONTINUE}],
                    })
                    log({"type": "continuation", "turn": turn, "text": CONTINUE})
                    print(f"[c0] turn {turn}: no tool call", flush=True)
                    continue

                call = calls[0]
                action, error = _tool_action(call)
                if error is not None:
                    record_result(turn, call, error, executed=False)
                    print(f"[c0] turn {turn}: tool error", flush=True)
                    continue

                kind, command = action
                if kind == "finish":
                    log({"type": "finish", "turn": turn, "call_id": call["call_id"]})
                    usage["stop_reason"] = "finish"
                    print(f"[c0] turn {turn}: finish", flush=True)
                    break

                phase = "bash"
                usage["bash_calls"] += 1
                bash_started = time.monotonic()
                try:
                    result = environment.execute(command)
                finally:
                    usage["bash_seconds"] += time.monotonic() - bash_started

                observation = _observation(result, limit)
                record_result(
                    turn, call, observation, executed=True, command=command,
                    exit_code=result.returncode, timed_out=result.timed_out,
                    full_output=result.output,
                )
                print(f"[c0] turn {turn}: bash exit={result.returncode} "
                      f"timed_out={result.timed_out}", flush=True)
        except ContextLimitError as error:
            log({"type": "model_error", "turn": usage["model_requests"],
                 "error": "context_length_exceeded", "body": error.body})
            usage["stop_reason"] = "context_limit"
            print("[c0] context limit", flush=True)
        except KeyboardInterrupt:
            log({"type": "interrupted", "turn": usage["model_requests"], "phase": phase})
            usage["stop_reason"] = "interrupted"
            print("[c0] interrupted", flush=True)
        finally:
            usage["total_seconds"] = time.monotonic() - started
            log({"type": "stop", "stop_reason": usage["stop_reason"], "usage": usage})
            _write_json(log_dir / "history.json", history)
            _write_json(log_dir / "agent_usage.json", usage)
    return usage
