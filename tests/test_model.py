"""Model client tests against a fake Responses HTTP endpoint."""
from __future__ import annotations

import json

import pytest
import requests

from sweagent import model as model_module
from sweagent.model import ContextLimitError, ResponsesModel

HISTORY = [{"role": "user", "content": [{"type": "input_text", "text": "hello"}]}]
TOOLS = [{
    "type": "function",
    "name": "bash",
    "description": "Run a command.",
    "parameters": {"type": "object", "properties": {}, "required": [],
                   "additionalProperties": False},
    "strict": True,
}]


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} response", response=None)


def install_post(monkeypatch, *responses):
    calls = []
    remaining = list(responses)

    def fake_post(url, **kwargs):
        calls.append({"url": url, **kwargs})
        return remaining.pop(0)

    monkeypatch.setattr(model_module.requests, "post", fake_post)
    return calls


def test_request_is_stateless_and_uses_configured_settings(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "ignored-environment-key")
    body = {"id": "resp_1", "output": [{"type": "reasoning", "encrypted_content": "opaque"}]}
    calls = install_post(monkeypatch, FakeResponse(body))
    config = {
        "model": "test-model",
        "base_url": "http://endpoint.invalid/v1/",
        "api_key": "sk-model-secret",
        "reasoning_effort": "high",
        "temperature": 0,
        "top_p": 0.9,
    }

    result = ResponsesModel(config).complete(HISTORY, TOOLS)

    assert result == body
    assert len(calls) == 1
    call = calls[0]
    assert call["url"] == "http://endpoint.invalid/v1/responses"
    assert call["timeout"] is None
    assert call["stream"] is False
    assert call["headers"] == {"Authorization": "Bearer sk-model-secret"}
    assert call["json"] == {
        "model": "test-model",
        "input": HISTORY,
        "tools": TOOLS,
        "reasoning": {"effort": "high"},
        "include": ["reasoning.encrypted_content"],
        "parallel_tool_calls": False,
        "store": False,
        "truncation": "disabled",
        "temperature": 0,
        "top_p": 0.9,
    }
    for forbidden in ("stream", "previous_response_id", "conversation",
                      "max_output_tokens", "max_steps"):
        assert forbidden not in call["json"]
    assert "sk-model-secret" not in json.dumps(call["json"])


def test_defaults_omit_optional_sampling_and_authorization(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "ignored-environment-key")
    calls = install_post(monkeypatch, FakeResponse({"id": "resp_2"}))
    config = {"model": "test-model", "base_url": "http://endpoint.invalid/v1"}

    ResponsesModel(config).complete(HISTORY, TOOLS)

    call = calls[0]
    assert call["url"] == "http://endpoint.invalid/v1/responses"
    assert call["json"] == {
        "model": "test-model",
        "input": HISTORY,
        "tools": TOOLS,
        "reasoning": {"effort": "medium"},
        "include": ["reasoning.encrypted_content"],
        "parallel_tool_calls": False,
        "store": False,
        "truncation": "disabled",
    }
    assert "Authorization" not in call["headers"]


@pytest.mark.parametrize("api_key", [None, ""])
def test_empty_api_key_does_not_fall_back_to_environment(monkeypatch, api_key):
    monkeypatch.setenv("OPENAI_API_KEY", "ignored-environment-key")
    calls = install_post(monkeypatch, FakeResponse({"id": "resp_3"}))
    config = {
        "model": "test-model",
        "base_url": "http://endpoint.invalid/v1",
        "api_key": api_key,
    }
    ResponsesModel(config).complete(HISTORY, TOOLS)
    assert "Authorization" not in calls[0]["headers"]


def test_logged_config_omits_only_key_without_mutating_config():
    config = {"model": "test-model", "reasoning_effort": "medium", "api_key": "private-key"}
    assert model_module.logged_config(config) == {"model": "test-model", "reasoning_effort": "medium"}
    assert config["api_key"] == "private-key"


def test_context_length_error_retains_error_body(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    error_body = {
        "error": {
            "message": "maximum context length exceeded",
            "type": "invalid_request_error",
            "code": "context_length_exceeded",
        }
    }
    calls = install_post(monkeypatch, FakeResponse(error_body, status_code=400))

    with pytest.raises(ContextLimitError) as raised:
        ResponsesModel(
            {"model": "test-model", "base_url": "http://endpoint.invalid/v1"}
        ).complete(HISTORY, TOOLS)

    assert raised.value.body == error_body
    assert "maximum context length exceeded" in str(raised.value)
    assert len(calls) == 1


def test_other_http_errors_propagate_without_retry(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    calls = install_post(
        monkeypatch,
        FakeResponse({"error": {"message": "bad request", "code": "invalid_value"}}, 400),
        FakeResponse({"error": {"message": "server error"}}, 500),
    )
    model = ResponsesModel({"model": "test-model", "base_url": "http://endpoint.invalid/v1"})

    with pytest.raises(requests.HTTPError):
        model.complete(HISTORY, TOOLS)
    with pytest.raises(requests.HTTPError):
        model.complete(HISTORY, TOOLS)

    assert len(calls) == 2


def test_response_level_failed_context_error_is_recognized(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    failed = {
        "id": "resp_failed",
        "status": "failed",
        "error": {"message": "input too long", "code": "context_length_exceeded"},
        "output": [],
    }
    calls = install_post(monkeypatch, FakeResponse(failed))

    with pytest.raises(ContextLimitError) as raised:
        ResponsesModel(
            {"model": "test-model", "base_url": "http://endpoint.invalid/v1"}
        ).complete(HISTORY, TOOLS)

    assert raised.value.body == failed
    assert len(calls) == 1


def test_response_level_server_failure_does_not_become_a_continuation(monkeypatch):
    failure = {"status": "failed", "error": {"code": "server_error", "message": "failed"}}
    calls = install_post(monkeypatch, FakeResponse(failure))
    with pytest.raises(RuntimeError, match="server_error"):
        ResponsesModel({"model": "test", "base_url": "http://endpoint.invalid/v1"}).complete(HISTORY, TOOLS)
    assert len(calls) == 1


def test_non_json_http_error_propagates_http_status_without_retry(monkeypatch):
    class BrokenResponse(FakeResponse):
        def json(self):
            raise requests.exceptions.JSONDecodeError("not JSON", "<html>error</html>", 0)

    calls = install_post(monkeypatch, BrokenResponse(None, status_code=502))
    with pytest.raises(requests.HTTPError, match="502"):
        ResponsesModel({"model": "test", "base_url": "http://endpoint.invalid/v1"}).complete(HISTORY, TOOLS)
    assert len(calls) == 1
