"""Minimal client for the stateless OpenAI Responses API."""
from __future__ import annotations

import requests


def logged_config(config: dict) -> dict:
    """Record model settings without copying the API credential into artifacts."""
    return {name: value for name, value in config.items() if name != "api_key"}


class ContextLimitError(RuntimeError):
    """The endpoint rejected a request because the context window was exhausted."""

    def __init__(self, message: str, body: dict):
        super().__init__(message)
        self.body = body


class ResponsesModel:
    """POST one request to config['base_url'] + '/responses' and return raw JSON."""

    def __init__(self, config: dict):
        self.config = config
        self.url = config["base_url"].rstrip("/") + "/responses"

    def complete(self, history: list[dict], tools: list[dict]) -> dict:
        payload = {
            "model": self.config["model"],
            "input": history,
            "tools": tools,
            "reasoning": {"effort": self.config.get("reasoning_effort", "medium")},
            "include": ["reasoning.encrypted_content"],
            "parallel_tool_calls": False,
            "store": False,
            "truncation": "disabled",
        }
        if "temperature" in self.config:
            payload["temperature"] = self.config["temperature"]
        if "top_p" in self.config:
            payload["top_p"] = self.config["top_p"]

        headers = {}
        key = self.config.get("api_key")
        if key:
            headers["Authorization"] = f"Bearer {key}"

        response = requests.post(
            self.url, json=payload, headers=headers, timeout=None, stream=False
        )
        try:
            body = response.json()
        except requests.exceptions.JSONDecodeError:
            response.raise_for_status()
            raise
        error = body.get("error")
        if error and error.get("code") == "context_length_exceeded":
            raise ContextLimitError(error["message"], body)
        response.raise_for_status()
        if body.get("status") == "failed":
            raise RuntimeError(error)
        return body
