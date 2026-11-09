"""Exercise actual HTTP requests, C0 tool dispatch, and runner artifacts together."""
import json
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from sweagent.dataset import Task
from sweagent.sweep import run


@pytest.fixture
def responses_server():
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append((self.path, body, self.headers.get("Authorization")))
            if len(received) == 1:
                call = {"type": "function_call", "id": "fc_bash", "call_id": "call_bash",
                        "name": "bash", "arguments": json.dumps({"command": "printf 'value = 2\\n' > c0_integration.py"})}
                output = [{"type": "reasoning", "id": "rs_1", "summary": [],
                           "encrypted_content": "opaque-test-reasoning"}, call]
            else:
                output = [{"type": "function_call", "id": "fc_finish", "call_id": "call_finish",
                           "name": "finish", "arguments": "{}"}]
            payload = json.dumps({
                "id": f"resp_{len(received)}", "object": "response", "status": "completed",
                "output": output, "usage": {"input_tokens": 100, "output_tokens": 20},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", received
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


@pytest.mark.parametrize("environment", ["local", "docker"])
def test_http_to_agent_to_submission(tmp_path, responses_server, environment):
    import os
    if environment == "docker" and os.environ.get("RUN_DOCKER_TESTS") != "1":
        pytest.skip("Set RUN_DOCKER_TESTS=1 for the actual SWE-bench container")
    from sweagent.agents.c0_agent import run_agent

    base_url, received = responses_server
    workspace_root = tmp_path / "workspaces"
    repo = workspace_root / "sympy__sympy-14248"
    repo.mkdir(parents=True)
    subprocess.run(["git", "init", "--quiet", str(repo)], check=True)
    (repo / "original.txt").write_text("original\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "initial"], cwd=repo, check=True)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    task = Task("sympy__sympy-14248", "sympy/sympy", sha,
                "swebench/sweb.eval.x86_64.sympy_1776_sympy-14248:latest",
                "Make the requested edit.", ["secret-reference-location.py"])
    output = tmp_path / "run"
    path = run([task], run_agent, workspace_root if environment == "local" else None, output,
               {"base_url": base_url, "model": "scripted-endpoint", "api_key": "integration-key"},
               "c0", "sweagent.agents.c0_agent:run_agent", environment)
    prediction = json.loads(path.read_text())
    assert "c0_integration.py" in prediction["model_patch"]
    assert "+value = 2" in prediction["model_patch"]
    usage = json.loads((output / task.instance_id / "usage.json").read_text())["usage"]
    assert usage["stop_reason"] == "finish"
    assert usage["model_requests"] == 2
    assert usage["bash_calls"] == 1
    assert usage["input_tokens"] == 200
    assert usage["output_tokens"] == 40
    assert len(received) == 2
    assert received[0][0] == "/v1/responses"
    assert all(request[2] == "Bearer integration-key" for request in received)
    for artifact in output.rglob("*"):
        if artifact.is_file():
            assert "integration-key" not in artifact.read_text(encoding="utf-8")
    assert received[0][1]["reasoning"] == {"effort": "medium"}
    history = received[1][1]["input"]
    assert any(item.get("encrypted_content") == "opaque-test-reasoning" for item in history)
    assert any(item["type"] == "function_call_output" for item in history if "type" in item)
    assert "secret-reference-location" not in json.dumps(received)
