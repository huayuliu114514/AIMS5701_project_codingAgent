"""Deliver a real SIGINT while Bash is running, then inspect the saved submission."""
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import docker
import pytest

WORKER = r'''
import json, sys
from pathlib import Path
from sweagent.dataset import Task
from sweagent.sweep import run
kind, workspace, output, ready = sys.argv[1:]
image = "swebench/sweb.eval.x86_64.sympy_1776_sympy-14248:latest"
tasks = [Task(f"owner__repo-{i}", "owner/repo", "unused", image, "issue") for i in range(2)]
def agent(issue, environment, config, log_dir):
    details = {"container_id": environment.container.id if kind == "docker" else None}
    (Path(output) / "active.json").write_text(json.dumps(details))
    environment.execute("printf preserved > interrupted_edit.txt; sleep 30 & touch " + ready + "; wait")
    return {"stop_reason": "finish"}
run(tasks, agent, Path(workspace), Path(output), {"model": "interruption-test"}, "test", "test:agent", kind)
'''


@pytest.mark.parametrize("kind", ["local", "docker"])
def test_actual_sigint_saves_patch_and_stops_everything(tmp_path, kind):
    if kind == "docker" and os.environ.get("RUN_DOCKER_TESTS") != "1":
        pytest.skip("Set RUN_DOCKER_TESTS=1 for the actual container interruption check")
    workspace = tmp_path / "workspaces"
    if kind == "local":
        for i in range(2):
            repo = workspace / f"owner__repo-{i}"
            repo.mkdir(parents=True)
            subprocess.run(["git", "init", "--quiet", str(repo)], check=True)
            (repo / "initial.txt").write_text("initial\n")
            subprocess.run(["git", "add", "."], cwd=repo, check=True)
            subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                            "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "initial"],
                           cwd=repo, check=True)
    output = tmp_path / "output"
    ready = "/tmp/c0-command-ready" if kind == "docker" else str(tmp_path / "ready")
    client = docker.from_env() if kind == "docker" else None
    container = None
    with (tmp_path / "worker.log").open("w+") as log:
        process = subprocess.Popen(
            [sys.executable, "-c", WORKER, kind, str(workspace), str(output), shlex.quote(ready)],
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 15
            active = output / "active.json"
            while time.monotonic() < deadline:
                if active.exists():
                    if kind == "docker":
                        if container is None:
                            container = client.containers.get(json.loads(active.read_text())["container_id"])
                        if container.exec_run(["test", "-f", ready]).exit_code == 0:
                            break
                    elif Path(ready).exists():
                        break
                assert process.poll() is None, "worker exited before its command started"
                time.sleep(0.05)
            else:
                pytest.fail("Bash did not reach its ready marker")
            os.kill(process.pid, signal.SIGINT)
            process.wait(timeout=15)
            assert process.returncode in (-signal.SIGINT, 130)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            if client is not None:
                if container is not None and process.returncode not in (-signal.SIGINT, 130):
                    try:
                        container.remove(force=True)
                    except docker.errors.NotFound:
                        pass
                client.close()
    rows = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["instance_id"] == "owner__repo-0"
    assert "interrupted_edit.txt" in rows[0]["model_patch"]
    usage = json.loads((output / "owner__repo-0" / "usage.json").read_text())
    assert usage["usage"]["stop_reason"] == "interrupted"
    assert not (output / "owner__repo-1").exists()
    if container is not None:
        check = docker.from_env()
        try:
            with pytest.raises(docker.errors.NotFound):
                check.containers.get(container.id)
        finally:
            check.close()
