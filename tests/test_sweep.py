"""Exercise the sweep with real temporary Git repos; grading itself stays official."""
from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from sweagent import sweep
from sweagent.dataset import DATASET, Task, save


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True, encoding="utf-8")


def commit(repo: Path) -> None:
    git(repo, "add", "--all")
    git(repo, "-c", "user.name=Sweep Test", "-c", "user.email=test@example.invalid",
        "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "test snapshot")


@pytest.fixture
def prepared(tmp_path):
    workspace_root = tmp_path / "workspaces"
    tasks = []
    pristine = {}
    for index in range(2):
        iid = f"owner__repo-{index}"
        repo = workspace_root / iid
        repo.mkdir(parents=True)
        git(repo, "init", "--quiet")
        (repo / ".gitattributes").write_text("* text eol=lf\n", encoding="utf-8")
        (repo / "module.py").write_text("value = 1\n", encoding="utf-8")
        (repo / "obsolete.py").write_text("old = True\n", encoding="utf-8")
        commit(repo)
        clone = tmp_path / f"pristine-{index}"
        subprocess.run(["git", "clone", "--quiet", str(repo), str(clone)], check=True)
        pristine[iid] = clone
        tasks.append(Task(iid, "owner/repo", git(repo, "rev-parse", "HEAD").strip(),
                          "unused-image", f"Fix issue {index}: café", ["secret_gold_location.py"]))
    return tasks, workspace_root, pristine


def test_runner_captures_replayable_patches_and_records_usage(tmp_path, prepared):
    tasks, workspace_root, pristine = prepared
    config = {"model": "test-model", "reasoning_effort": "medium",
              "command_timeout": 60, "max_observation_chars": 20000,
              "api_key": "private-run-key"}
    calls = []

    def agent(issue, environment, received_config, log_dir):
        repo_dir = Path(environment.workdir)
        calls.append(issue)
        assert received_config == config
        assert "secret_gold_location" not in issue
        (log_dir / "trajectory.json").write_text('[]\n', encoding="utf-8")
        (repo_dir / "module.py").write_text("value = 2\n", encoding="utf-8")
        # Even a commit made by the agent must be part of the final submission.
        commit(repo_dir)
        (repo_dir / "new.py").write_text("label = 'café'\n", encoding="utf-8")
        (repo_dir / "obsolete.py").unlink()
        return {"steps": 1, "prompt_tokens": 12, "completion_tokens": 8}

    run_dir = tmp_path / "run"
    predictions = sweep.run(tasks, agent, workspace_root, run_dir, config, "c0", "test:agent", "local")
    rows = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines()]
    assert calls == [task.problem_statement for task in tasks]
    assert [row["instance_id"] for row in rows] == [task.instance_id for task in tasks]
    for row, task in zip(rows, tasks):
        assert set(row) == {"instance_id", "model_name_or_path", "model_patch"}
        assert row["model_name_or_path"] == "test-model"
        patch = tmp_path / f"{task.instance_id}.patch"
        patch.write_text(row["model_patch"], encoding="utf-8")
        git(pristine[task.instance_id], "apply", str(patch))
        assert (pristine[task.instance_id] / "module.py").read_text() == "value = 2\n"
        assert (pristine[task.instance_id] / "new.py").read_text(encoding="utf-8") == "label = 'café'\n"
        assert not (pristine[task.instance_id] / "obsolete.py").exists()
        usage = json.loads((run_dir / task.instance_id / "usage.json").read_text())
        assert usage["start_commit"] == task.base_commit
        assert usage["usage"]["prompt_tokens"] == 12
        assert usage["elapsed_seconds"] >= 0
    metadata = json.loads((run_dir / "run.json").read_text())
    assert metadata["instance_ids"] == [task.instance_id for task in tasks]
    assert metadata["config"] == {name: value for name, value in config.items() if name != "api_key"}
    assert "private-run-key" not in (run_dir / "run.json").read_text()
    assert config["api_key"] == "private-run-key"
    assert metadata["scaffold"] == "c0"


def test_runner_leaves_completed_rows_when_agent_raises(tmp_path, prepared):
    tasks, workspace_root, _ = prepared

    def agent(issue, repo_dir, config, log_dir):
        if issue == tasks[1].problem_statement:
            raise RuntimeError("agent failure")
        return {"steps": 0}

    run_dir = tmp_path / "partial"
    with pytest.raises(RuntimeError, match="agent failure"):
        sweep.run(tasks, agent, workspace_root, run_dir, {"model": "test"}, "c0", "test:agent", "local")
    rows = [json.loads(line) for line in (run_dir / "predictions.jsonl").read_text().splitlines()]
    assert [row["instance_id"] for row in rows] == [tasks[0].instance_id]
    assert rows[0]["model_patch"] == ""


@pytest.mark.parametrize("predictions", ["gold", "predictions.jsonl"])
def test_grade_delegates_selected_ids_and_preserves_harness_failures(tmp_path, monkeypatch, predictions):
    task = Task("sympy__sympy-14248", "sympy/sympy", "base", "image", "issue")
    recorded = {}

    def launch(command, *, cwd, check):
        recorded.update(command=command, cwd=cwd, check=check)
        raise subprocess.CalledProcessError(7, command)

    monkeypatch.setattr(sweep.subprocess, "run", launch)
    output_dir = tmp_path / "grade"
    with pytest.raises(subprocess.CalledProcessError) as failure:
        sweep.grade([task], predictions, "test-grade", output_dir, max_workers=2, timeout=90)
    assert failure.value.returncode == 7
    command = recorded["command"]
    assert command[:3] == [sys.executable, "-m", "swebench.harness.run_evaluation"]
    assert command[command.index("--dataset_name") + 1] == DATASET
    assert command[command.index("--instance_ids") + 1:command.index("--max_workers")] == [task.instance_id]
    assert command[command.index("--max_workers") + 1] == "2"
    assert command[command.index("--timeout") + 1] == "90"
    expected_path = "gold" if predictions == "gold" else str(Path(predictions).resolve())
    assert command[command.index("--predictions_path") + 1] == expected_path
    assert recorded["cwd"] == output_dir
    assert recorded["check"] is True
    assert json.loads((output_dir / "grade.json").read_text())["instance_ids"] == [task.instance_id]


def test_run_cli_loads_callable_and_filters_frozen_order(tmp_path, prepared, monkeypatch):
    tasks, workspace_root, _ = prepared
    tasks_path = tmp_path / "tasks.jsonl"
    save(tasks, tasks_path)
    config_path = tmp_path / "config.json"
    config_path.write_text('{"model": "cli-model"}', encoding="utf-8")
    module = types.ModuleType("sweep_test_agent")
    seen = []

    def agent(issue, environment, config, log_dir):
        seen.append((issue, Path(environment.workdir)))
        return {"steps": 0}

    module.run_agent = agent
    monkeypatch.setitem(sys.modules, module.__name__, module)
    output_dir = tmp_path / "cli-run"
    sweep.main([
        "run", "--tasks", str(tasks_path), "--instance-ids", tasks[1].instance_id,
        "--run-id", "cli-run", "--output-dir", str(output_dir),
        "--agent", "sweep_test_agent:run_agent", "--scaffold", "c0",
        "--config", str(config_path), "--environment", "local", "--workspace-root", str(workspace_root),
    ])
    assert seen == [(tasks[1].problem_statement, workspace_root / tasks[1].instance_id)]
    assert json.loads((output_dir / "run.json").read_text())["instance_ids"] == [tasks[1].instance_id]


def test_grade_cli_uses_selected_ids_and_default_output(tmp_path, prepared, monkeypatch):
    tasks, _, _ = prepared
    tasks_path = tmp_path / "tasks.jsonl"
    save(tasks, tasks_path)
    monkeypatch.setattr(sweep, "ROOT", tmp_path)
    calls = []
    monkeypatch.setattr(sweep, "grade", lambda *args: calls.append(args))
    sweep.main([
        "grade", "--tasks", str(tasks_path), "--instance-ids", tasks[0].instance_id,
        "--run-id", "gold-check", "--predictions", "gold",
    ])
    assert calls == [([tasks[0]], "gold", "gold-check", tmp_path / "runs" / "grading" / "gold-check", 1, 1800)]


@pytest.mark.parametrize("agent_catches_interrupt", [False, True])
def test_interrupt_saves_current_patch_and_stops_sweep(tmp_path, prepared, agent_catches_interrupt):
    tasks, workspace_root, _ = prepared
    calls = []

    def agent(issue, environment, config, log_dir):
        calls.append(issue)
        environment.execute("printf interrupted > interruption.txt")
        if agent_catches_interrupt:
            return {"stop_reason": "interrupted", "model_requests": 2}
        raise KeyboardInterrupt

    run_dir = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        sweep.run(tasks, agent, workspace_root, run_dir, {"model": "test"}, "c0", "test:agent", "local")
    rows = [json.loads(line) for line in (run_dir / "predictions.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert "interruption.txt" in rows[0]["model_patch"]
    assert calls == [tasks[0].problem_statement]
    usage = json.loads((run_dir / tasks[0].instance_id / "usage.json").read_text())
    assert usage["usage"]["stop_reason"] == "interrupted"


def test_runner_context_stop_submits_then_continues(tmp_path, prepared):
    tasks, workspace_root, _ = prepared

    def agent(issue, environment, config, log_dir):
        return {"stop_reason": "context_limit"}

    predictions = sweep.run(tasks, agent, workspace_root, tmp_path / "context", {"model": "test"},
                            "c0", "test:agent", "local")
    assert len(predictions.read_text().splitlines()) == 2


def test_docker_selection_and_cleanup_on_agent_failure(tmp_path, prepared, monkeypatch):
    tasks, _, _ = prepared
    created = []
    closed = []

    class FakeDocker:
        workdir = "/testbed"

        def __init__(self, image, command_timeout):
            created.append((image, command_timeout))

        def git(self, *args):
            return "image-head\n"

        def close(self):
            closed.append(True)

    monkeypatch.setattr("sweagent.envs.docker_env.DockerEnv", FakeDocker)

    def agent(issue, environment, config, log_dir):
        assert environment.workdir == "/testbed"
        raise RuntimeError("model unavailable")

    with pytest.raises(RuntimeError, match="model unavailable"):
        sweep.run(tasks, agent, None, tmp_path / "docker", {"model": "test", "command_timeout": 5},
                  "c0", "test:agent")
    assert created == [(tasks[0].image, 5)]
    assert closed == [True]
    assert (tmp_path / "docker" / "predictions.jsonl").read_text() == ""


@pytest.mark.parametrize("stop_reason", ["finish", "interrupted"])
def test_snapshot_survives_stale_git_lock_without_touching_real_index(tmp_path, prepared, stop_reason):
    tasks, workspace_root, _ = prepared
    repo = workspace_root / tasks[0].instance_id
    original_index = (repo / ".git" / "index").read_bytes()

    def agent(issue, environment, config, log_dir):
        environment.execute("printf stale > .git/index.lock; printf saved > locked_edit.txt")
        return {"stop_reason": stop_reason}

    output = tmp_path / "stale-lock"
    if stop_reason == "interrupted":
        with pytest.raises(KeyboardInterrupt):
            sweep.run(tasks[:1], agent, workspace_root, output, {"model": "test"}, "c0", "test:agent", "local")
    else:
        sweep.run(tasks[:1], agent, workspace_root, output, {"model": "test"}, "c0", "test:agent", "local")
    assert "locked_edit.txt" in json.loads((output / "predictions.jsonl").read_text())["model_patch"]
    assert (repo / ".git" / "index").read_bytes() == original_index
    assert (repo / ".git" / "index.lock").exists()


def test_cli_defaults_to_c0_and_docker(tmp_path, prepared, monkeypatch):
    from sweagent.agents.c0_agent import run_agent
    tasks, _, _ = prepared
    tasks_path = tmp_path / "tasks.jsonl"
    save(tasks, tasks_path)
    config_path = tmp_path / "config.json"
    config_path.write_text('{"model": "test", "base_url": "http://endpoint.invalid/v1"}')
    calls = []
    monkeypatch.setattr(sweep, "run", lambda *args: calls.append(args) or tmp_path / "predictions.jsonl")
    sweep.main(["run", "--tasks", str(tasks_path), "--config", str(config_path),
                "--run-id", "defaults", "--output-dir", str(tmp_path / "run")])
    arguments = calls[0]
    assert arguments[1] is run_agent
    assert arguments[2] is None
    assert arguments[5:] == ("c0", "sweagent.agents.c0_agent:run_agent", "docker")
