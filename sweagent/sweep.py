"""Run an agent in prepared checkouts, or grade saved patches with SWE-bench."""
from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from sweagent.dataset import DATASET, SPLIT, Task, load
from sweagent.envs import capture_patch
from sweagent.envs.local_env import LocalEnv
from sweagent.model import logged_config


def create_environment(kind: str, task: Task, workspace_root: Path | None, config: dict):
    if kind == "local":
        return LocalEnv(workspace_root / task.instance_id, config["command_timeout"])
    from sweagent.envs.docker_env import DockerEnv
    return DockerEnv(task.image, config["command_timeout"])

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASKS = ROOT / "data" / "tasks.jsonl"


def git(repo_dir: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=repo_dir, text=True, encoding="utf-8"
    )


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(
    tasks: list[Task],
    agent: Callable[..., dict],
    workspace_root: Path | None,
    run_dir: Path,
    config: dict,
    scaffold: str,
    agent_name: str,
    environment: str = "docker",
) -> Path:
    """Call agent(issue, environment, config, log_dir) and capture its submission.

    Local mode uses prepared checkouts; Docker mode starts fresh task containers.
    Returned usage must be JSON-serializable. Config may include api_key;
    that field is omitted from recorded metadata.
    """
    workspace_root = workspace_root.resolve() if workspace_root is not None else None
    config = {"reasoning_effort": "medium", "command_timeout": 60,
              "max_observation_chars": 20000, **config}
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True)
    write_json(run_dir / "run.json", {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": DATASET,
        "split": SPLIT,
        "instance_ids": [task.instance_id for task in tasks],
        "scaffold": scaffold,
        "agent": agent_name,
        "config": logged_config(config),
        "environment": environment,
        "workspace_root": str(workspace_root) if workspace_root is not None else None,
        "project_commit": git(ROOT, "rev-parse", "HEAD").strip(),
        "project_dirty": bool(git(ROOT, "status", "--porcelain")),
    })

    predictions_path = run_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8", newline="\n") as predictions:
        for index, task in enumerate(tasks, 1):
            print(f"[{index}/{len(tasks)}] {task.instance_id}", flush=True)
            log_dir = run_dir / task.instance_id
            log_dir.mkdir()
            env = create_environment(environment, task, workspace_root, config)
            try:
                start_commit = env.git("rev-parse", "HEAD").strip()
                write_json(log_dir / "environment.json", {
                    "kind": environment,
                    "workdir": env.workdir,
                    "image": task.image if environment == "docker" else None,
                    "start_commit": start_commit,
                })
                started = time.monotonic()
                try:
                    usage = agent(task.problem_statement, env, config, log_dir)
                except KeyboardInterrupt:
                    usage = {"stop_reason": "interrupted"}
                elapsed = time.monotonic() - started
                patch = capture_patch(env, start_commit)
                write_json(log_dir / "usage.json", {
                    "instance_id": task.instance_id,
                    "start_commit": start_commit,
                    "elapsed_seconds": elapsed,
                    "usage": usage,
                })
                predictions.write(json.dumps({
                    "instance_id": task.instance_id,
                    "model_name_or_path": config["model"],
                    "model_patch": patch,
                }, ensure_ascii=False) + "\n")
                predictions.flush()
                if usage.get("stop_reason") == "interrupted":
                    raise KeyboardInterrupt
            finally:
                env.close()
    return predictions_path


def grade(
    tasks: list[Task],
    predictions_path: str,
    run_id: str,
    output_dir: Path,
    max_workers: int = 1,
    timeout: int = 1800,
) -> None:
    """Delegate patch application, test execution, and reports to the official harness."""
    output_dir = output_dir.resolve()
    predictions_path = (
        "gold" if predictions_path == "gold" else str(Path(predictions_path).resolve())
    )
    command = [
        sys.executable, "-m", "swebench.harness.run_evaluation",
        "--dataset_name", DATASET,
        "--split", SPLIT,
        "--predictions_path", predictions_path,
        "--instance_ids", *[task.instance_id for task in tasks],
        "--max_workers", str(max_workers),
        "--run_id", run_id,
        "--timeout", str(timeout),
    ]
    output_dir.mkdir(parents=True)
    write_json(output_dir / "grade.json", {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "instance_ids": [task.instance_id for task in tasks],
    })
    # Inherit output; the harness prints counts and writes its own detailed logs.
    subprocess.run(command, cwd=output_dir, check=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="Run an agent locally or in SWE-bench containers")
    grade_parser = commands.add_parser("grade", help="Grade saved predictions (or gold)")
    for child in (run_parser, grade_parser):
        child.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
        child.add_argument("--instance-ids", nargs="+")
        child.add_argument("--run-id", required=True)
        child.add_argument("--output-dir", type=Path)

    run_parser.add_argument("--agent", default="sweagent.agents.c0_agent:run_agent",
                            help="Python callable: module:function")
    run_parser.add_argument("--scaffold", default="c0", help="Experiment label")
    run_parser.add_argument("--config", type=Path, required=True, help="JSON model and execution config")
    run_parser.add_argument("--environment", choices=["local", "docker"], default="docker")
    run_parser.add_argument("--workspace-root", type=Path, help="Prepared checkouts; required in local mode")
    grade_parser.add_argument("--predictions", required=True, help="Predictions JSONL path, or gold")
    grade_parser.add_argument("--max-workers", type=int, default=1)
    grade_parser.add_argument("--timeout", type=int, default=1800, help="Grading seconds per task")
    args = parser.parse_args(argv)

    tasks = load(args.tasks)
    if args.instance_ids is not None:
        tasks = [task for task in tasks if task.instance_id in args.instance_ids]

    if args.command == "run":
        module_name, function_name = args.agent.split(":")
        agent = getattr(importlib.import_module(module_name), function_name)
        config = json.loads(args.config.read_text(encoding="utf-8"))
        output_dir = args.output_dir or ROOT / "runs" / args.run_id
        predictions = run(
            tasks, agent, args.workspace_root, output_dir, config, args.scaffold, args.agent,
            args.environment,
        )
        print(f"Predictions: {predictions}")
    else:
        output_dir = args.output_dir or ROOT / "runs" / "grading" / args.run_id
        grade(tasks, args.predictions, args.run_id, output_dir, args.max_workers, args.timeout)


if __name__ == "__main__":
    main()
