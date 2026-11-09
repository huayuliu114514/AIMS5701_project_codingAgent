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
    workspace_root: Path,
    run_dir: Path,
    config: dict,
    scaffold: str,
    agent_name: str,
) -> Path:
    """Call agent(issue, repo_dir, config, log_dir); it edits files and returns usage.

    The returned usage dictionary must be JSON-serializable.
    Workspaces are fresh, prepared Git checkouts at workspace_root/instance_id.
    The agent writes its trajectory in log_dir. Config contains a model label
    under 'model'; credentials belong in environment variables.
    """
    workspace_root = workspace_root.resolve()
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True)
    write_json(run_dir / "run.json", {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": DATASET,
        "split": SPLIT,
        "instance_ids": [task.instance_id for task in tasks],
        "scaffold": scaffold,
        "agent": agent_name,
        "config": config,
        "workspace_root": str(workspace_root),
        "project_commit": git(ROOT, "rev-parse", "HEAD").strip(),
        "project_dirty": bool(git(ROOT, "status", "--porcelain")),
    })

    predictions_path = run_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8", newline="\n") as predictions:
        for index, task in enumerate(tasks, 1):
            print(f"[{index}/{len(tasks)}] {task.instance_id}", flush=True)
            repo_dir = workspace_root / task.instance_id
            start_commit = git(repo_dir, "rev-parse", "HEAD").strip()
            log_dir = run_dir / task.instance_id
            log_dir.mkdir()
            started = time.monotonic()
            usage = agent(task.problem_statement, repo_dir, config, log_dir)
            elapsed = time.monotonic() - started

            # Include non-ignored new files, staged edits, and agent commits.
            git(repo_dir, "add", "--intent-to-add", "--all")
            patch = git(repo_dir, "-c", "core.fileMode=false", "diff", "--binary", start_commit, "--")
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
    run_parser = commands.add_parser("run", help="Run an agent in prepared workspaces")
    grade_parser = commands.add_parser("grade", help="Grade saved predictions (or gold)")
    for child in (run_parser, grade_parser):
        child.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
        child.add_argument("--instance-ids", nargs="+")
        child.add_argument("--run-id", required=True)
        child.add_argument("--output-dir", type=Path)

    run_parser.add_argument("--agent", required=True, help="Python callable: module:function")
    run_parser.add_argument("--scaffold", required=True, help="Experiment label, e.g. c0")
    run_parser.add_argument("--config", type=Path, required=True, help="JSON agent config with 'model'")
    run_parser.add_argument("--workspace-root", type=Path, required=True)
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
            tasks, agent, args.workspace_root, output_dir, config, args.scaffold, args.agent
        )
        print(f"Predictions: {predictions}")
    else:
        output_dir = args.output_dir or ROOT / "runs" / "grading" / args.run_id
        grade(tasks, args.predictions, args.run_id, output_dir, args.max_workers, args.timeout)


if __name__ == "__main__":
    main()
