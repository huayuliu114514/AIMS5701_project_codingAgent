"""Fresh Bash commands in a prepared Linux checkout."""
from __future__ import annotations

import json
import signal
import subprocess
import sys
from pathlib import Path

from sweagent.envs import CommandResult
from sweagent.envs.process import SUPERVISOR


class LocalEnv:
    def __init__(self, repo_dir: Path, command_timeout: float = 60):
        self.workdir = str(repo_dir.resolve())
        self.command_timeout = command_timeout

    def execute(self, command: str) -> CommandResult:
        process = subprocess.Popen(
            [sys.executable, "-c", SUPERVISOR, command, str(self.command_timeout), ""],
            cwd=self.workdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            output, _ = process.communicate()
        except KeyboardInterrupt:
            try:
                process.send_signal(signal.SIGINT)
            except ProcessLookupError:
                pass
            process.communicate()
            raise
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, "command supervisor", output)
        return CommandResult(**json.loads(output))

    def git(self, *args: str) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=self.workdir, text=True, encoding="utf-8"
        )

    def shell(self, script: str, *args: str) -> str:
        return subprocess.check_output(
            ["/bin/bash", "-c", script, "snapshot", *args],
            cwd=self.workdir, text=True, encoding="utf-8",
        )

    def close(self) -> None:
        pass  # Local task checkouts intentionally remain available for inspection.
