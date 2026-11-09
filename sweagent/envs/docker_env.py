"""Execute actions in a fresh prepared SWE-bench instance container."""
from __future__ import annotations

import json
import subprocess

import docker

from sweagent.envs import CommandResult
from sweagent.envs.process import SUPERVISOR

SETUP = "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed && "


class DockerEnv:
    def __init__(self, image: str, command_timeout: float = 60):
        self.workdir = "/testbed"
        self.command_timeout = command_timeout
        self.client = docker.from_env(timeout=None)
        self.container = self.client.containers.run(
            image, ["/bin/bash", "-c", "sleep infinity"],
            detach=True, working_dir=self.workdir,
            labels={"swe-scaffold-study": "agent"},
        )

    def execute(self, command: str) -> CommandResult:
        try:
            result = self.container.exec_run(
                ["/opt/miniconda3/bin/python", "-c", SUPERVISOR, command, str(self.command_timeout), SETUP],
                workdir=self.workdir,
            )
        except KeyboardInterrupt:
            # Stop every in-container process, then reopen the same filesystem
            # so the runner can capture the interrupted task's patch.
            self.container.kill()
            self.container.start()
            raise
        if result.exit_code != 0:
            raise subprocess.CalledProcessError(result.exit_code, "container supervisor", result.output)
        return CommandResult(**json.loads(result.output))

    def git(self, *args: str) -> str:
        result = self.container.exec_run(["git", *args], workdir=self.workdir)
        if result.exit_code != 0:
            raise subprocess.CalledProcessError(result.exit_code, ["git", *args], result.output)
        return result.output.decode("utf-8")

    def shell(self, script: str, *args: str) -> str:
        result = self.container.exec_run(
            ["/bin/bash", "-c", script, "snapshot", *args], workdir=self.workdir
        )
        if result.exit_code != 0:
            raise subprocess.CalledProcessError(result.exit_code, "workspace snapshot", result.output)
        return result.output.decode("utf-8")

    def close(self) -> None:
        self.container.remove(force=True)
        self.client.close()
