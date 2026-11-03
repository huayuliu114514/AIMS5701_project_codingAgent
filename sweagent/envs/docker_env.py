from __future__ import annotations
import io
import json
import os
import shutil
import subprocess
import tarfile
import threading
import time
from dataclasses import dataclass
from typing import Optional

import docker
from docker.errors import APIError, ImageNotFound
from docker.models.containers import Container


@dataclass
class CommandResult:
    output: str
    returncode: int
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class DockerEnvError(RuntimeError):
    pass


def docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        subprocess.run(
            ["docker", "info"], check=True, capture_output=True, timeout=20
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False
    return True

