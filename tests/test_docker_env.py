"""Opt-in integration checks against an actual prepared SWE-bench image."""
import os

import docker
import pytest

from sweagent.envs import capture_patch
from sweagent.envs.docker_env import DockerEnv

IMAGE = "swebench/sweb.eval.x86_64.sympy_1776_sympy-14248:latest"
pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_DOCKER_TESTS") != "1",
    reason="Set RUN_DOCKER_TESTS=1 to use Docker Desktop and the prepared task image",
)


@pytest.fixture
def environment():
    env = DockerEnv(IMAGE)
    try:
        yield env
    finally:
        env.close()


def test_image_environment_fresh_shell_patch_and_cleanup():
    env = DockerEnv(IMAGE)
    container_id = env.container.id
    try:
        start = env.git("rev-parse", "HEAD").strip()
        result = env.execute("python -c 'import sys, sympy; print(sys.prefix); print(sympy.__file__)'")
        assert result.returncode == 0
        assert "/envs/testbed" in result.output
        assert "/testbed/sympy/" in result.output
        assert env.execute("mkdir -p /tmp/c0-dir; cd /tmp/c0-dir; export C0_FLAG=1").returncode == 0
        fresh = env.execute('pwd; printf "flag=%s\\n" "${C0_FLAG-unset}"')
        assert fresh.output == "/testbed\nflag=unset\n"
        env.execute("printf 'value = 2\\n' > c0_smoke_new.py")
        patch = capture_patch(env, start)
        assert "c0_smoke_new.py" in patch
        assert "+value = 2" in patch
    finally:
        env.close()
    client = docker.from_env()
    try:
        with pytest.raises(docker.errors.NotFound):
            client.containers.get(container_id)
    finally:
        client.close()


def test_timeout_stops_in_container_children_and_keeps_files(environment):
    environment.command_timeout = 2
    result = environment.execute("printf partial; sleep 30 & echo $! > /tmp/c0-child.pid; wait")
    assert result.timed_out
    assert result.output == "partial"
    assert result.returncode != 0
    check = environment.container.exec_run([
        "/bin/bash", "-c",
        'pid=$(cat /tmp/c0-child.pid); test ! -e /proc/$pid/stat || test "$(awk \'{print $3}\' /proc/$pid/stat)" = Z',
    ])
    assert check.exit_code == 0
    assert environment.execute("printf after").output == "after"


def test_timeout_reaps_detached_orphans(environment):
    environment.command_timeout = 2
    result = environment.execute("printf detached; setsid sleep 30 & echo $! > /tmp/c0-detached.pid")
    assert result.timed_out
    assert result.output == "detached"
    assert environment.container.exec_run([
        "/bin/bash", "-c", 'test ! -d /proc/$(cat /tmp/c0-detached.pid)',
    ]).exit_code == 0


def test_snapshot_ignores_stale_index_lock_and_preserves_real_index(environment):
    start = environment.git("rev-parse", "HEAD").strip()
    environment.execute("printf stale > .git/index.lock; printf captured > c0_lock_test.txt")
    patch = capture_patch(environment, start)
    assert "c0_lock_test.txt" in patch
    assert environment.container.exec_run(["test", "-f", "/testbed/.git/index.lock"]).exit_code == 0
    assert environment.git("diff", "--cached", "--name-only") == ""


def test_interrupt_restarts_same_filesystem_for_patch_capture(environment, monkeypatch):
    environment.execute("printf before > c0_interrupted.txt")
    original = environment.container.exec_run

    def interrupt(*args, **kwargs):
        monkeypatch.setattr(environment.container, "exec_run", original)
        raise KeyboardInterrupt

    monkeypatch.setattr(environment.container, "exec_run", interrupt)
    with pytest.raises(KeyboardInterrupt):
        environment.execute("sleep 30")
    assert environment.execute("cat c0_interrupted.txt").output == "before"
    assert "c0_interrupted.txt" in capture_patch(environment, environment.git("rev-parse", "HEAD").strip())
