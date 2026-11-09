"""Real Linux shell behavior, including cancellation of a command's children."""
import subprocess
import time
from pathlib import Path

import pytest

from sweagent.envs.local_env import LocalEnv


def test_fresh_shell_preserves_files_but_not_directory_or_exports(tmp_path):
    env = LocalEnv(tmp_path)
    first = env.execute("mkdir sub; cd sub; export C0_PRIVATE_FLAG=present; printf data > file.txt")
    assert first.returncode == 0
    assert (tmp_path / "sub" / "file.txt").read_text() == "data"
    second = env.execute('pwd; printf "flag=%s\\n" "${C0_PRIVATE_FLAG-unset}"; printf error >&2; exit 3')
    assert second.returncode == 3
    assert str(tmp_path) in second.output
    assert "flag=unset" in second.output
    assert "error" in second.output
    assert not second.timed_out
    env.close()
    assert tmp_path.exists()


def test_timeout_keeps_partial_output_and_kills_background_children(tmp_path):
    env = LocalEnv(tmp_path, command_timeout=0.2)
    started = time.monotonic()
    result = env.execute("printf partial; sleep 30 & echo $! > child.pid; wait")
    assert time.monotonic() - started < 5
    assert result.timed_out
    assert result.returncode != 0
    assert result.output == "partial"
    pid = (tmp_path / "child.pid").read_text().strip()
    stat = Path(f"/proc/{pid}/stat")
    assert not stat.exists() or stat.read_text().split()[2] == "Z"
    assert env.execute("printf next").output == "next"


def test_timeout_reaps_detached_orphans_after_parent_shell_exits(tmp_path):
    env = LocalEnv(tmp_path, command_timeout=0.2)
    started = time.monotonic()
    result = env.execute("printf partial; setsid sleep 30 & echo $! > detached.pid")
    assert result.timed_out
    assert result.output == "partial"
    assert time.monotonic() - started < 5
    pid = (tmp_path / "detached.pid").read_text().strip()
    assert not Path(f"/proc/{pid}").exists()
    assert env.execute("printf after").output == "after"


def test_keyboard_interrupt_kills_command_group_before_propagating(tmp_path, monkeypatch):
    actual = subprocess.Popen.communicate
    processes = []

    def interrupt_once(process, *args, **kwargs):
        if not processes:
            processes.append(process)
            raise KeyboardInterrupt
        return actual(process, *args, **kwargs)

    monkeypatch.setattr(subprocess.Popen, "communicate", interrupt_once)
    with pytest.raises(KeyboardInterrupt):
        LocalEnv(tmp_path).execute("sleep 30")
    assert processes[0].poll() is not None
