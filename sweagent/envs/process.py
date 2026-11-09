"""Linux command supervision, also executed inside the prepared containers."""

# A separate subreaper owns orphaned descendants, including programs using setsid.
# This keeps an escaped child's inherited output pipe from defeating the timeout.
SUPERVISOR = r"""
import ctypes, json, os, signal, subprocess, sys
from pathlib import Path

ctypes.CDLL(None).prctl(36, 1, 0, 0, 0)  # PR_SET_CHILD_SUBREAPER
command, timeout, setup = sys.argv[1], float(sys.argv[2]), sys.argv[3]
process = subprocess.Popen(['/bin/bash', '-c', setup + command],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)

def cancel_children():
    children_file = Path('/proc/{0}/task/{0}/children'.format(os.getpid()))
    while True:
        children = children_file.read_text().split()
        if not children:
            break
        for child in children:
            try:
                os.kill(int(child), signal.SIGKILL)
            except ProcessLookupError:
                pass
        child, status = os.waitpid(-1, 0)
        if child == process.pid:
            process.returncode = os.waitstatus_to_exitcode(status)

timed_out = False
try:
    output, _ = process.communicate(timeout=timeout)
except subprocess.TimeoutExpired:
    timed_out = True
    cancel_children()
    output, _ = process.communicate()
except KeyboardInterrupt:
    cancel_children()
    output, _ = process.communicate()
print(json.dumps({'output': output.decode('utf-8', errors='replace'),
    'returncode': process.returncode, 'timed_out': timed_out}))
"""
