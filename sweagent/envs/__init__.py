"""The small execution contract shared by local and Docker workspaces."""
from dataclasses import dataclass


@dataclass
class CommandResult:
    output: str
    returncode: int
    timed_out: bool = False


PATCH_SCRIPT = """
set -e
index_dir=$(mktemp -d)
export GIT_INDEX_FILE="$index_dir/index"
trap 'rm -f "$GIT_INDEX_FILE"; rmdir "$index_dir"' EXIT
git read-tree HEAD
git -c core.fileMode=false add --all
git -c core.fileMode=false diff --cached --binary "$1" --
"""


def capture_patch(environment, start_commit: str) -> str:
    """Snapshot through a temporary index, even after an interrupted Git command."""
    return environment.shell(PATCH_SCRIPT, start_commit)
