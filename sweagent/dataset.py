# choose which tasks to run from SWE-bench Verified, and load/save them in a simple JSONL format
from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

DATASET = "SWE-bench/SWE-bench_Verified"
SPLIT = "test"

#检查路径用的，暂时不用去管他这个正则匹配，出问题再改
_TEST_DIR = re.compile(r"(^|/)(tests?|testing|spec|fixtures)(/|$)", re.IGNORECASE)
_TEST_FILE = re.compile(r"(^test_.*\.py$|.*_test\.py$|^conftest\.py$)", re.IGNORECASE)
# Trees a fix would not touch: documentation, runnable examples, helper scripts.
_NOT_SOURCE_DIR = re.compile(
    r"(^|/)(docs?|doc|examples?|samples?|scripts?|benchmarks?|bench)/", re.IGNORECASE
)


def patch_paths(patch: str) -> list[str]:
    """Every file a unified diff touches, de-duplicated, in order of appearance."""
    seen: dict[str, None] = {}
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                seen.setdefault(_strip_prefix(parts[3]), None)
        elif line.startswith(("+++ ", "--- ")):
            raw = line[4:].split("\t", 1)[0].strip()
            if raw and raw != "/dev/null":
                seen.setdefault(_strip_prefix(raw), None)
    return list(seen)

def _strip_prefix(path: str) -> str:
    return path[2:] if path[:2] in {"a/", "b/"} else path

def is_source_file(path: str) -> bool:
    """A Python source file that a fix would edit, as opposed to a test or doc."""
    cleaned = _strip_prefix(path)
    if not cleaned.endswith((".py", ".pyx", ".pyi")):
        return False
    if _TEST_DIR.search(cleaned) or _TEST_FILE.match(cleaned.rsplit("/", 1)[-1]):
        return False
    return not _NOT_SOURCE_DIR.search(cleaned)

def source_files_touched(patch: str) -> list[str]:
    """The measure the proposal's 'two to four files' rule is applied to."""
    return [p for p in patch_paths(patch) if is_source_file(p)]

@dataclass(frozen=True)
class Task:
    """One SWE-bench instance, carrying everything later stages need."""
    instance_id: str
    repo: str
    base_commit: str
    image: str            
    #the prebuilt container the agent works in
    problem_statement: str
    source_files: list[str] = field(default_factory=list)

    @property
    def n_source_files(self) -> int:
        return len(self.source_files)

    def to_dict(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "repo": self.repo,
            "base_commit": self.base_commit,
            "image": self.image,
            "problem_statement": self.problem_statement,
            "source_files": self.source_files,
            "n_source_files": self.n_source_files,
        }


def load_verified() -> list[Task]:
    """Download SWE-bench Verified (cached by `datasets` after the first run)."""
    from datasets import load_dataset

    raw = load_dataset(DATASET, split=SPLIT)
    return [
        # 针对官方文档的harness里面的类定义
        Task(
            instance_id=row["instance_id"],
            repo=row["repo"],
            base_commit=row["base_commit"],
            image=row["image"],
            problem_statement=row["problem_statement"],
            source_files=source_files_touched(row["patch"]),
        )
        for row in raw
    ]


def select(
    tasks: list[Task],
    limit: int = 20,
    min_files: int = 2,
    max_files: int = 4,
    min_repos: int = 3,
    per_repo_cap: int = 4,
) -> list[Task]:

    eligible = [t for t in tasks if min_files <= t.n_source_files <= max_files]
    if len({t.repo for t in eligible}) < min_repos:
        raise ValueError(f"need {min_repos} repositories, the filter left fewer")
    # Stratify so the set spans two, three, and four file fixes instead of
    # collapsing onto whichever size happens to be most common.
    by_size: dict[int, list[Task]] = {}
    for task in sorted(eligible, key=lambda t: (t.n_source_files, t.instance_id)):
        by_size.setdefault(task.n_source_files, []).append(task)
    ordered: list[Task] = []
    while any(by_size.values()):
        for size in sorted(by_size):
            if by_size[size]:
                ordered.append(by_size[size].pop(0))

    chosen: list[Task] = []
    taken: dict[str, int] = {}
    for task in ordered:
        if len(chosen) >= limit:
            break
        if taken.get(task.repo, 0) < per_repo_cap:
            chosen.append(task)
            taken[task.repo] = taken.get(task.repo, 0) + 1

    if len(chosen) < limit:
        raise ValueError(
            f"selected {len(chosen)} of {limit} tasks: {len(taken)} repositories "
            f"at a cap of {per_repo_cap} leave only {len(taken) * per_repo_cap} slots"
        )
    return chosen


def save(tasks: list[Task], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task.to_dict(), ensure_ascii=False) + "\n")


def load(path: Path) -> list[Task]:
    with path.open(encoding="utf-8") as handle:
        return [
            Task(
                instance_id=row["instance_id"],
                repo=row["repo"],
                base_commit=row["base_commit"],
                image=row["image"],
                problem_statement=row["problem_statement"],
                source_files=row["source_files"],
            )
            for row in map(json.loads, filter(str.strip, handle))
        ]
