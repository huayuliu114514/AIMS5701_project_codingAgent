"""The rules behind task selection, checked on diffs rather than on the network."""

from __future__ import annotations

import pytest

from sweagent.dataset import Task, is_source_file, patch_paths, select, source_files_touched

MIXED_DIFF = """diff --git a/astropy/io/fits/card.py b/astropy/io/fits/card.py
--- a/astropy/io/fits/card.py
+++ b/astropy/io/fits/card.py
@@ -1 +1 @@
-old
+new
diff --git a/astropy/io/fits/tests/test_card.py b/astropy/io/fits/tests/test_card.py
--- a/astropy/io/fits/tests/test_card.py
+++ b/astropy/io/fits/tests/test_card.py
@@ -1 +1 @@
-old
+new
"""


@pytest.mark.parametrize(
    "path",
    [
        "astropy/io/fits/tests/test_card.py",
        "tests/test_thing.py",
        "pkg/conftest.py",
        "pkg/testing/helpers.py",
    ],
)
def test_test_files_are_not_source(path: str) -> None:
    assert not is_source_file(path)


@pytest.mark.parametrize(
    "path",
    [
        # pytest's own implementation, not pytest's test suite
        "src/_pytest/python.py",
        "docs/conf.py",
        "examples/plot.py",
        "benchmarks/run.py",
        "README.md",
    ],
)
def test_non_source_files_are_excluded(path: str) -> None:
    expected_source = path == "src/_pytest/python.py"
    assert is_source_file(path) is expected_source


def test_patch_paths_are_deduplicated() -> None:
    # Each file appears in both a "diff --git" line and a "--- / +++" pair.
    assert patch_paths(MIXED_DIFF) == [
        "astropy/io/fits/card.py",
        "astropy/io/fits/tests/test_card.py",
    ]


def test_only_the_source_side_is_counted() -> None:
    assert source_files_touched(MIXED_DIFF) == ["astropy/io/fits/card.py"]


def _task(iid: str, repo: str, n_files: int, patch_bytes: int = 0) -> Task:
    return Task(
        instance_id=iid,
        repo=repo,
        base_commit="0" * 40,
        image=f"img/{iid}",
        problem_statement="",
        source_files=[f"pkg/m{i}.py" for i in range(n_files)],
    )


def test_selection_keeps_the_file_count_inside_the_profile() -> None:
    tasks = [
        _task("one", "a/a", 1),
        _task("two", "a/a", 2),
        _task("three", "b/b", 3),
        _task("four", "c/c", 4),
        _task("five", "c/c", 5),
    ]
    chosen = select(tasks, limit=3, min_repos=3)
    assert {t.instance_id for t in chosen} == {"two", "three", "four"}


def test_selection_spans_sizes_instead_of_piling_on_the_smallest() -> None:
    # Twelve two-file tasks and one of each larger size. Taking whatever comes
    # first would return only two-file tasks and hide any scaffold difference.
    tasks = [_task(f"s{i}", f"r{i}/r{i}", 2) for i in range(12)]
    tasks += [_task("m3", "x/x", 3), _task("m4", "y/y", 4)]
    chosen = select(tasks, limit=6, min_repos=3)
    assert {t.n_source_files for t in chosen} == {2, 3, 4}


def test_selection_fails_loudly_when_too_few_repos_survive() -> None:
    tasks = [_task(f"a{i}", "a/a", 2) for i in range(5)]
    with pytest.raises(ValueError, match="repositories"):
        select(tasks, min_repos=3, limit=5)


def test_selection_fails_loudly_when_repo_caps_cannot_fill_the_quota() -> None:
    # Three repositories pass min_repos, but at a cap of 2 they hold six slots
    # between them and the seventh task has nowhere to go. Returning six would
    # silently shrink the experiment's denominator.
    tasks = [_task(f"{r}{i}", f"{r}/{r}", 2) for r in "abc" for i in range(3)]
    with pytest.raises(ValueError, match="only 6 slots"):
        select(tasks, limit=7, min_repos=3, per_repo_cap=2)
