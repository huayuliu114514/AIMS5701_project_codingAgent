# Agent Scaffolding Study

Comparing agent scaffoldings for small coding models on repository maintenance
tasks — three scaffolds, one model, one budget.

| Config | Scaffold | What it adds |
|--------|----------|--------------|
| **C0** | baseline | minimal, bash-only, linear history |
| **C1** | full agent | repo map, symbol retrieval, planning, incremental patching, test-driven recovery |
| **C2** | no-map | C1 minus automatic repository maps and symbol rankings |

**RQ1** — under the same model and resource limits, how does model performance
vary across differently designed scaffolds?
**RQ2** — how does scaffolding affect resolution, reasoning, and inference cost?

Measurement is taken from the official SWE-bench harness rather than
reimplemented, so resolution numbers stay comparable with published results.

## Status

Early. The evaluation chain is validated end to end; the scaffolds are not yet
built.

- [x] Task selection with the proposal's filter (2–4 non-test source files, ≥3 repos)
- [x] Verify official grading on one instance (gold patch resolves)
- [ ] Container execution layer (`sweagent/envs/docker_env.py`, in progress)
- [ ] C0 baseline scaffold
- [ ] Repository mapping + symbol retrieval (C1, C2 differ here)
- [ ] Planning, incremental patching, test-driven recovery
- [ ] Full 20-task run across C0 / C1 / C2

## Setup

```bash
conda create -y -p ./env python=3.11
./env/bin/pip install -e ".[dev]"
```

`swebench` needs Python ≥3.10. Docker must be running: the evaluator pulls
`swebench/sweb.eval.x86_64.*` images (~2–3 GB each) and runs the hidden tests
inside them.

## Pipeline

```
HuggingFace (SWE-bench Verified, 500 instances)
      │  load_verified()
      ▼
list[Task] (500)          all instances, each with its source-file list
      │  select()           size filter → stratification → per-repo cap
      ▼
list[Task] (20)           the frozen experiment set
      │  save()
      ▼
data/tasks.jsonl          committed: the record of what a result was measured on
      │
      ▼
   [agent]  ──→  predictions.jsonl  ──→  swebench.harness.run_evaluation
```

## Layout

```
sweagent/
  dataset.py           task selection and the files/repos filter
  envs/docker_env.py   container lifecycle, exec, patch capture  (in progress)
tests/
  test_dataset.py      the selection rules, checked on diffs rather than the network
data/
  tasks.jsonl          the 20 evaluation instances
```

## Design notes

**Patches are diffed against HEAD, not the instance's base commit.** Every
SWE-bench image sits on a marker commit named `SWE-bench` that carries the
environment adaptations — for astropy, a `pyproject.toml` tweak. Diffing against
`base_commit` folds those edits into the agent's submission, which then fails to
apply when the evaluator replays it.

**The model layer is a boundary the scaffolds call through**, configured in one
place, so RQ1 (same model, different scaffold) holds by construction rather than
by discipline.

**Task selection is stratified by source-file count.** Sorting by size alone
returns almost exclusively two-file tasks, which clusters everything at the easy
end and compresses the differences between scaffolds — the exact quantity the
study exists to measure.

**`Task` deliberately omits the gold patch and the FAIL_TO_PASS / PASS_TO_PASS
lists.** They are the answer. The official harness loads them itself at grading
time, keyed by `instance_id`.

## Running the tests

```bash
./env/bin/python -m pytest tests/ -q
```

## License

MIT — see [LICENSE](LICENSE).
