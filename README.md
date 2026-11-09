# Agent Scaffolding Study

Comparing agent scaffoldings for small coding models on repository maintenance
tasks — three scaffolds, one model, shared execution settings.

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

Early. The runner, official grading integration, and C0 baseline are implemented;
C1 and C2 are not yet built.

- [x] Task selection with the proposal's filter (2–4 non-test source files, ≥3 repos)
- [x] Sequential runner and prediction recording
- [x] Verify official grading on one instance (gold patch resolves)
- [x] C0 baseline scaffold (benchmark evaluation pending)
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
inside them. Agents can use these prepared containers or local Linux checkouts.
C0 calls an OpenAI Responses-compatible endpoint; start from
`configs/c0.example.json` and use `python -m sweagent.sweep run --help` for options.

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
  sweep.py             agent runner and official grading integration
  model.py             shared Responses-compatible model client
  agents/c0_agent.py   model-directed baseline with bash and finish tools
  envs/                local Linux and prepared-container execution
tests/
  test_dataset.py      the selection rules, checked on diffs rather than the network
  test_sweep.py        runner, patch capture, and grading integration
  test_c0.py           baseline conversation and tool behavior
data/
  tasks.jsonl          the 20 evaluation instances
```

## Design notes

**Patches are diffed against the workspace's starting HEAD.** Local checkouts
start at the instance's `base_commit`. Image-based workspaces can include
additional environment adaptations, so their marker HEAD is the correct base.
The runner records the starting commit before invoking the agent, ensuring
that any commits the agent makes are also included in the submitted patch.

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
