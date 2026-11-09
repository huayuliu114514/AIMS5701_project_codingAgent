"""Agent scaffoldings for small coding models on repository maintenance tasks.

C0 / C1 / C2 -> sweep run -> predictions.jsonl -> sweep grade -> SWE-bench

The scaffolds share one model configuration and resource budget. The runner
records patches, trajectories, usage, and time; the official SWE-bench harness
supplies resolution judgments. Human evaluation and analysis come later.

Agents execute locally in WSL. The project's DockerEnv is unused; Docker is
managed by the official harness for grading only.
"""
