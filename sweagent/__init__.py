"""Agent scaffoldings for small coding models on repository maintenance tasks.

C0 / C1 / C2 -> sweep run -> predictions.jsonl -> sweep grade -> SWE-bench

The scaffolds share model configuration and execution settings; total resource
usage is measured rather than capped. The runner records patches, trajectories,
usage, and time; the official SWE-bench harness supplies resolution judgments.
Human evaluation and analysis come later.

The agent loop runs in Linux/WSL, with commands in local workspaces or prepared
SWE-bench containers. Official grading runs in separate containers.
"""
