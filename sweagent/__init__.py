"""Agent scaffoldings for small coding models on repository maintenance tasks.

Exit status                 resolved              reasoning        token/cost
  C0  baseline     \
  C1  full agent    >--- sweep.py ---> sweep.py  ---> judge.py ---> report
  C2  no-map       /

The contribution of this project is the leftmost column: several scaffolds,
one model, one budget. Everything to the right of the arrow is measurement,
and measurement is taken from the official SWE-bench harness rather than
reimplemented, so numbers stay comparable with published results.

The model itself never appears here. It is a boundary the scaffolds call
through, configured in one place, so RQ1 (same model, different scaffold) holds
by construction instead of by discipline.
"""
