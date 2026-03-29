"""Built-in helix template for ``helix init``.

The template is a dict mapping filename to content string.
Placeholders use ``{{key}}`` syntax and are filled by ``init.scaffold()``.
"""

from __future__ import annotations


_HELIX_YAML = """\
name: {{name}}
version: 1.0.0
domain: {{domain}}
description: >
  {{description}}

scope:
  editable:
    - solver.py
  readonly:
    - evaluate.py
    - helix.yaml
    - program.md

metrics:
  primary:
    name: score
    optimize: maximize
  evaluate:
    command: python evaluate.py
    timeout_seconds: 300
    output_format: pattern
    patterns:
      primary: '^score:\\s+([\\d.]+)'

requirements:
  python: ">=3.11"
"""

_PROGRAM_MD = """\
# {{name}}

{{description}}

## Setup

1. Implement your evaluator in `evaluate.py`. At the end of the run it must print:

```
---
score: 0.9512
```

2. Implement your starting point in `solver.py`.

## Constraints

- Modify `solver.py` freely — any approach is in scope.
- Do NOT modify `evaluate.py`, `helix.yaml`, or `program.md`.

## Metric

**Primary: `score` (maximize).** Higher is better.

## Experiment loop

LOOP FOREVER:

1. Choose an optimization idea. Do not repeat what has already been tried.
2. Modify `solver.py`.
3. `git commit` with a short description.
4. Run: `python evaluate.py > run.log 2>&1 & echo $! > run.pid; wait $!; rm -f run.pid`
5. Extract results: `grep "score:" run.log`
6. If results are empty: run crashed. Check `tail -n 50 run.log`. Fix if trivial, otherwise log as crash and move on.
7. Append a row to `results.tsv` (tab-separated, columns: commit score status description).
   - `status` must be exactly `keep`, `discard`, or `crash` — no other values.
8. If `score` improved: **keep** (commit stays). Status = `keep`.
9. Otherwise: **discard** (`git reset --hard HEAD~1`). Status = `discard`.

**NEVER STOP.** Run until interrupted.
"""

_README_MD = """\
# {{name}}

{{description}}

## Quickstart

```bash
pip install helices
git init
helix run
```

## Metric

**Primary: `score` (maximize).** Higher is better.

## Scope

The agent may only modify `solver.py`. All other files are read-only.

---

Built with [helix](https://github.com/VectorInstitute/helix).
"""

_SOLVER_PY = """\
\"\"\"Solver — modify this file freely.\"\"\"


def solve() -> float:
    \"\"\"Return a score. Replace with your actual solver logic.\"\"\"
    return 0.0


if __name__ == "__main__":
    print(f"score: {solve():.4f}")
"""

_EVALUATE_PY = """\
\"\"\"Evaluation harness — do not modify.

Runs the solver and prints machine-readable results.
\"\"\"

import time

from solver import solve

TIME_BUDGET = 60  # seconds

if __name__ == "__main__":
    t_start = time.time()
    score = solve()
    elapsed = time.time() - t_start

    print("---")
    print(f"score:        {score:.4f}")
    print(f"time_elapsed: {elapsed:.1f}")
"""

#: Files created by ``helix init``.
TEMPLATE: dict[str, str] = {
    "README.md": _README_MD,
    "helix.yaml": _HELIX_YAML,
    "program.md": _PROGRAM_MD,
    "solver.py": _SOLVER_PY,
    "evaluate.py": _EVALUATE_PY,
}
