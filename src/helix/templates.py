"""Built-in helix templates for ``helix init``.

Each template is a dict mapping filename to content string.
Placeholders use ``{{key}}`` syntax and are filled by ``init.scaffold()``.
"""

from __future__ import annotations


AVAILABLE: frozenset[str] = frozenset({"generic", "ai-inference"})

_GENERIC_HELIX_YAML = """\
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

_GENERIC_PROGRAM_MD = """\
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
8. If `score` improved: **keep** (the commit stays).
9. Otherwise: **discard** (`git reset --hard HEAD~1`).

**NEVER STOP.** Run until interrupted.
"""

_GENERIC_SOLVER_PY = """\
\"\"\"Solver — modify this file freely.\"\"\"


def solve() -> float:
    \"\"\"Return a score. Replace with your actual solver logic.\"\"\"
    return 0.0


if __name__ == "__main__":
    print(f"score: {solve():.4f}")
"""

_GENERIC_EVALUATE_PY = """\
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

_AI_INFERENCE_HELIX_YAML = """\
name: {{name}}
version: 1.0.0
domain: AI/ML
description: >
  {{description}}

scope:
  editable:
    - infer.py
  readonly:
    - prepare.py
    - helix.yaml
    - program.md

metrics:
  primary:
    name: tokens_per_sec
    optimize: maximize
  quality_guard:
    name: bpb
    optimize: minimize
    max_degradation: 0.01
  evaluate:
    command: uv run infer.py
    timeout_seconds: 300
    output_format: pattern
    patterns:
      primary: '^tokens_per_sec:\\s+([\\d.]+)'
      quality_guard: '^bpb:\\s+([\\d.]+)'

requirements:
  python: ">=3.11"
  gpu: "GPU recommended"
"""

_AI_INFERENCE_PROGRAM_MD = """\
# {{name}}

{{description}}

## Setup

1. Set `HELIX_MODEL` to your HuggingFace model ID (default: `Qwen/Qwen2.5-0.5B-Instruct`).
2. Run `uv run prepare.py` once to download the model and cache WikiText-2.
3. Read `prepare.py` to understand the fixed evaluation harness.
4. Read `infer.py` to understand the current inference strategy.

## Constraints

- Modify `infer.py` freely: batching, quantization, `torch.compile`, kernel tricks, etc.
- Do NOT modify `prepare.py`, `helix.yaml`, or `program.md`.
- Do not fine-tune or modify the model weights.
- Do not add new dependencies beyond `pyproject.toml`.

## Metrics

**Primary: `tokens_per_sec` (maximize)**

WikiText-2 tokens scored per wall-clock second within the 5-minute budget.

**Quality guard: `bpb` (must not degrade)**

Bits per byte. Must stay within 1% of the baseline. If it rises, discard the experiment.

## Output format

```
---
tokens_per_sec:   1240.5
bpb:              0.9753
chunks_processed: 620
time_elapsed:     300.1
```

Extract metrics with: `grep "tokens_per_sec:\\|bpb:" run.log`

## Logging results

Write to `results.tsv` (tab-separated, do not git-commit). Header:

```
commit	tokens_per_sec	bpb	status	description
```

## Experiment loop

LOOP FOREVER:

1. Choose an optimization idea. Do not repeat failed experiments.
2. Modify `infer.py`.
3. `git commit` with a short description.
4. Run: `uv run infer.py > run.log 2>&1 & echo $! > run.pid; wait $!; rm -f run.pid`
5. Extract results: `grep "tokens_per_sec:\\|bpb:" run.log`
6. If results are empty: crashed. Check `tail -n 50 run.log`. Fix or move on.
7. Append a row to `results.tsv`.
8. If `tokens_per_sec` improved AND `bpb` did not degrade: **keep** (the commit stays).
9. Otherwise: **discard** (`git reset --hard HEAD~1`).

**NEVER STOP.** Run until interrupted.
"""

#: Map from template name to {filename: content} dict.
TEMPLATES: dict[str, dict[str, str]] = {
    "generic": {
        "helix.yaml": _GENERIC_HELIX_YAML,
        "program.md": _GENERIC_PROGRAM_MD,
        "solver.py": _GENERIC_SOLVER_PY,
        "evaluate.py": _GENERIC_EVALUATE_PY,
    },
    "ai-inference": {
        "helix.yaml": _AI_INFERENCE_HELIX_YAML,
        "program.md": _AI_INFERENCE_PROGRAM_MD,
    },
}
