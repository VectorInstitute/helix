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
  editable: []  # TODO: list files the agent may modify, e.g. [train.py, model.py]
  readonly:
    - helix.yaml
    - program.md
    - pyproject.toml
    - .python-version

metrics:
  primary:
    name: score
    optimize: maximize
  evaluate:
    # TODO: replace with the command that runs your evaluation and prints metrics to stdout
    command: python evaluate.py
    timeout_seconds: 300
    output_format: pattern
    patterns:
      primary: '^score:\\s+([\\d.]+)'

requirements:
  python: ">=3.12"
"""

_PROGRAM_MD = """\
# {{name}}

{{description}}

## Codebase

<!-- Describe the repository layout. For each key file, explain what it does. -->
<!-- Example:
- `train.py` — entry point; accepts `--epochs` and `--lr` flags
- `model.py` — model definition
- `data.py` — dataset loading and preprocessing
-->

## Goal

<!-- What are you optimizing and why? What does a good result look like? -->

Maximize `score` (defined in `helix.yaml`). Higher is better.

## Constraints

- Do NOT modify `helix.yaml`, `program.md`, `pyproject.toml`, or `.python-version`.
<!-- Add further constraints specific to your problem, e.g.: -->
<!-- - Do not change the model architecture. -->
<!-- - Do not add new dependencies. -->

## Techniques to try

<!-- List domain-specific ideas ordered roughly by expected impact. -->
<!-- The agent will work through these and discover others. -->

## Experiment loop

LOOP FOREVER:

1. Choose an idea not yet tried.
2. Implement it.
3. `git commit` with a short description.
4. Run the evaluation command from `helix.yaml`.
5. Extract the metric value from the output.
6. If the run crashed: log as `crash` in `results.tsv` and move on.
7. Append a row to `results.tsv` (columns: commit, score, status, description).
   - `status` must be exactly `keep`, `discard`, or `crash`.
8. If the primary metric improved: keep (commit stays). Status = `keep`.
9. Otherwise: `git reset --hard HEAD~1`. Status = `discard`.

**NEVER STOP.** Run until interrupted.
"""

_README_MD = """\
# {{name}}

{{description}}

## Quickstart

```bash
uv sync
helix run
```

## Scope

The agent may only modify files listed under `scope.editable` in `helix.yaml`.

---

Built with [helix](https://github.com/VectorInstitute/helix).
"""

_PYPROJECT_TOML = """\
[project]
name = "{{name}}"
version = "0.1.0"
description = "{{description}}"
requires-python = ">=3.12"
dependencies = []
"""

_PYTHON_VERSION = "3.12\n"

#: Files created by ``helix init``. Values are rendered before writing.
TEMPLATE: dict[str, str] = {
    "helix.yaml": _HELIX_YAML,
    "program.md": _PROGRAM_MD,
    "README.md": _README_MD,
    "pyproject.toml": _PYPROJECT_TOML,
    ".python-version": _PYTHON_VERSION,
}
