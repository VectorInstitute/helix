<p align="center">
  <img src="assets/logo.svg" alt="helix" width="700"/>
</p>

[![code checks](https://github.com/VectorInstitute/helix/actions/workflows/code_checks.yml/badge.svg)](https://github.com/VectorInstitute/helix/actions/workflows/code_checks.yml)
[![unit tests](https://github.com/VectorInstitute/helix/actions/workflows/unit_tests.yml/badge.svg)](https://github.com/VectorInstitute/helix/actions/workflows/unit_tests.yml)
[![codecov](https://codecov.io/github/VectorInstitute/helix/graph/badge.svg)](https://codecov.io/github/VectorInstitute/helix)
[![PyPI](https://img.shields.io/pypi/v/helices)](https://pypi.org/project/helices/)
![GitHub License](https://img.shields.io/github/license/VectorInstitute/helix)

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch), helix generalizes
the idea of autonomous AI research loops beyond LLM training. Give an agent a codebase, a metric,
and a fixed time budget. It experiments overnight. You wake up to results.

The git history is the research trail. `experiments.tsv` is the proof. Anyone can clone a helix,
run it on their hardware, and independently verify every result.

## Concepts

| Term | Meaning |
|---|---|
| **helix** | A git repo containing `helix.yaml` + `program.md` + a codebase the agent can modify |
| `helix.yaml` | Machine-readable spec: what to optimize, how to measure it, which files are editable |
| `program.md` | Human-written instructions for the agent: domain knowledge, constraints, techniques to try |
| `experiments.tsv` | Append-only ledger of every experiment: commit, metric, status, description |
| `helix run` | CLI command that launches an autonomous session on your hardware |

## Quick start

helix is agent-agnostic. Pick a backend or bring your own.

| Backend | Install | Requires |
|---|---|---|
| `ClaudeBackend` (default) | `pip install 'helices[claude]'` | [Claude Code CLI](https://claude.ai/download) |
| `GeminiBackend` | `pip install helices` | [Gemini CLI](https://github.com/google-gemini/gemini-cli) |
| Custom | `pip install helices` | Implement the `AgentBackend` protocol |

### Run an existing helix

```bash
# from within a helix directory (one that has helix.yaml)
helix run              # start a session tagged with today's date
helix run --tag exp1   # custom tag
helix status           # show current best and recent experiments
```

## Examples

[helix-examples](https://github.com/VectorInstitute/helix-examples) is a curated gallery of
standalone helices, each in its own repo and included as a git submodule.

```bash
git clone --recurse-submodules git@github.com:VectorInstitute/helix-examples.git
cd helix-examples/inference-opt
uv run prepare.py   # one-time: download model + dataset
helix run
```

The first example, [helix-inference-opt](https://github.com/VectorInstitute/helix-inference-opt),
optimizes inference throughput for a causal language model on WikiText-2. The agent modifies
`infer.py` (batching, quantization, `torch.compile`, etc.) and automatically merges improvements
back to main.

## Writing your own helix

The typical starting point is an existing research codebase. `helix init` drops the helix
layer on top without touching your code.

```bash
cd my-research-project        # your existing git repo
pip install 'helices[claude]'
helix init . --domain "AI/ML" --description "Optimize X for task Y."
```

`helix init` is non-destructive — it skips any file that already exists, so running it
against a repo with an existing `pyproject.toml` or `uv.lock` is safe.

Then:

1. Edit `helix.yaml` — set `scope.editable` to the files the agent may modify, and set `evaluate.command` to your evaluation script.
2. Edit `program.md` — describe your codebase, goal, constraints, and techniques to try.
3. Run `helix run`.

If you're starting from scratch, `helix init` will also scaffold the directory for you:

```bash
helix init my-project --domain "AI/ML" --description "Optimize X for task Y."
cd my-project && git init
# add your codebase, fill in helix.yaml and program.md, then:
helix run
```

### Minimal `helix.yaml`

```yaml
name: my-helix
domain: AI/ML
description: Optimize X for task Y.

scope:
  editable: [train.py]
  readonly: [evaluate.py, program.md, helix.yaml]

metrics:
  primary:
    name: accuracy
    optimize: maximize
  evaluate:
    command: python evaluate.py
    timeout_seconds: 120
    output_format: pattern
    patterns:
      primary: '^accuracy:\s+([\d.]+)'
```
