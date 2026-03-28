<p align="center">
  <img src="assets/logo.svg" alt="helix" width="700"/>
</p>

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

```bash
pip install helices
```

### Start from a template

```bash
helix init my-project --template generic --domain "AI/ML" --description "Optimize X for task Y."
cd my-project
git init
helix run
```

### Run an existing helix

```bash
# from within a helix directory (one that has helix.yaml)
helix run              # start a session tagged with today's date
helix run --tag exp1   # custom tag
helix status           # show current best and recent experiments
```

## Templates

| Template | Description |
|---|---|
| `generic` | Blank slate: `solver.py` + `evaluate.py`. Print `score: <value>` at the end. |
| `ai-inference` | LLM inference throughput on WikiText-2. Metrics: `tokens_per_sec` + `bpb`. |

## Reference helix

`examples/inference-opt/` is a complete working helix that optimizes inference
throughput for a causal language model on WikiText-2.

```bash
cd examples/inference-opt
uv run prepare.py   # one-time: download model + dataset
helix run
```

The agent modifies `infer.py` (batching, quantization, `torch.compile`, etc.) and
automatically merges improvements back to the main branch.

## Writing your own helix

1. Create a new git repo.
2. Add `helix.yaml` describing your metric, evaluation command, and editable scope.
3. Add `program.md` with domain-specific instructions for the agent.
4. Add your codebase.
5. Run `helix run`.

Minimal `helix.yaml`:

```yaml
name: my-helix
domain: AI/ML
description: Optimize X for task Y.

scope:
  editable: [solver.py]
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

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `HELIX_MODEL` | `Qwen/Qwen2.5-0.5B-Instruct` | HuggingFace model ID (inference-opt) |
| `HELIX_HARDWARE` | `unknown` | Hardware description shown in the startup panel |
| `HELIX_TIME_BUDGET` | `300` | Seconds per experiment (inference-opt) |
| `HELIX_CHUNK_TOKENS` | `512` | Tokens per WikiText-2 chunk (inference-opt) |
