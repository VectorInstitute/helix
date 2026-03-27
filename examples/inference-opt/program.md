# inference-opt

Autonomous research into inference throughput for a causal language model.

## Setup

Before starting a session:

1. Verify the model is cached: check `~/.cache/autoresearch/`. If not, run `uv run prepare.py` once.
2. Read `prepare.py` to understand the fixed evaluation harness.
3. Read `infer.py` to understand the current inference strategy.

## Experimentation

Each experiment runs for a **fixed time budget of 5 minutes** (wall clock, excluding model loading ~30s).

**What you CAN do:**

- Modify `infer.py` freely. Any inference-time technique is in scope: batching, quantization,
  `torch.compile`, custom CUDA kernels, reduced-overhead scheduling, speculative decoding, etc.
- There are no prescribed approaches. Reason from first principles.

**What you CANNOT do:**

- Modify `prepare.py`. It owns the evaluation harness and ground-truth scoring. It is read-only.
- Fine-tune or modify the model weights. The model is fixed.
- Install new packages beyond those in `pyproject.toml`.

## Metrics

**Primary: `tokens_per_sec` (maximize)**

WikiText-2 tokens scored per wall-clock second within the 5-minute budget. Pure throughput.

**Quality guard: `bpb` (must not degrade)**

Bits per byte. Measures whether the inference produces correct log-probabilities. If `bpb` rises
meaningfully from baseline, the experiment is producing wrong scores and must be discarded.

## Output format

After `uv run infer.py` completes, the script prints a machine-readable summary:

```
---
tokens_per_sec:   1240.5
bpb:              0.9753
chunks_processed: 620
time_elapsed:     300.1
```

Extract metrics with:

```bash
grep "tokens_per_sec:\|bpb:" run.log
```

## Logging results

Write to `results.tsv` (tab-separated, do not git-commit this file). Header:

```
commit	tokens_per_sec	bpb	status	description
```

Columns:

1. `commit`: 7-char git short hash
2. `tokens_per_sec`: e.g. `1240.5`
3. `bpb`: e.g. `0.9753`
4. `status`: `keep`, `discard`, or `crash`
5. `description`: brief description of the technique tried

Example:

```
commit	tokens_per_sec	bpb	status	description
a1b2c3d	1240.5	0.9753	keep	baseline serial forward pass
b2c3d4e	2180.3	0.9758	keep	batch_size=8 padded batching
c3d4e5f	2410.1	1.1200	discard	batch_size=16 bpb degraded
d4e5f6g	0.0	inf	crash	custom kernel OOM
```

## The experiment loop

LOOP FOREVER:

1. Choose an optimization idea. Look at what has already been tried; do not repeat failures.
2. Modify `infer.py`.
3. `git commit` the change with a short description.
4. Run: `uv run infer.py > run.log 2>&1 & echo $! > run.pid; wait $!; rm -f run.pid`
5. Extract results: `grep "tokens_per_sec:\|bpb:" run.log`
6. If results are empty, the run crashed. Check `tail -n 50 run.log`. Fix if trivial; otherwise log as crash and move on.
7. Append a row to `results.tsv`.
8. If `tokens_per_sec` improved AND `bpb` did not degrade: **keep** (the commit stays).
9. Otherwise: **discard** (`git reset --hard HEAD~1`).

**Timeout:** If a run exceeds 10 minutes total, kill it (`kill $(cat run.pid)`) and treat as crash.

**Simplicity:** All else equal, simpler code that achieves the same throughput is preferred.

**NEVER STOP:** Once the loop begins, do not pause to ask for confirmation. You are autonomous.
The human may be asleep. Run until interrupted, period.
