"""Inference strategy for the inference-opt helix.

Agent instructions:
- Modify the INFERENCE STRATEGY section freely.
- Goal: maximize ``tokens_per_sec`` without degrading ``bpb``.
- Baseline: serial forward pass, one chunk at a time.
- Ideas: batching, quantization, torch.compile, reduced overhead, etc.

Usage:
    uv run infer.py
"""

import torch
import torch.nn.functional as F  # noqa: N812
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from prepare import evaluate


console = Console()

# Hyperparameters (edit freely)
BATCH_SIZE = 1


# INFERENCE STRATEGY — modify this section
def infer(model, _tokenizer, chunks: list[list[int]]) -> list[float]:  # type: ignore[no-untyped-def]
    """Compute log-probabilities for a batch of token chunks.

    Args:
        model: The loaded causal LM (do not modify weights).
        tokenizer: The tokenizer (for reference; not needed for pure logit scoring).
        chunks: List of token ID lists, each of length CHUNK_TOKENS.

    Returns:
        List of floats — the sum of log-probs (nats) for each chunk.
    """
    results = []
    device = model.device
    with torch.inference_mode():
        for chunk in chunks:
            ids = torch.tensor([chunk], dtype=torch.long, device=device)
            logits = model(ids).logits
            log_probs = F.cross_entropy(
                logits[:, :-1, :].transpose(1, 2),
                ids[:, 1:],
                reduction="none",
            )
            results.append(-log_probs.sum().item())
    return results


# Entry point (do not modify)
if __name__ == "__main__":
    results = evaluate(infer, batch_size=BATCH_SIZE)

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim")
    table.add_column(style="bold")
    table.add_row("tokens_per_sec", f"[cyan]{results['tokens_per_sec']:.1f}[/cyan]")
    table.add_row("bpb", f"{results['bpb']:.4f}")
    table.add_row("chunks_processed", str(results["chunks_processed"]))
    table.add_row("time_elapsed", f"{results['time_elapsed']:.1f}s")

    console.print()
    console.print(Panel(table, title="[bold cyan]results[/bold cyan]", expand=False))

    # Machine-parseable summary for grep in the experiment loop
    print("---")
    print(f"tokens_per_sec:   {results['tokens_per_sec']:.1f}")
    print(f"bpb:              {results['bpb']:.4f}")
    print(f"chunks_processed: {results['chunks_processed']}")
    print(f"time_elapsed:     {results['time_elapsed']:.1f}")
