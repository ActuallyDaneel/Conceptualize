from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from .patching import NodeSpec, patch_nodes_and_score, compute_metric_for_text


@dataclass
class ACDCResult:
    selected: List[NodeSpec]
    history: List[Tuple[Optional[NodeSpec], float]]  # (added_node, patched_metric)
    clean_metric: float
    corrupted_metric: float


def greedy_acdc(
    model,
    tokenizer,
    clean_texts: Sequence[str],
    corrupted_texts: Sequence[str],
    candidates: List[NodeSpec],
    metric_fn: Callable[[torch.Tensor], float],
    eps: float = 1e-3,
    max_steps: Optional[int] = None,
) -> ACDCResult:
    """Simplified ACDC: greedily add nodes that maximize recovery of the metric.

    - Computes baseline clean and corrupted metrics (averaged).
    - Iteratively evaluates patched_metric for S ∪ {n} and adds the best if it
      improves by ≥ eps.

    Returns selected node list and a history of additions with patched metric values.
    """
    assert len(clean_texts) == len(corrupted_texts), "Datasets must be aligned"

    # Baselines
    clean_vals = [compute_metric_for_text(model, tokenizer, t, metric_fn) for t in clean_texts]
    corrupted_vals = [compute_metric_for_text(model, tokenizer, t, metric_fn) for t in corrupted_texts]
    clean_metric = float(torch.tensor(clean_vals).mean().item())
    corrupted_metric = float(torch.tensor(corrupted_vals).mean().item())

    selected: List[NodeSpec] = []
    remaining = list(candidates)
    history: List[Tuple[Optional[NodeSpec], float]] = [(None, corrupted_metric)]

    best_val = corrupted_metric
    steps = 0

    while remaining and (max_steps is None or steps < max_steps):
        steps += 1
        best_node = None
        best_node_val = best_val

        for n in remaining:
            patched_val = patch_nodes_and_score(
                model, tokenizer, list(clean_texts), list(corrupted_texts), selected + [n], metric_fn
            )
            if patched_val > best_node_val + 0:  # strictly greater to break ties later via eps check
                best_node = n
                best_node_val = patched_val

        if best_node is None or (best_node_val - best_val) < eps:
            break

        selected.append(best_node)
        remaining.remove(best_node)
        best_val = best_node_val
        history.append((best_node, best_val))

    # Final patched value for selected set (if not already appended)
    if not history or history[-1][0] is not None:
        final_val = patch_nodes_and_score(
            model, tokenizer, list(clean_texts), list(corrupted_texts), selected, metric_fn
        ) if selected else corrupted_metric
        history.append((None, final_val))

    return ACDCResult(selected=selected, history=history, clean_metric=clean_metric, corrupted_metric=corrupted_metric)

