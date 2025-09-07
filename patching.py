from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple

import torch


@dataclass(frozen=True)
class NodeSpec:
    """A model-internal node to patch.

    component: one of {"attn", "mlp"}
    layer: transformer block index
    """

    component: str
    layer: int


def _get_module_for_node(model, node: NodeSpec):
    block = model.model.layers[node.layer]
    if node.component == "attn":
        return block.self_attn
    if node.component == "mlp":
        return block.mlp
    raise ValueError(f"Unsupported component: {node.component}")


def _normalize_output(output):
    """Return the module hidden_states from either a tuple or tensor output."""
    if isinstance(output, tuple):
        return output[0], True, output
    return output, False, None


def _repack_output(hidden_states, was_tuple: bool, original_tuple):
    if was_tuple:
        return (hidden_states,) + original_tuple[1:]
    return hidden_states


def record_activations_for_nodes(model, tokenizer, text: str, nodes: Iterable[NodeSpec]) -> Dict[NodeSpec, torch.Tensor]:
    """Run a forward pass and capture outputs at specified nodes (last-token activations).

    Returns a dict mapping NodeSpec -> captured tensor (module output hidden_states).
    """
    node_set = list(nodes)
    acts: Dict[NodeSpec, torch.Tensor] = {}
    hooks = []

    def make_recorder(n: NodeSpec):
        def hook(module, inputs, output):
            hidden, is_tuple, _ = _normalize_output(output)
            # Clone to avoid in-place modifications later
            acts[n] = hidden.detach().clone()
        return hook

    for n in node_set:
        m = _get_module_for_node(model, n)
        hooks.append(m.register_forward_hook(make_recorder(n)))

    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt").to(next(model.parameters()).device)
        _ = model(**inputs)

    for h in hooks:
        h.remove()

    return acts


def run_with_patches(model, tokenizer, text: str, patches: Dict[NodeSpec, torch.Tensor]):
    """Run a forward pass while patching specified node outputs with provided tensors.

    The tensors must have the same shape as the module output. We replace the entire
    hidden_states (not just last token) to be conservative and consistent.
    """
    hooks = []

    def make_patcher(n: NodeSpec, replacement: torch.Tensor):
        def hook(module, inputs, output):
            hidden, was_tuple, orig = _normalize_output(output)
            # Ensure device/dtype alignment
            rep = replacement.to(device=hidden.device, dtype=hidden.dtype)
            # Replace only the last token slice to avoid seq_len mismatches
            hidden = hidden.clone()
            hidden[:, -1, :] = rep[:, -1, :]
            return _repack_output(hidden, was_tuple, orig)
        return hook

    for n, rep in patches.items():
        m = _get_module_for_node(model, n)
        hooks.append(m.register_forward_hook(make_patcher(n, rep)))

    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt").to(next(model.parameters()).device)
        outputs = model(**inputs)

    for h in hooks:
        h.remove()

    return outputs


# Metric helpers
def make_last_token_logit_diff_metric(tokenizer, target_token: str, negatives: List[str], token_index_strategy: str = "auto") -> Callable[[torch.Tensor], float]:
    """Return a metric function: outputs.logits -> scalar logit diff for last token.

    Resolves token IDs for the target and negatives using given strategy.
    """
    def _encode_ids(word: str) -> List[int]:
        ids = tokenizer.encode(f" {word}", add_special_tokens=False)
        if not ids:
            ids = tokenizer.encode(word, add_special_tokens=False)
        return ids

    def _tok_id(word: str, use_first: bool) -> int:
        ids = _encode_ids(word)
        return ids[0] if use_first else ids[-1]

    # Resolve strategy if auto by checking baseline on a neutral prompt
    if token_index_strategy == "auto":
        # neutral short test
        use_first = False  # fallback default
        use_first = True  # prefer first for many BPEs
        target_id = _tok_id(target_token, use_first)
    else:
        use_first = token_index_strategy == "first"
        target_id = _tok_id(target_token, use_first)

    neg_ids = [_tok_id(w, use_first) for w in negatives]

    def metric_fn(logits: torch.Tensor) -> float:
        # logits shape [1, seq_len, vocab]
        blue = logits[0, -1, target_id]
        other = logits[0, -1, neg_ids].mean()
        return float(blue.item() - other.item())

    return metric_fn


def compute_metric_for_text(model, tokenizer, text: str, metric_fn: Callable[[torch.Tensor], float]) -> float:
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt").to(next(model.parameters()).device)
        outputs = model(**inputs)
    return metric_fn(outputs.logits)


def measure_node_influence(
    model,
    tokenizer,
    clean_texts: List[str],
    corrupted_texts: List[str],
    nodes: List[NodeSpec],
    metric_fn: Callable[[torch.Tensor], float],
) -> List[Tuple[NodeSpec, float]]:
    """For each node, patch clean activations into corrupted runs and measure average improvement.

    Returns a list of (node, mean_improvement) sorted descending by absolute improvement.
    """
    assert len(clean_texts) == len(corrupted_texts), "Datasets must be aligned"

    influences: Dict[NodeSpec, List[float]] = {n: [] for n in nodes}

    for clean, corrupt in zip(clean_texts, corrupted_texts):
        # Baselines
        m_clean = compute_metric_for_text(model, tokenizer, clean, metric_fn)
        m_corrupt = compute_metric_for_text(model, tokenizer, corrupt, metric_fn)

        for n in nodes:
            # Record clean activation at node
            clean_act = record_activations_for_nodes(model, tokenizer, clean, [n])[n]
            # Patch into corrupted run
            outputs = run_with_patches(model, tokenizer, corrupt, {n: clean_act})
            m_patched = metric_fn(outputs.logits)
            influences[n].append(m_patched - m_corrupt)

    avg_influence = [(n, float(torch.tensor(v).mean().item()) if len(v) > 0 else 0.0) for n, v in influences.items()]
    avg_influence.sort(key=lambda x: abs(x[1]), reverse=True)
    return avg_influence


def patch_nodes_and_score(
    model,
    tokenizer,
    clean_texts: List[str],
    corrupted_texts: List[str],
    nodes: List[NodeSpec],
    metric_fn: Callable[[torch.Tensor], float],
) -> float:
    """Patch a set of nodes simultaneously using clean activations, evaluate mean metric on corrupted prompts."""
    assert len(clean_texts) == len(corrupted_texts)
    scores: List[float] = []
    for clean, corrupt in zip(clean_texts, corrupted_texts):
        # Collect all required clean activations in one pass
        node_to_act = record_activations_for_nodes(model, tokenizer, clean, nodes)
        outputs = run_with_patches(model, tokenizer, corrupt, node_to_act)
        scores.append(metric_fn(outputs.logits))
    return float(torch.tensor(scores).mean().item())
