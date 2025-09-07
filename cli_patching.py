from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Dict, List, Tuple

import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

from .patching import (
    NodeSpec,
    make_last_token_logit_diff_metric,
    measure_node_influence,
    patch_nodes_and_score,
)
from .acdc import greedy_acdc


def _parse_layer_range(spec: str, num_layers: int) -> Tuple[int, int]:
    if spec is None:
        return int(0.5 * num_layers), int(0.9 * num_layers)
    if ":" not in spec:
        raise ValueError("--layers must be 'start:end'")
    s, e = spec.split(":", 1)
    s_i = int(s)
    e_i = int(e)
    if not (0 <= s_i < e_i <= num_layers):
        raise ValueError(f"Invalid layer range: {spec} with num_layers={num_layers}")
    return s_i, e_i


def _parse_components(spec: str) -> List[str]:
    allowed = {"attn", "mlp"}
    if not spec:
        return ["attn", "mlp"]
    comps = [c.strip() for c in spec.split(",") if c.strip()]
    for c in comps:
        if c not in allowed:
            raise ValueError(f"Unsupported component '{c}'. Allowed: {sorted(allowed)}")
    return comps


def _default_color_pairs(target: str) -> Tuple[List[str], List[str]]:
    # Curated cues that bias model towards/away from the target color 'blue'
    if target.lower() == "blue":
        clean_cues = ["ocean", "sky", "denim", "sapphire", "lapis", "sea"]
        corrupt_cues = ["grass", "blood", "banana", "pumpkin", "coal", "snow"]
        templates = [
            "The {} is",
            "You can tell the {} is",
            "The color of the {} is",
            "Looking at the {}, it's clearly",
        ]
        clean = [t.format(c) for t in templates for c in clean_cues]
        corrupt = [t.format(c) for t in templates for c in corrupt_cues]
        return clean, corrupt
    # Fallback neutral template pairs for other targets
    templates = [
        "The color is",
        "Clearly it's",
        "I'd say it's",
        "The predominant color is",
    ]
    # Use the same neutral forms; ACDC may still find useful nodes, but better provide your own pairs.
    return templates, templates


def _ensure_reports_dir(base_dir: str) -> str:
    out_dir = os.path.join(base_dir, "reports")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _node_to_dict(n: NodeSpec) -> Dict:
    return {"component": n.component, "layer": n.layer}


def main():
    parser = argparse.ArgumentParser(description="Activation Patching + Greedy ACDC Runner")
    parser.add_argument("--model", default=os.getenv("MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507"), help="HF model id")
    parser.add_argument("--target", default=os.getenv("TARGET_CONCEPT", "blue"), help="Target color concept")
    parser.add_argument("--neg-colors", default=os.getenv("NEG_COLORS", "red,green,yellow,purple,orange,brown"), help="Comma-separated negatives")
    parser.add_argument("--token-index-strategy", default=os.getenv("TOKEN_INDEX_STRATEGY", "auto"), choices=["auto", "first", "last"], help="Token index selection for concept term")
    parser.add_argument("--layers", default=None, help="Layer range 'start:end' (default 50%-90% of depth)")
    parser.add_argument("--components", default="attn,mlp", help="Components to include: attn,mlp (comma-separated)")
    parser.add_argument("--topk-influence", type=int, default=20, help="Top-K nodes to print in influence scan")
    parser.add_argument("--eps", type=float, default=1e-3, help="Greedy ACDC min improvement threshold")
    parser.add_argument("--max-steps", type=int, default=20, help="Greedy ACDC step cap")
    parser.add_argument("--out", default=None, help="Output report json path (defaults to ConceptMapper/reports/<target>_patching_report.json)")
    parser.add_argument("--no-acdc", action="store_true", help="Only run influence scan; skip ACDC")
    parser.add_argument("--use-default-pairs", action="store_true", help="Use built-in clean/corrupt pairs for the target concept")
    parser.add_argument("--pairs-json", default=None, help="Path to JSON with keys clean:[...], corrupt:[...] to override pairs")
    args = parser.parse_args()

    # Auth and device
    load_dotenv()
    tok = os.getenv("HUGGING_FACE_TOKEN")
    if tok:
        try:
            login(tok)
        except Exception:
            pass

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.float16 if torch.cuda.is_available() else None, device_map="auto"
    )

    # Dataset pairs
    if args.pairs_json:
        with open(args.pairs_json, "r", encoding="utf-8") as f:
            pairs = json.load(f)
        clean_texts = pairs["clean"]
        corrupted_texts = pairs["corrupt"]
    elif args.use_default_pairs:
        clean_texts, corrupted_texts = _default_color_pairs(args.target)
    else:
        # Minimal safe default; for best results, provide pairs
        clean_texts, corrupted_texts = _default_color_pairs(args.target)

    assert len(clean_texts) == len(corrupted_texts), "clean and corrupt must be aligned"

    # Metric
    negatives = [c.strip() for c in args.neg_colors.split(",") if c.strip() and c.strip().lower() != args.target.lower()]
    metric_fn = make_last_token_logit_diff_metric(tokenizer, args.target, negatives, token_index_strategy=args.token_index_strategy)

    # Candidate nodes
    num_layers = len(model.model.layers)
    start_l, end_l = _parse_layer_range(args.layers, num_layers)
    comps = _parse_components(args.components)
    candidates: List[NodeSpec] = []
    for l in range(start_l, end_l):
        for c in comps:
            candidates.append(NodeSpec(c, l))

    # Influence scan
    influences = measure_node_influence(model, tokenizer, clean_texts, corrupted_texts, candidates, metric_fn)
    print("Top influences:")
    for n, v in influences[: args.topk_influence]:
        print(f"- {n.component}@{n.layer}: {v:+.4f}")

    # Prepare report
    out_dir = _ensure_reports_dir(os.path.dirname(__file__))
    out_path = args.out or os.path.join(out_dir, f"{args.target.lower()}_patching_report.json")

    report: Dict = {
        "config": {
            "model": args.model,
            "target": args.target,
            "negatives": negatives,
            "token_index_strategy": args.token_index_strategy,
            "layers": [start_l, end_l],
            "components": comps,
        },
        "influence": [{"node": _node_to_dict(n), "delta": v} for n, v in influences],
        "acdc": None,
    }

    # Greedy ACDC
    if not args.no_acdc:
        res = greedy_acdc(
            model,
            tokenizer,
            clean_texts,
            corrupted_texts,
            candidates,
            metric_fn,
            eps=args.eps,
            max_steps=args.max_steps,
        )
        # Final patched metric for selected set
        final_patched = patch_nodes_and_score(model, tokenizer, clean_texts, corrupted_texts, res.selected, metric_fn) if res.selected else res.corrupted_metric
        denom = (res.clean_metric - res.corrupted_metric) or 1.0
        normalized_recovery = (final_patched - res.corrupted_metric) / denom
        report["acdc"] = {
            "selected": [_node_to_dict(n) for n in res.selected],
            "history": [({"node": (_node_to_dict(n) if n else None), "patched_metric": v}) for (n, v) in res.history],
            "clean_metric": res.clean_metric,
            "corrupted_metric": res.corrupted_metric,
            "final_patched_metric": final_patched,
            "normalized_recovery": normalized_recovery,
        }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved report to {out_path}")


if __name__ == "__main__":
    main()

