import os
import math
import argparse
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")


class ConceptIsolator:
    def __init__(self,
                 model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
                 target_concept: str = "blue",
                 neg_colors: Optional[List[str]] = None,
                 token_index_strategy: Optional[str] = None,
                 steering_mode: Optional[str] = None,
                 top_k: Optional[int] = None,
                 start_layer: Optional[int] = None,
                 end_layer: Optional[int] = None,
                 ):

        # Load environment variables from .env file
        load_dotenv()

        hf_token = os.getenv('HUGGING_FACE_TOKEN')
        if not hf_token:
            raise ValueError("HUGGING_FACE_TOKEN not found in .env file")
        login(hf_token)

        if not torch.cuda.is_available():
            raise RuntimeError("This implementation requires a CUDA-capable GPU")

        self.device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Configure model for GPU execution
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16,  # Use half precision for GPU memory efficiency
            "device_map": "auto"
        }

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )

        # Store attention head metrics/state
        self.head_metrics: Dict[Tuple[int, int], float] = {}
        self.selected_heads: List[Tuple[int, int]] = []
        self.num_layers = len(self.model.model.layers)
        config = self.model.config
        self.num_heads = getattr(config, 'num_attention_heads', 32)
        self.hidden_size = getattr(config, 'hidden_size', None)
        if self.hidden_size is None:
            self.hidden_size = self.model.model.embed_tokens.embedding_dim

        # Runtime-tunable strategies (env overrides + CLI)
        self.target_concept = os.getenv('TARGET_CONCEPT', target_concept).strip()
        self.token_index_strategy = (token_index_strategy or os.getenv('TOKEN_INDEX_STRATEGY', 'auto')).strip()
        self.steering_mode = (steering_mode or os.getenv('STEERING_MODE', 'broadcast')).strip()
        if neg_colors is not None:
            self.neg_colors = [c for c in neg_colors if c]
        else:
            neg_colors_env = os.getenv('NEG_COLORS', '')
            if neg_colors_env:
                self.neg_colors = [c.strip() for c in neg_colors_env.split(',') if c.strip()]
            else:
                self.neg_colors = ["red", "green", "brown", "yellow", "orange", "purple", "black", "white"]
        self.neg_colors = [c for c in self.neg_colors if c.lower() != self.target_concept.lower()]
        self.top_k = int(top_k or os.getenv('TOP_K', 12))
        self._start_layer = start_layer
        self._end_layer = end_layer
        self._resolved_token_index_strategy = None

        # Try to load saved heads for this concept
        safe_concept = ''.join(ch for ch in self.target_concept.lower() if ch.isalnum() or ch in ('-', '_')) or 'concept'
        self.saved_data_path = f'{safe_concept}_concept_heads.pt'
        try:
            saved_data = torch.load(self.saved_data_path)
            self.selected_heads = saved_data.get('selected_heads', [])
            self.head_metrics = saved_data.get('head_metrics', {})
            print(f"\nLoaded {len(self.selected_heads)} saved mediating heads for '{self.target_concept}' from {self.saved_data_path}")
        except Exception:
            print(f"\nNo saved heads found for '{self.target_concept}', will scan for mediating heads")

    def get_attention_hooks(self, layer_idx: int, head_outputs: Dict):
        """Hook function to capture attention module outputs per layer."""
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            head_outputs[layer_idx] = output
        return hook

    def _color_token_id(self, color: str) -> int:
        strategy = self._resolved_token_index_strategy or self.token_index_strategy
        ids = self.tokenizer.encode(f" {color}", add_special_tokens=False)
        if not ids:
            ids = self.tokenizer.encode(color, add_special_tokens=False)
        if strategy == 'first':
            return ids[0]
        return ids[-1]

    def find_mediating_heads(self, pos_examples: List[str], neg_examples: List[str],
                              start_layer: Optional[int] = None, end_layer: Optional[int] = None) -> List[Tuple[int, int]]:
        """Find attention heads that mediate the concept via head masking on last token."""
        if start_layer is None:
            start_layer = self._start_layer if self._start_layer is not None else int(0.5 * self.num_layers)
        if end_layer is None:
            end_layer = self._end_layer if self._end_layer is not None else int(0.9 * self.num_layers)

        print(f"\nAnalyzing layers {start_layer} to {end_layer} for '{self.target_concept}'")

        cloze_prompts = [
            "The color is",
            "The color is definitely",
            "I would call this",
            "It looks like",
            "The predominant color is",
            "I'd say it's",
            "The shade is",
            "It appears to be",
            "The hue is",
            "Clearly it's",
        ]

        head_metrics: Dict[Tuple[int, int], float] = {}

        # Resolve token index strategy if 'auto'
        def _tok_id(color: str, use_first: bool) -> int:
            ids = self.tokenizer.encode(f" {color}", add_special_tokens=False)
            if not ids:
                ids = self.tokenizer.encode(color, add_special_tokens=False)
            return ids[0] if use_first else ids[-1]

        def _avg_margin(use_first: bool) -> float:
            concept_id = _tok_id(self.target_concept, use_first)
            other_ids = [_tok_id(c, use_first) for c in self.neg_colors]
            vals = []
            with torch.no_grad():
                for p in cloze_prompts:
                    inputs = self.tokenizer(p, return_tensors="pt").to(self.device)
                    outputs = self.model(**inputs)
                    c_logit = outputs.logits[0, -1, concept_id]
                    o_logit = outputs.logits[0, -1, other_ids].mean()
                    vals.append((c_logit - o_logit).item())
            return float(np.mean(vals))

        if self.token_index_strategy == 'auto':
            first_m = _avg_margin(True)
            last_m = _avg_margin(False)
            self._resolved_token_index_strategy = 'first' if abs(first_m) >= abs(last_m) else 'last'
            print(f"Resolved token index strategy: {self._resolved_token_index_strategy} (first={first_m:.4f}, last={last_m:.4f})")
        else:
            self._resolved_token_index_strategy = self.token_index_strategy

        concept_token_id = self._color_token_id(self.target_concept)
        neg_token_ids = [self._color_token_id(c) for c in self.neg_colors]

        def get_score(prompt: str) -> float:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            c_logit = outputs.logits[0, -1, concept_token_id]
            o_logit = outputs.logits[0, -1, neg_token_ids].mean()
            return (c_logit - o_logit).item()

        baseline_diff = float(np.mean([get_score(p) for p in cloze_prompts]))
        print(f"Baseline difference: {baseline_diff:.4f}")

        for layer_idx in range(start_layer, end_layer):
            print(f"\nAnalyzing layer {layer_idx}/{end_layer-1}")
            for head_idx in tqdm(range(self.num_heads), desc=f"Layer {layer_idx} heads"):
                def head_mask_hook(module, input, output):
                    if isinstance(output, tuple):
                        hidden_states = output[0]
                    else:
                        hidden_states = output
                    head_size = hidden_states.shape[-1] // self.num_heads
                    s = head_idx * head_size
                    e = (head_idx + 1) * head_size
                    hidden_states[:, -1, s:e] = 0
                    if isinstance(output, tuple):
                        return (hidden_states,) + output[1:]
                    return hidden_states

                hook = self.model.model.layers[layer_idx].self_attn.register_forward_hook(head_mask_hook)
                try:
                    masked_diff = float(np.mean([get_score(p) for p in cloze_prompts]))
                    effect = baseline_diff - masked_diff
                    head_metrics[(layer_idx, head_idx)] = effect
                    label = "STRONG" if abs(effect) > 0.5 else ("MEDIUM" if abs(effect) > 0.1 else "weak")
                    sign = "+" if effect > 0 else ("-" if effect < 0 else "=")
                    print(f"Layer {layer_idx}, Head {head_idx}: {sign} {effect:.4f} [{label}]")
                finally:
                    hook.remove()

        sorted_heads = sorted(head_metrics.items(), key=lambda x: abs(x[1]), reverse=True)
        top_k = int(self.top_k)
        self.selected_heads = [head for head, _ in sorted_heads[:top_k]]
        self.head_metrics = head_metrics

        print("\nSelected head significance levels:")
        for head, effect in sorted_heads[:top_k]:
            sig_level = "STRONG" if abs(effect) > 0.5 else "MEDIUM" if abs(effect) > 0.1 else "weak"
            print(f"Layer {head[0]}, Head {head[1]}: {effect:.4f} [{sig_level}]")

        print("\nTop contributing heads:")
        for (layer_idx, head_idx), effect in sorted_heads[:top_k]:
            print(f"Layer {layer_idx}, Head {head_idx}: {effect:.4f}")

        return self.selected_heads

    def extract_head_activations(self, texts: List[str]) -> torch.Tensor:
        """Extract concatenated per-head activations at last token for selected heads."""
        if not self.selected_heads:
            return torch.zeros((len(texts), 0), device=self.device)

        # Deduplicate hooks per layer
        layers_to_hook = sorted({l for (l, _) in self.selected_heads})
        head_outputs: Dict[int, torch.Tensor] = {}
        hooks = []
        for layer_idx in layers_to_hook:
            hook = self.model.model.layers[layer_idx].self_attn.register_forward_hook(
                self.get_attention_hooks(layer_idx, head_outputs)
            )
            hooks.append(hook)

        all_text_activations = []
        with torch.no_grad():
            for text in tqdm(texts):
                inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
                _ = self.model(**inputs)
                text_activations = []
                for layer_idx, head_idx in self.selected_heads:
                    hidden_states = head_outputs[layer_idx]
                    hidden_dim = hidden_states.shape[-1]
                    head_size = hidden_dim // self.num_heads
                    s = head_idx * head_size
                    e = (head_idx + 1) * head_size
                    head_vec = hidden_states[0, -1, s:e]
                    text_activations.append(head_vec.view(-1))
                combined = torch.cat(text_activations) if text_activations else torch.tensor([], device=self.device)
                all_text_activations.append(combined)

        try:
            result = torch.stack(all_text_activations)
        except Exception:
            result = torch.zeros((len(texts), 0), device=self.device)

        for hook in hooks:
            hook.remove()

        return result

    def get_head_direction(self, A_pos: torch.Tensor, A_neg: torch.Tensor) -> torch.Tensor:
        v = A_pos.mean(dim=0) - A_neg.mean(dim=0)
        n = float(v.norm().item()) if v.numel() > 0 else 0.0
        if n == 0.0:
            return v
        return v / v.norm()

    def _build_residual_direction(self, v_headspace: torch.Tensor, target_layer: int) -> torch.Tensor:
        hidden_dim = self.hidden_size
        head_size = hidden_dim // self.num_heads
        v_resid = torch.zeros(hidden_dim, device=self.device, dtype=v_headspace.dtype)
        for k, (layer_idx, head_idx) in enumerate(self.selected_heads):
            if layer_idx != target_layer:
                continue
            sb = k * head_size
            eb = (k + 1) * head_size
            if eb > v_headspace.numel():
                break
            block = v_headspace[sb:eb]
            s = head_idx * head_size
            e = (head_idx + 1) * head_size
            v_resid[s:e] += block
        norm = v_resid.norm()
        if norm > 0:
            v_resid = v_resid / norm
        return v_resid

    def causal_test(self, test_texts: List[str], v_vec: torch.Tensor, alphas: torch.Tensor) -> Dict[float, float]:
        def make_hook(alpha: float):
            def steering_hook(module, input, output):
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                bsz, seqlen, hidden_dim = hidden_states.shape
                if self.steering_mode == 'targeted':
                    v_residual = v_vec
                else:
                    v_resized = v_vec.view(-1)
                    if v_resized.size(0) < hidden_dim:
                        repeat_factor = (hidden_dim + v_resized.size(0) - 1) // v_resized.size(0)
                        v_residual = v_resized.repeat(repeat_factor)[:hidden_dim]
                    else:
                        v_residual = v_resized[:hidden_dim]
                v_expanded = torch.zeros_like(hidden_states)
                v_expanded[:, -1, :] = v_residual.view(1, 1, hidden_dim).expand(bsz, 1, hidden_dim)
                if isinstance(output, tuple):
                    return (hidden_states + alpha * v_expanded,) + output[1:]
                return hidden_states + alpha * v_expanded
            return steering_hook

        all_results: Dict[float, float] = {}
        for alpha in alphas:
            hook = self.model.model.layers[self.layer_idx].register_forward_hook(make_hook(float(alpha)))
            logit_diffs = []
            with torch.no_grad():
                for text in test_texts:
                    inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
                    outputs = self.model(**inputs)
                    concept_id = self._color_token_id(self.target_concept)
                    other_ids = [self._color_token_id(c) for c in self.neg_colors]
                    c_logits = outputs.logits[0, -1, concept_id]
                    o_logits = outputs.logits[0, -1, other_ids].mean()
                    logit_diffs.append((c_logits - o_logits).item())
            hook.remove()
            all_results[float(alpha)] = float(np.mean(logit_diffs))
        return all_results

    def generate_paired_data(self) -> Tuple[List[str], List[str]]:
        templates = [
            "The {} stands out against the background.",
            "I would describe this as purely {}.",
            "The color is definitely {}.",
            "It's the most vibrant shade of {} I've ever seen.",
            "The entire room was painted {}.",
            "Looking closer, I noticed it was {}.",
            "Without doubt, that's {}.",
            "The dominant color was {}.",
            "It could only be described as {}.",
            "The main feature was its {} color."
        ]

        pos, neg = [], []
        if self.target_concept.lower() == "blue":
            variants = ["blue", "azure", "cobalt", "navy", "cyan", "turquoise"]
        else:
            syn_env = os.getenv('TARGET_SYNONYMS', '')
            variants = [s.strip() for s in syn_env.split(',') if s.strip()] if syn_env else [self.target_concept]
        other_colors = ["red", "green", "yellow", "purple", "orange", "brown"]
        for t in templates:
            for v in variants:
                pos.append(t.format(v))
            for c in other_colors:
                neg.append(t.format(c))
        pos = pos * 5
        neg = neg * 5
        return pos, neg

    def isolate_concept(self) -> Tuple[torch.Tensor, Dict[float, float]]:
        pos_examples, neg_examples = self.generate_paired_data()

        if not self.selected_heads:
            print("Finding mediating heads...")
            self.find_mediating_heads(pos_examples, neg_examples)
        else:
            print(f"Using {len(self.selected_heads)} previously identified mediating heads")

        print("Extracting positive activations...")
        A_pos = self.extract_head_activations(pos_examples)
        print("Extracting negative activations...")
        A_neg = self.extract_head_activations(neg_examples)

        direction = self.get_head_direction(A_pos, A_neg)

        test_texts = [
            "The color I see is",
            "When I look at this, I'd say it's",
            "The primary color here is clearly",
            "Without hesitation, I'd call this",
            "If I had to pick one color, it's",
            "The most accurate description would be",
            "Looking at this, I immediately think",
            "The predominant shade is definitely",
            "This can only be described as",
            "The color that stands out most is"
        ]
        alphas = torch.tensor([-10.0, -7.5, -5.0, -2.5, -1.0, -0.5, 0.0, 0.5, 1.0, 2.5, 5.0, 7.5, 10.0], device=self.device)

        if self.selected_heads:
            self.layer_idx = max(h[0] for h in self.selected_heads)
        else:
            self.layer_idx = self.num_layers - 1
            print("\nWarning: No heads selected, using last layer for steering")

        if self.steering_mode == 'targeted':
            direction_vec = self._build_residual_direction(direction, self.layer_idx)
        else:
            direction_vec = direction
        results = self.causal_test(test_texts, direction_vec, alphas)

        torch.save({
            'direction': direction,
            'selected_heads': self.selected_heads,
            'head_metrics': self.head_metrics,
            'results': results,
        }, self.saved_data_path)

        return direction, results

    def isolate_blue_concept(self):
        # Backward-compatible alias
        return self.isolate_concept()

    def rescan_heads(self):
        self.selected_heads = []
        self.head_metrics = {}
        if os.path.exists(self.saved_data_path):
            os.remove(self.saved_data_path)
        print("Cleared saved heads. Will perform full scan on next run.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Isolate and steer a color concept in Qwen3-4B Instruct")
    parser.add_argument("--target", default=os.getenv('TARGET_CONCEPT', 'blue'), help="Target color concept (e.g., blue)")
    parser.add_argument("--model", default=os.getenv('MODEL_NAME', 'Qwen/Qwen3-4B-Instruct-2507'), help="HF model id")
    parser.add_argument("--neg-colors", default=os.getenv('NEG_COLORS', ''), help="Comma-separated negative colors override")
    parser.add_argument("--token-index-strategy", default=os.getenv('TOKEN_INDEX_STRATEGY', 'auto'), choices=['auto','first','last'], help="Choose token position for concept term")
    parser.add_argument("--mode", default=os.getenv('STEERING_MODE', 'broadcast'), choices=['broadcast','targeted'], help="Steering mode")
    parser.add_argument("--topk", type=int, default=int(os.getenv('TOP_K', 12)), help="Top-K heads to select")
    parser.add_argument("--start-layer", type=int, default=None, help="Start layer index for scan")
    parser.add_argument("--end-layer", type=int, default=None, help="End layer index (exclusive) for scan")
    parser.add_argument("--rescan", action="store_true", help="Force rescan of mediating heads")
    args = parser.parse_args()

    negs = [c.strip() for c in args.neg_colors.split(',') if c.strip()] if args.neg_colors else None

    isolator = ConceptIsolator(
        model_name=args.model,
        target_concept=args.target,
        neg_colors=negs,
        token_index_strategy=args.token_index_strategy,
        steering_mode=args.mode,
        top_k=args.topk,
        start_layer=args.start_layer,
        end_layer=args.end_layer,
    )

    if args.rescan:
        isolator.rescan_heads()

    direction, results = isolator.isolate_concept()

    # Print results
    print("\nSelected attention heads:")
    for layer_idx, head_idx in isolator.selected_heads:
        effect = isolator.head_metrics[(layer_idx, head_idx)]
        print(f"Layer {layer_idx}, Head {head_idx}: Effect = {effect:.3f}")

    print("\nCausal testing results:")
    logit_diffs = list(results.values())
    logit_range = max(logit_diffs) - min(logit_diffs)
    print(f"Logit range: {logit_range:.3f} ({min(logit_diffs):.3f} to {max(logit_diffs):.3f})")

    print("\nDetailed results:")
    for alpha, diff in sorted(results.items()):
        print(f"Alpha: {alpha:6.2f}, Logit diff: {diff:6.2f}")
