import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from typing import List, Tuple, Dict
from tqdm import tqdm
import math

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
from huggingface_hub import login
from typing import List, Tuple, Dict
import numpy as np
from tqdm import tqdm
import math

class ConceptIsolator:
    def __init__(self, model_name: str = "Qwen/Qwen3-4B-Instruct-2507"):
        import os
        
        hf_token = os.getenv('HUGGING_FACE_TOKEN')
        if not hf_token:
            raise ValueError("HUGGING_FACE_TOKEN environment variable is not set")
        login(hf_token)
        
        if not torch.cuda.is_available():
            raise RuntimeError("This implementation requires a CUDA-capable GPU")
            
        self.device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Configure model for maximum GPU utilization
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16,  # Use half precision for better performance
            "device_map": "auto"
        }
        
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
        # Store attention head metrics
        self.head_metrics = {}
        self.selected_heads = []
        self.num_layers = len(self.model.model.layers)
        # For Qwen3, we need to infer number of heads from attention shape
        # Typical values are 32 for Qwen-4B
        config = self.model.config
        self.num_heads = getattr(config, 'num_attention_heads', 32)  # Default to 32 if not specified
        
    def get_attention_hooks(self, layer_idx: int, head_outputs: Dict):
        """Hook function to capture attention head outputs."""
        def hook(module, input, output):
            # For Qwen3, output is a tuple where the first element is the attention output
            if isinstance(output, tuple):
                output = output[0]
            
            # Store the output for analysis
            head_outputs[layer_idx] = output
            
        return hook
        return hook

    def find_mediating_heads(self, pos_examples: List[str], neg_examples: List[str], 
                       start_layer: int = None, end_layer: int = None) -> List[Tuple[int, int]]:
        """Find attention heads that mediate the concept using activation patching."""
        if start_layer is None:
            start_layer = int(0.5 * self.num_layers)
        if end_layer is None:
            end_layer = int(0.9 * self.num_layers)
        
        print(f"\nAnalyzing layers {start_layer} to {end_layer}")
        print(f"Total examples: {len(pos_examples)} positive, {len(neg_examples)} negative")
        
        head_metrics = {}
        blue_token_id = self.tokenizer.encode(" blue")[0]
        
        # Function to get score for a single example
        def get_score(text):
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            return outputs.logits[0, -1, blue_token_id].item()
        
        # Increase sample size for more reliable measurements
        sample_size = min(30, len(pos_examples))  # Increased from 10
        
        # Use multiple measurements per example
        def get_reliable_score(text, n_measurements=5):
            scores = []
            for _ in range(n_measurements):
                scores.append(get_score(text))
            return np.mean(scores)
        
        # Get more reliable baseline scores
        baseline_pos = np.mean([get_reliable_score(text) for text in pos_examples[:sample_size]])
        baseline_neg = np.mean([get_reliable_score(text) for text in neg_examples[:sample_size]])
        baseline_diff = baseline_pos - baseline_neg
        
        print(f"Baseline difference: {baseline_diff:.4f}")
        
        # Analyze each layer
        for layer_idx in range(start_layer, end_layer):
            print(f"\nAnalyzing layer {layer_idx}/{end_layer-1}")
            
            # Process each head in this layer
            for head_idx in tqdm(range(self.num_heads), desc=f"Layer {layer_idx} heads"):
                # Function to mask out the specific head
                def head_mask_hook(module, input, output):
                    if isinstance(output, tuple):
                        hidden_states = output[0]
                    else:
                        hidden_states = output
                    
                    # Zero out the specific head's portion
                    head_size = hidden_states.shape[-1] // self.num_heads
                    start_idx = head_idx * head_size
                    end_idx = (head_idx + 1) * head_size
                    hidden_states[:, :, start_idx:end_idx] = 0
                    
                    if isinstance(output, tuple):
                        return (hidden_states,) + output[1:]
                    return hidden_states
                
                # Register the hook
                hook = self.model.model.layers[layer_idx].self_attn.register_forward_hook(head_mask_hook)
                
                try:
                    # Measure effect with this head masked
                    masked_pos = np.mean([get_score(text) for text in pos_examples[:sample_size]])
                    masked_neg = np.mean([get_score(text) for text in neg_examples[:sample_size]])
                    masked_diff = masked_pos - masked_neg
                    
                    # Effect is how much this head contributes to the difference
                    effect = baseline_diff - masked_diff
                    head_metrics[(layer_idx, head_idx)] = effect
                    
                    # Report the effect
                    effect_str = "STRONG" if abs(effect) > 0.5 else "MEDIUM" if abs(effect) > 0.1 else "weak"
                    direction = "+" if effect > 0 else "-" if effect < 0 else "="
                    print(f"Layer {layer_idx}, Head {head_idx}: {direction} {effect:.4f} [{effect_str}]")
                
                finally:
                    # Always remove the hook
                    hook.remove()
        
        # Sort heads by effect size and keep top k
        sorted_heads = sorted(head_metrics.items(), key=lambda x: abs(x[1]), reverse=True)
        top_k = 8  # Increased from 6
        
        # Always select top k heads, but print their significance level
        self.selected_heads = [head for head, _ in sorted_heads[:top_k]]
        self.head_metrics = head_metrics
        
        print("\nSelected head significance levels:")
        for head, effect in sorted_heads[:top_k]:
            sig_level = "STRONG" if abs(effect) > 0.5 else "MEDIUM" if abs(effect) > 0.1 else "weak"
            print(f"Layer {head[0]}, Head {head[1]}: {effect:.4f} [{sig_level}]")
        
        # Print summary of top heads
        print("\nTop contributing heads:")
        for (layer_idx, head_idx), effect in sorted_heads[:top_k]:
            print(f"Layer {layer_idx}, Head {head_idx}: {effect:.4f}")
    
        return self.selected_heads


    def extract_head_activations(self, texts: List[str]) -> torch.Tensor:
        """Extract activations from selected attention heads."""
        head_activations = []
        head_outputs = {}
        
        # Register hooks for selected heads
        if not self.selected_heads:
            return torch.zeros((len(texts), 0), device=self.device)
            
        hooks = []
        for layer_idx, head_idx in self.selected_heads:
            hook = self.model.model.layers[layer_idx].self_attn.register_forward_hook(
                self.get_attention_hooks(layer_idx, head_outputs)
            )
            hooks.append(hook)
        
        # Process each text
        all_text_activations = []
        with torch.no_grad():
            for text in tqdm(texts):
                inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
                _ = self.model(**inputs)
                
                # Collect activations from selected heads
                text_activations = []
                for layer_idx, head_idx in self.selected_heads:
                    head_output = head_outputs[layer_idx]
                    # Extract and flatten the activation for this head
                    head_activation = head_output[0, -1, head_idx].view(-1)
                    text_activations.append(head_activation)
                
                # Combine all head activations for this text
                combined = torch.cat(text_activations) if text_activations else torch.tensor([], device=self.device)
                all_text_activations.append(combined)
        
        try:
            # Stack all text activations into a single tensor
            result = torch.stack(all_text_activations)
        except:
            # If stacking fails, return empty tensor with correct size
            result = torch.zeros((len(texts), 0), device=self.device)
        
        # Clean up hooks
        for hook in hooks:
            hook.remove()
            
        return result

    def get_head_direction(self, A_pos: torch.Tensor, A_neg: torch.Tensor) -> torch.Tensor:
        """Compute the steering direction in head-output space."""
        # Compute the direction that maximizes the difference between positive and negative examples
        direction = A_pos.mean(dim=0) - A_neg.mean(dim=0)
        return direction / direction.norm()

    def causal_test(self, test_texts: List[str], v: torch.Tensor, 
                   alphas: torch.Tensor) -> Dict[float, float]:
        """Perform causal testing with different steering strengths."""
        results = {}
        
        def steering_hook(module, input, output):
            # For Qwen3, output might be a tuple where the first element is the attention output
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
                
            # Reshape the steering vector to match the hidden states
            # hidden_states shape is [batch_size, seq_len, hidden_dim]
            batch_size, seq_len, hidden_dim = hidden_states.shape
            
            # Ensure v matches hidden_dim by repeating or truncating as needed
            v_resized = v.view(-1)
            if v_resized.size(0) < hidden_dim:
                # If v is smaller, repeat it to fill hidden_dim
                repeat_factor = (hidden_dim + v_resized.size(0) - 1) // v_resized.size(0)
                v_resized = v_resized.repeat(repeat_factor)[:hidden_dim]
            else:
                # If v is larger, truncate it
                v_resized = v_resized[:hidden_dim]
                
            # Expand to match batch and sequence dimensions
            steered = v_resized.view(1, 1, hidden_dim).expand(batch_size, seq_len, hidden_dim)
            
            # Add scaled direction to the residual stream
            if isinstance(output, tuple):
                return (hidden_states + alpha * steered,) + output[1:]
            return hidden_states + alpha * steered

        for alpha in alphas:
            hook = self.model.model.layers[self.layer_idx].register_forward_hook(steering_hook)
            
            logit_diffs = []
            with torch.no_grad():
                for text in test_texts:
                    inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
                    outputs = self.model(**inputs)
                    
                    # Calculate logit difference for blue-related tokens
                    blue_token_id = self.tokenizer.encode(" blue")[0]
                    other_color_token_ids = [self.tokenizer.encode(f" {c}")[0] 
                                          for c in ["red", "green", "brown"]]
                    
                    blue_logits = outputs.logits[0, -1, blue_token_id]
                    other_logits = outputs.logits[0, -1, other_color_token_ids].mean()
                    logit_diffs.append((blue_logits - other_logits).item())
            
            hook.remove()
            # Store the mean difference for this alpha value
            results[float(alpha)] = np.mean(logit_diffs)
        
        return results

    def generate_paired_data(self) -> Tuple[List[str], List[str]]:
        """Generate more contrastive paired examples."""
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
        
        positive_examples = []
        negative_examples = []
        
        # Generate more varied "blue" examples
        blue_variants = ["blue", "azure", "cobalt", "navy", "cyan", "turquoise"]
        other_colors = ["red", "green", "yellow", "purple", "orange", "brown"]
        
        for template in templates:
            for blue in blue_variants:
                positive_examples.append(template.format(blue))
            for color in other_colors:
                negative_examples.append(template.format(color))
        
        # Multiply for more training data
        positive_examples = positive_examples * 5
        negative_examples = negative_examples * 5
        
        return positive_examples, negative_examples

    def isolate_blue_concept(self):
        """Main method to isolate the concept of 'blue' using head steering."""
        # 1. Generate paired data
        pos_examples, neg_examples = self.generate_paired_data()
        
        # 2. Find mediating heads
        print("Finding mediating heads...")
        self.find_mediating_heads(pos_examples, neg_examples)
        
        # 3. Extract head-specific activations
        print("Extracting positive activations...")
        pos_head_activations = self.extract_head_activations(pos_examples)
        print("Extracting negative activations...")
        neg_head_activations = self.extract_head_activations(neg_examples)
        
        # 4. Compute head-space direction
        direction = self.get_head_direction(pos_head_activations, neg_head_activations)
        
        # 5. Causal testing
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
        # Test with different steering strengths
        # Convert alphas to tensor to ensure consistent handling
        alphas = torch.tensor([-10.0, -7.5, -5.0, -2.5, -1.0, -0.5, 0.0, 0.5, 1.0, 2.5, 5.0, 7.5, 10.0], 
                            device=self.device)  # Steering strength values
        
        # Use the layer from the first selected head for steering, or fallback to last layer
        if self.selected_heads:
            self.layer_idx = max(head[0] for head in self.selected_heads)
        else:
            self.layer_idx = self.num_layers - 1
            print("\nWarning: No heads selected, using last layer for steering")
            
        results = self.causal_test(test_texts, direction, alphas)
        
        # Save the results
        torch.save({
            'direction': direction,
            'selected_heads': self.selected_heads,
            'head_metrics': self.head_metrics,
            'results': results
        }, 'blue_concept_heads.pt')
        
        return direction, results

if __name__ == "__main__":
    isolator = ConceptIsolator()
    direction, results = isolator.isolate_blue_concept()
    
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