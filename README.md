# ConceptMapper

A tool for analyzing and steering concept representations in large language models, specifically designed for the Qwen-4B Instruct model.

## Features

- Identifies attention heads that mediate specific concepts (currently focused on color concepts)
- Extracts and analyzes head-specific activations
- Supports concept steering with controllable strength
- Provides detailed analysis of head contributions
- Implements causal testing for concept manipulation
 - Modular activation patching utilities and a simplified ACDC (greedy) search

## Requirements

- Python 3.8+
- PyTorch with CUDA support
- Transformers, Accelerate, Safetensors
- Hugging Face account with appropriate model access

## Installation

1. Install required packages:
```bash
pip install -r ConceptMapper/requirements.txt
```

2. Create a `.env` file in the project root directory with your Hugging Face token:
```plaintext
HUGGING_FACE_TOKEN="your_token_here"
```

The `.env` file is already in `.gitignore` and won't be pushed to the repository.

## Usage

Run the CLI (downloads the model on first run):

```bash
python ConceptMapper/main.py --target blue --mode targeted --topk 12 --rescan
```

Common flags:
- `--target`: target color concept (default: `blue`).
- `--mode`: `broadcast` (head-space) or `targeted` (residual of selected heads).
- `--topk`: number of mediating heads to select (default: 12).
- `--start-layer/--end-layer`: limit layer scan range.
- `--token-index-strategy`: `auto` (default), `first`, or `last` token of the word.
- `--neg-colors`: comma-separated list to override negatives.
- `--rescan`: ignore previously saved heads for this concept.

Programmatic use:

```python
from ConceptMapper.main import ConceptIsolator

isolator = ConceptIsolator(target_concept="blue", steering_mode="targeted", top_k=12)
direction, results = isolator.isolate_concept()
```

### Activation Patching (Modular)

```python
from ConceptMapper.patching import NodeSpec, make_last_token_logit_diff_metric, measure_node_influence
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen3-4B-Instruct-2507"
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype="auto", device_map="auto")

target = "blue"
negatives = ["red","green","yellow","purple","orange","brown"]
metric = make_last_token_logit_diff_metric(tok, target, negatives, token_index_strategy="auto")

# Aligned clean/corrupt prompts for causal testing
clean = ["The color is"]
corrupt = ["The color is"]

# Candidate nodes to test (e.g., mid/late layers)
nodes = [NodeSpec("attn", l) for l in range(16, 28)] + [NodeSpec("mlp", l) for l in range(16, 28)]

influences = measure_node_influence(model, tok, clean, corrupt, nodes, metric)
print(influences[:10])  # top 10 nodes by influence
```

CLI runner to perform scan + greedy ACDC and save a JSON report:

```bash
python -m ConceptMapper.cli_patching --target blue --use-default-pairs --layers 16:28 --components attn,mlp --topk-influence 20 --eps 1e-3 --max-steps 20
```

Options:
- `--pairs-json`: provide a JSON file with `{"clean": [...], "corrupt": [...]}` for aligned datasets.
- `--use-default-pairs`: use built-in cue templates (good for blue); default if none provided.
- `--no-acdc`: only run the influence scan.
- `--out`: file path to save the report (defaults under `ConceptMapper/reports/`).

### ACDC (Simplified Greedy Circuit Discovery)

```python
from ConceptMapper.patching import NodeSpec, make_last_token_logit_diff_metric
from ConceptMapper.acdc import greedy_acdc
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen3-4B-Instruct-2507"
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype="auto", device_map="auto")

target = "blue"; negatives = ["red","green","yellow","purple","orange","brown"]
metric = make_last_token_logit_diff_metric(tok, target, negatives)

clean = ["The color is", "Clearly it's", "I'd say it's"]
corrupt = ["The color is", "Clearly it's", "I'd say it's"]

candidates = [NodeSpec("attn", l) for l in range(16, 28)] + [NodeSpec("mlp", l) for l in range(16, 28)]
res = greedy_acdc(model, tok, clean, corrupt, candidates, metric, eps=1e-3, max_steps=20)
print("Selected nodes:", res.selected)
print("History:", res.history)
```

Notes:
- The above examples use identical clean/corrupt skeletons for color cloze; you can substitute genuinely contrastive pairs for your concept/task.
- NodeSpec supports `attn` and `mlp`; extend as needed for other components.

## Project Structure

- `main.py`: Core implementation + CLI for concept isolation and steering
- `patching.py`: Activation patching primitives (record, patch, scoring, node influence)
- `acdc.py`: Simplified greedy ACDC over nodes (attn/mlp)
- `README.md`: Project documentation
- `.gitignore`: Git ignore rules

## License

MIT License
