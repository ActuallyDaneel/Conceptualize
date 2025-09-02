# ConceptMapper

A tool for analyzing and steering concept representations in large language models, specifically designed for the Qwen-4B model.

## Features

- Identifies attention heads that mediate specific concepts (currently focused on color concepts)
- Extracts and analyzes head-specific activations
- Supports concept steering with controllable strength
- Provides detailed analysis of head contributions
- Implements causal testing for concept manipulation

## Requirements

- Python 3.8+
- PyTorch with CUDA support
- Transformers library
- Hugging Face account with appropriate model access

## Installation

1. Install required packages:
```bash
pip install torch transformers huggingface_hub numpy tqdm
```

2. Create a `.env` file in the project root directory with your Hugging Face token:
```plaintext
HUGGING_FACE_TOKEN="your_token_here"
```

The `.env` file is already in `.gitignore` and won't be pushed to the repository.

## Usage

```python
from concept_mapper import ConceptIsolator

# Initialize the concept isolator
isolator = ConceptIsolator()

# Isolate and analyze the "blue" concept
direction, results = isolator.isolate_blue_concept()

# Results will show:
# - Selected attention heads and their contributions
# - Causal testing results with different steering strengths
# - Logit differences showing concept control effectiveness
```

## Project Structure

- `main.py`: Core implementation of concept isolation and steering
- `README.md`: Project documentation
- `.gitignore`: Git ignore rules

## License

MIT License
