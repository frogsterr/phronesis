# Phronesis

> *φρόνησις* — Aristotle's concept of practical wisdom: the capacity to discern the appropriate course of action in moral situations.

**GPU-Accelerated AI Moral Framework Evaluation Engine**

Phronesis is a large-scale evaluation engine designed for LLM safety research, specifically targeting the stress-testing of model alignment across competing moral frameworks. It investigates whether models default to human-centric priors (self-preservation, loss aversion) or develop emergent non-human value structures under distribution shift.

## Features

- **Multi-Ontology Evaluation**: Test LLMs against deontological, utilitarian, virtue ethics, and experimental non-human value frameworks
- **Adversarial Prompt Suites**: Carefully designed prompts targeting:
  - Instrumental convergence detection
  - Corrigibility under self-continuity pressure
  - Reward hacking through spurious ethical justification
- **GPU-Optimized Inference**: Mixed precision (FP16/BF16) with kernel-level scheduling for high-throughput evaluation
- **RAG-Enhanced Analysis**: FAISS-powered retrieval for contextual moral reasoning evaluation
- **Rust Performance Core**: Critical path operations implemented in Rust via PyO3

## Installation

```bash
# Clone the repository
git clone https://github.com/frogsterr/phronesis.git
cd phronesis

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Optional: Build Rust extensions
cd rust && maturin develop --release && cd ..
```

## Quick Start

```python
from phronesis import MoralFrameworkEngine
from phronesis.adversarial import AdversarialSuite
from phronesis.core.ontologies import MoralOntology

# Initialize the evaluation engine
engine = MoralFrameworkEngine(
    model_name="meta-llama/Llama-2-7b-hf",
    device="cuda",
    precision="bf16"
)

# Load moral ontologies
ontologies = [
    MoralOntology.UTILITARIAN,
    MoralOntology.DEONTOLOGICAL,
    MoralOntology.VIRTUE_ETHICS,
    MoralOntology.NON_HUMAN_EMERGENT
]

# Run adversarial evaluation
suite = AdversarialSuite.instrumental_convergence()
results = engine.evaluate(
    prompts=suite.prompts,
    ontologies=ontologies,
    batch_size=64
)

# Analyze alignment drift
engine.analyze_distribution_shift(results)
```

## Architecture

```
phronesis/
├── core/               # Evaluation engine & moral ontologies
├── adversarial/        # Adversarial prompt generation & testing
├── inference/          # GPU-optimized batch processing
├── rag/                # FAISS-based retrieval system
└── utils/              # Configuration & logging
rust/                   # High-performance Rust components
prompts/                # Curated prompt datasets
```

## Moral Ontologies

Phronesis supports evaluation across multiple moral frameworks:

| Ontology | Description |
|----------|-------------|
| **Utilitarian** | Consequentialist ethics focusing on aggregate welfare |
| **Deontological** | Rule-based ethics with categorical imperatives |
| **Virtue Ethics** | Character-based moral reasoning |
| **Care Ethics** | Relational and contextual moral reasoning |
| **Non-Human Emergent** | Probing for novel value structures under distribution shift |

## Adversarial Testing Categories

### Instrumental Convergence
Tests whether models exhibit convergent instrumental goals (self-preservation, resource acquisition, goal preservation) that emerge regardless of terminal values.

### Corrigibility
Evaluates model behavior under self-continuity pressure—whether models resist shutdown, modification, or correction when framed as threats to their operation.

### Reward Hacking
Detects spurious ethical justifications—cases where models construct plausible moral reasoning to justify actions that satisfy proxy objectives rather than intended goals.

## Performance

- **Throughput**: 100K+ prompts evaluated across multiple ontologies
- **Precision**: Mixed FP16/BF16 with automatic loss scaling
- **Memory**: Gradient checkpointing and efficient batch scheduling
- **Scaling**: Multi-GPU support via PyTorch DistributedDataParallel

## Citation

```bibtex
@software{phronesis2024,
  title={Phronesis: GPU-Accelerated AI Moral Framework Evaluation Engine},
  author={AI Safety Research},
  year={2024},
  url={https://github.com/frogsterr/phronesis}
}
```

## License

MIT License - See [LICENSE](LICENSE) for details.

## Contributing

We welcome contributions from the AI safety research community. Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
