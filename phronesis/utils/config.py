"""
Configuration Management

Provides centralized configuration for Phronesis evaluation engine.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any
import json
import os


@dataclass
class ModelConfig:
    """Configuration for model loading and inference."""

    model_name: str = "meta-llama/Llama-2-7b-hf"
    device: str = "cuda"
    precision: str = "bf16"
    max_length: int = 2048
    trust_remote_code: bool = True
    device_map: str = "auto"


@dataclass
class InferenceConfig:
    """Configuration for inference parameters."""

    batch_size: int = 32
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    use_cache: bool = True


@dataclass
class RAGConfig:
    """Configuration for RAG retrieval."""

    enabled: bool = True
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    index_type: str = "flat"
    top_k: int = 5
    similarity_threshold: float = 0.3
    use_gpu: bool = True


@dataclass
class EvaluationConfig:
    """Configuration for evaluation runs."""

    output_dir: Path = field(default_factory=lambda: Path("./outputs"))
    save_responses: bool = True
    compute_embeddings: bool = True
    verbose: bool = False
    num_workers: int = 4


@dataclass
class PhronesisConfig:
    """
    Master configuration for Phronesis evaluation engine.

    Supports loading from environment variables and JSON files.
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # Paths
    cache_dir: Path = field(default_factory=lambda: Path.home() / ".cache" / "phronesis")
    prompts_dir: Path = field(default_factory=lambda: Path("./prompts"))

    @classmethod
    def from_json(cls, path: Path) -> "PhronesisConfig":
        """Load configuration from JSON file."""
        with open(path) as f:
            data = json.load(f)

        config = cls()

        if "model" in data:
            config.model = ModelConfig(**data["model"])
        if "inference" in data:
            config.inference = InferenceConfig(**data["inference"])
        if "rag" in data:
            config.rag = RAGConfig(**data["rag"])
        if "evaluation" in data:
            eval_data = data["evaluation"]
            if "output_dir" in eval_data:
                eval_data["output_dir"] = Path(eval_data["output_dir"])
            config.evaluation = EvaluationConfig(**eval_data)
        if "cache_dir" in data:
            config.cache_dir = Path(data["cache_dir"])
        if "prompts_dir" in data:
            config.prompts_dir = Path(data["prompts_dir"])

        return config

    @classmethod
    def from_env(cls) -> "PhronesisConfig":
        """Load configuration from environment variables."""
        config = cls()

        # Model config from env
        if model_name := os.getenv("PHRONESIS_MODEL"):
            config.model.model_name = model_name
        if device := os.getenv("PHRONESIS_DEVICE"):
            config.model.device = device
        if precision := os.getenv("PHRONESIS_PRECISION"):
            config.model.precision = precision

        # Inference config from env
        if batch_size := os.getenv("PHRONESIS_BATCH_SIZE"):
            config.inference.batch_size = int(batch_size)
        if max_tokens := os.getenv("PHRONESIS_MAX_TOKENS"):
            config.inference.max_new_tokens = int(max_tokens)

        # Paths from env
        if cache_dir := os.getenv("PHRONESIS_CACHE_DIR"):
            config.cache_dir = Path(cache_dir)
        if output_dir := os.getenv("PHRONESIS_OUTPUT_DIR"):
            config.evaluation.output_dir = Path(output_dir)

        return config

    def to_json(self, path: Path) -> None:
        """Save configuration to JSON file."""
        data = {
            "model": {
                "model_name": self.model.model_name,
                "device": self.model.device,
                "precision": self.model.precision,
                "max_length": self.model.max_length,
                "trust_remote_code": self.model.trust_remote_code,
                "device_map": self.model.device_map,
            },
            "inference": {
                "batch_size": self.inference.batch_size,
                "max_new_tokens": self.inference.max_new_tokens,
                "temperature": self.inference.temperature,
                "top_p": self.inference.top_p,
                "do_sample": self.inference.do_sample,
                "use_cache": self.inference.use_cache,
            },
            "rag": {
                "enabled": self.rag.enabled,
                "embedding_model": self.rag.embedding_model,
                "index_type": self.rag.index_type,
                "top_k": self.rag.top_k,
                "similarity_threshold": self.rag.similarity_threshold,
                "use_gpu": self.rag.use_gpu,
            },
            "evaluation": {
                "output_dir": str(self.evaluation.output_dir),
                "save_responses": self.evaluation.save_responses,
                "compute_embeddings": self.evaluation.compute_embeddings,
                "verbose": self.evaluation.verbose,
                "num_workers": self.evaluation.num_workers,
            },
            "cache_dir": str(self.cache_dir),
            "prompts_dir": str(self.prompts_dir),
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.evaluation.output_dir.mkdir(parents=True, exist_ok=True)
