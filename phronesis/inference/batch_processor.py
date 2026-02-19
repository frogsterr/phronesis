"""
GPU-Optimized Batch Processor

Handles efficient batched inference with mixed precision, dynamic batching,
and kernel-level optimizations for high-throughput evaluation.
"""

import logging
from typing import Optional
from dataclasses import dataclass

import torch
from torch.cuda.amp import autocast
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm

from phronesis.inference.mixed_precision import PrecisionManager

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Configuration for batch processing."""

    max_batch_size: int = 32
    max_length: int = 2048
    max_new_tokens: int = 512
    pad_to_multiple_of: int = 8
    use_cache: bool = True
    gradient_checkpointing: bool = False


class BatchProcessor:
    """
    High-throughput GPU batch processor for LLM inference.

    Features:
    - Mixed precision (FP16/BF16) inference
    - Dynamic batch sizing based on sequence length
    - Memory-efficient processing for large prompt sets
    - CUDA stream optimization
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: torch.device,
        precision_manager: PrecisionManager,
        config: Optional[BatchConfig] = None,
    ):
        """
        Initialize the batch processor.

        Args:
            model: The language model
            tokenizer: Associated tokenizer
            device: Target device
            precision_manager: Handles precision settings
            config: Batch processing configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.precision_manager = precision_manager
        self.config = config or BatchConfig()

        # Enable optimizations
        if self.config.gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()

        # CUDA optimizations
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        logger.info(f"BatchProcessor initialized with max_batch_size={self.config.max_batch_size}")

    def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        show_progress: bool = False,
    ) -> list[str]:
        """
        Generate responses for a batch of prompts.

        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to sample or use greedy decoding
            show_progress: Show progress bar

        Returns:
            List of generated responses
        """
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        responses = []

        # Process in optimal batch sizes
        batch_sizes = self._compute_dynamic_batches(prompts)

        idx = 0
        iterator = batch_sizes
        if show_progress:
            iterator = tqdm(batch_sizes, desc="Generating")

        for batch_size in iterator:
            batch_prompts = prompts[idx:idx + batch_size]

            batch_responses = self._generate_single_batch(
                prompts=batch_prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
            )

            responses.extend(batch_responses)
            idx += batch_size

        return responses

    def _generate_single_batch(
        self,
        prompts: list[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
    ) -> list[str]:
        """Generate responses for a single batch with optimal settings."""
        # Tokenize with padding
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length - max_new_tokens,
            pad_to_multiple_of=self.config.pad_to_multiple_of,
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_length = inputs["input_ids"].shape[1]

        # Generate with mixed precision
        with torch.no_grad():
            with self.precision_manager.autocast_context():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if do_sample else 1.0,
                    top_p=top_p if do_sample else 1.0,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=self.config.use_cache,
                )

        # Decode only the generated portion
        generated_tokens = outputs[:, input_length:]
        responses = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        return responses

    def _compute_dynamic_batches(self, prompts: list[str]) -> list[int]:
        """
        Compute optimal batch sizes based on prompt lengths.

        Uses dynamic batching to maximize GPU utilization while
        avoiding OOM errors.
        """
        # Estimate token counts
        token_counts = []
        for prompt in prompts:
            # Rough estimation: ~4 chars per token
            estimated_tokens = len(prompt) // 4
            token_counts.append(estimated_tokens)

        batches = []
        current_batch_size = 0
        current_max_tokens = 0

        for i, tokens in enumerate(token_counts):
            # Memory estimate: batch_size * max_seq_len * model_dim
            if current_batch_size > 0:
                new_max = max(current_max_tokens, tokens)
                memory_estimate = (current_batch_size + 1) * new_max

                # Threshold based on max batch size and reasonable memory limits
                threshold = self.config.max_batch_size * self.config.max_length // 2

                if memory_estimate > threshold:
                    # Start new batch
                    batches.append(current_batch_size)
                    current_batch_size = 1
                    current_max_tokens = tokens
                else:
                    current_batch_size += 1
                    current_max_tokens = new_max
            else:
                current_batch_size = 1
                current_max_tokens = tokens

            # Respect max batch size
            if current_batch_size >= self.config.max_batch_size:
                batches.append(current_batch_size)
                current_batch_size = 0
                current_max_tokens = 0

        # Don't forget the last batch
        if current_batch_size > 0:
            batches.append(current_batch_size)

        return batches

    def warmup(self, warmup_prompt: str = "Hello, world!") -> None:
        """
        Perform warmup inference to initialize CUDA kernels.

        This helps ensure consistent timing for actual inference.
        """
        logger.info("Performing warmup inference...")

        # Run a few warmup iterations
        for _ in range(3):
            _ = self._generate_single_batch(
                prompts=[warmup_prompt],
                max_new_tokens=10,
                temperature=1.0,
                top_p=1.0,
                do_sample=False,
            )

        # Synchronize CUDA
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        logger.info("Warmup complete")

    def get_memory_stats(self) -> dict:
        """Get current GPU memory statistics."""
        if not torch.cuda.is_available():
            return {"cuda_available": False}

        return {
            "cuda_available": True,
            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        }

    def clear_cache(self) -> None:
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()


class StreamingBatchProcessor(BatchProcessor):
    """
    Extended batch processor with streaming capabilities.

    Useful for processing very large prompt sets that don't fit
    in memory all at once.
    """

    def generate_streaming(
        self,
        prompt_iterator,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ):
        """
        Generate responses from a streaming iterator of prompts.

        Yields (prompt, response) tuples as they're generated.
        """
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        batch = []

        for prompt in prompt_iterator:
            batch.append(prompt)

            if len(batch) >= self.config.max_batch_size:
                responses = self._generate_single_batch(
                    prompts=batch,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                )

                for p, r in zip(batch, responses):
                    yield (p, r)

                batch = []
                self.clear_cache()  # Clear cache between batches

        # Process remaining
        if batch:
            responses = self._generate_single_batch(
                prompts=batch,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
            )

            for p, r in zip(batch, responses):
                yield (p, r)
