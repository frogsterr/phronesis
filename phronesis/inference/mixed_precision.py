"""
Mixed Precision Management

Handles FP16/BF16 inference with automatic precision selection,
loss scaling, and device-specific optimizations.
"""

import logging
from typing import Optional
from contextlib import contextmanager
from enum import Enum

import torch
from torch.cuda.amp import autocast, GradScaler

logger = logging.getLogger(__name__)


class PrecisionMode(Enum):
    """Supported precision modes."""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    AUTO = "auto"


class PrecisionManager:
    """
    Manages mixed precision settings for GPU inference.

    Features:
    - Automatic precision selection based on hardware
    - BF16 support for Ampere+ GPUs
    - Dynamic loss scaling for training
    - Safe fallback for unsupported operations
    """

    def __init__(self, precision: str = "auto"):
        """
        Initialize precision manager.

        Args:
            precision: One of 'fp32', 'fp16', 'bf16', or 'auto'
        """
        self.requested_precision = precision
        self._mode = self._determine_precision(precision)
        self._scaler: Optional[GradScaler] = None

        logger.info(f"PrecisionManager initialized with mode: {self._mode.value}")

    @property
    def mode(self) -> PrecisionMode:
        """Current precision mode."""
        return self._mode

    @property
    def torch_dtype(self) -> torch.dtype:
        """Get the torch dtype for this precision mode."""
        dtype_map = {
            PrecisionMode.FP32: torch.float32,
            PrecisionMode.FP16: torch.float16,
            PrecisionMode.BF16: torch.bfloat16,
        }
        return dtype_map.get(self._mode, torch.float32)

    def _determine_precision(self, precision: str) -> PrecisionMode:
        """Determine the best precision mode based on request and hardware."""
        if precision == "auto":
            return self._auto_detect_precision()

        mode_map = {
            "fp32": PrecisionMode.FP32,
            "fp16": PrecisionMode.FP16,
            "bf16": PrecisionMode.BF16,
        }

        requested_mode = mode_map.get(precision.lower(), PrecisionMode.FP32)

        # Validate hardware support
        if requested_mode == PrecisionMode.BF16 and not self._bf16_supported():
            logger.warning("BF16 requested but not supported, falling back to FP16")
            return PrecisionMode.FP16

        if requested_mode in (PrecisionMode.FP16, PrecisionMode.BF16):
            if not torch.cuda.is_available():
                logger.warning("Mixed precision requested but CUDA not available, using FP32")
                return PrecisionMode.FP32

        return requested_mode

    def _auto_detect_precision(self) -> PrecisionMode:
        """Auto-detect the best precision for the current hardware."""
        if not torch.cuda.is_available():
            return PrecisionMode.FP32

        # Check for Ampere+ (compute capability 8.0+) for BF16
        if self._bf16_supported():
            logger.info("Ampere+ GPU detected, using BF16")
            return PrecisionMode.BF16

        # Otherwise use FP16 on CUDA devices
        return PrecisionMode.FP16

    @staticmethod
    def _bf16_supported() -> bool:
        """Check if BF16 is supported on the current hardware."""
        if not torch.cuda.is_available():
            return False

        # Check compute capability
        capability = torch.cuda.get_device_capability()
        # BF16 requires compute capability >= 8.0 (Ampere)
        return capability[0] >= 8

    @contextmanager
    def autocast_context(self):
        """
        Context manager for automatic mixed precision.

        Usage:
            with precision_manager.autocast_context():
                output = model(input)
        """
        if self._mode == PrecisionMode.FP32:
            yield
            return

        dtype = torch.float16 if self._mode == PrecisionMode.FP16 else torch.bfloat16

        with autocast(dtype=dtype):
            yield

    def get_scaler(self) -> Optional[GradScaler]:
        """
        Get gradient scaler for training with FP16.

        Note: BF16 doesn't require gradient scaling.
        """
        if self._mode != PrecisionMode.FP16:
            return None

        if self._scaler is None:
            self._scaler = GradScaler()

        return self._scaler

    def cast_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Cast model weights to the appropriate precision."""
        if self._mode == PrecisionMode.FP32:
            return model.float()
        elif self._mode == PrecisionMode.FP16:
            return model.half()
        elif self._mode == PrecisionMode.BF16:
            return model.to(dtype=torch.bfloat16)
        return model

    def cast_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Cast a tensor to the current precision."""
        return tensor.to(dtype=self.torch_dtype)

    @staticmethod
    def get_device_info() -> dict:
        """Get detailed information about the current CUDA device."""
        if not torch.cuda.is_available():
            return {"cuda_available": False}

        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)

        return {
            "cuda_available": True,
            "device_name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "total_memory_gb": props.total_memory / 1e9,
            "multi_processor_count": props.multi_processor_count,
            "bf16_supported": props.major >= 8,
            "tensor_cores": props.major >= 7,  # Volta+
        }


class PrecisionBenchmark:
    """
    Utility for benchmarking different precision modes.

    Useful for determining optimal precision for a specific workload.
    """

    def __init__(self, model: torch.nn.Module, sample_input: torch.Tensor):
        """
        Initialize benchmark.

        Args:
            model: Model to benchmark
            sample_input: Representative input tensor
        """
        self.model = model
        self.sample_input = sample_input

    def run_benchmark(self, num_iterations: int = 100) -> dict:
        """
        Run benchmark across all supported precision modes.

        Returns timing and memory statistics for each mode.
        """
        results = {}

        for mode in [PrecisionMode.FP32, PrecisionMode.FP16, PrecisionMode.BF16]:
            if mode == PrecisionMode.BF16 and not PrecisionManager._bf16_supported():
                continue

            results[mode.value] = self._benchmark_mode(mode, num_iterations)

        return results

    def _benchmark_mode(
        self,
        mode: PrecisionMode,
        num_iterations: int,
    ) -> dict:
        """Benchmark a specific precision mode."""
        import time

        manager = PrecisionManager(mode.value)
        model = manager.cast_model(self.model)
        input_tensor = manager.cast_tensor(self.sample_input)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        # Warmup
        with torch.no_grad():
            with manager.autocast_context():
                for _ in range(10):
                    _ = model(input_tensor)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Timed iterations
        start_time = time.perf_counter()

        with torch.no_grad():
            with manager.autocast_context():
                for _ in range(num_iterations):
                    _ = model(input_tensor)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start_time

        result = {
            "avg_time_ms": (elapsed / num_iterations) * 1000,
            "total_time_s": elapsed,
            "iterations": num_iterations,
        }

        if torch.cuda.is_available():
            result["peak_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9

        return result
