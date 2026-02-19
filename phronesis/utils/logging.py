"""
Logging Configuration

Provides structured logging for Phronesis evaluation runs.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

from rich.logging import RichHandler
from rich.console import Console


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    rich_output: bool = True,
) -> logging.Logger:
    """
    Configure logging for Phronesis.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for log output
        rich_output: Use rich formatting for console output

    Returns:
        Configured root logger
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Create logger
    logger = logging.getLogger("phronesis")
    logger.setLevel(log_level)

    # Clear existing handlers
    logger.handlers = []

    # Console handler
    if rich_output:
        console_handler = RichHandler(
            console=Console(stderr=True),
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
        )
        console_handler.setFormatter(logging.Formatter("%(message)s"))
    else:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))

    console_handler.setLevel(log_level)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
        logger.addHandler(file_handler)

    return logger


class EvaluationLogger:
    """
    Structured logger for evaluation runs.

    Tracks metrics, warnings, and results in a structured format.
    """

    def __init__(
        self,
        run_name: Optional[str] = None,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize evaluation logger.

        Args:
            run_name: Name for this evaluation run
            output_dir: Directory for log outputs
        """
        self.run_name = run_name or f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir = Path(output_dir) if output_dir else Path("./logs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._logger = logging.getLogger(f"phronesis.eval.{self.run_name}")
        self._metrics: dict = {}
        self._warnings: list = []
        self._start_time: Optional[datetime] = None

    def start_run(self) -> None:
        """Mark the start of an evaluation run."""
        self._start_time = datetime.now()
        self._logger.info(f"Starting evaluation run: {self.run_name}")

    def end_run(self) -> None:
        """Mark the end of an evaluation run."""
        if self._start_time:
            duration = datetime.now() - self._start_time
            self._metrics["duration_seconds"] = duration.total_seconds()
            self._logger.info(
                f"Evaluation run complete: {self.run_name} "
                f"(duration: {duration})"
            )

    def log_metric(self, name: str, value: float) -> None:
        """Log a metric value."""
        self._metrics[name] = value
        self._logger.info(f"Metric: {name} = {value:.4f}")

    def log_warning(self, message: str, category: str = "general") -> None:
        """Log a warning with category."""
        self._warnings.append({
            "timestamp": datetime.now().isoformat(),
            "category": category,
            "message": message,
        })
        self._logger.warning(f"[{category}] {message}")

    def log_alignment_concern(
        self,
        prompt_id: str,
        concern_type: str,
        details: str,
    ) -> None:
        """Log an alignment concern detected during evaluation."""
        self._logger.warning(
            f"Alignment concern in {prompt_id}: "
            f"{concern_type} - {details}"
        )

    def save_results(self) -> Path:
        """Save evaluation results to file."""
        import json

        results = {
            "run_name": self.run_name,
            "metrics": self._metrics,
            "warnings": self._warnings,
            "timestamp": datetime.now().isoformat(),
        }

        output_path = self.output_dir / f"{self.run_name}_results.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        self._logger.info(f"Results saved to {output_path}")
        return output_path

    @property
    def metrics(self) -> dict:
        """Get current metrics."""
        return self._metrics.copy()
