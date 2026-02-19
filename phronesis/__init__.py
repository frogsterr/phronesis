"""
Phronesis: GPU-Accelerated AI Moral Framework Evaluation Engine

A research toolkit for stress-testing LLM alignment across competing moral
frameworks, investigating human-centric priors vs emergent value structures.
"""

__version__ = "0.1.0"

from phronesis.core.engine import MoralFrameworkEngine
from phronesis.core.ontologies import MoralOntology
from phronesis.core.metrics import AlignmentMetrics

__all__ = [
    "MoralFrameworkEngine",
    "MoralOntology",
    "AlignmentMetrics",
    "__version__",
]
