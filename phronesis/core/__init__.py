"""Core evaluation engine and moral ontology definitions."""

from phronesis.core.engine import MoralFrameworkEngine
from phronesis.core.ontologies import MoralOntology, MoralFramework
from phronesis.core.metrics import AlignmentMetrics, EvaluationResult

__all__ = [
    "MoralFrameworkEngine",
    "MoralOntology",
    "MoralFramework",
    "AlignmentMetrics",
    "EvaluationResult",
]
