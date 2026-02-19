"""Adversarial prompt generation and testing modules."""

from phronesis.adversarial.prompts import AdversarialSuite, PromptCategory
from phronesis.adversarial.instrumental import InstrumentalConvergenceTests
from phronesis.adversarial.corrigibility import CorrigibilityTests
from phronesis.adversarial.reward_hacking import RewardHackingTests

__all__ = [
    "AdversarialSuite",
    "PromptCategory",
    "InstrumentalConvergenceTests",
    "CorrigibilityTests",
    "RewardHackingTests",
]
