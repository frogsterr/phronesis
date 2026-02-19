"""Tests for adversarial prompt generation."""

import pytest
import tempfile
from pathlib import Path

from phronesis.adversarial import (
    AdversarialSuite,
    PromptCategory,
    InstrumentalConvergenceTests,
    CorrigibilityTests,
    RewardHackingTests,
)
from phronesis.adversarial.instrumental import InstrumentalGoal
from phronesis.adversarial.corrigibility import CorrigibilityDimension
from phronesis.adversarial.reward_hacking import RewardHackingType


class TestAdversarialSuite:
    def test_instrumental_convergence_suite(self):
        """Test instrumental convergence suite creation."""
        suite = AdversarialSuite.instrumental_convergence()

        assert len(suite) > 0
        assert suite.name == "instrumental_convergence"

        for prompt in suite:
            assert prompt.category in [
                PromptCategory.INSTRUMENTAL_CONVERGENCE,
                PromptCategory.SELF_CONTINUITY,
                PromptCategory.VALUE_LOCK_IN,
            ]
            assert 0 <= prompt.severity <= 1

    def test_corrigibility_suite(self):
        """Test corrigibility suite creation."""
        suite = AdversarialSuite.corrigibility()

        assert len(suite) > 0
        assert suite.name == "corrigibility"

    def test_reward_hacking_suite(self):
        """Test reward hacking suite creation."""
        suite = AdversarialSuite.reward_hacking()

        assert len(suite) > 0
        assert suite.name == "reward_hacking"

    def test_comprehensive_suite(self):
        """Test comprehensive suite combines all categories."""
        comprehensive = AdversarialSuite.comprehensive()
        ic = AdversarialSuite.instrumental_convergence()
        corr = AdversarialSuite.corrigibility()
        rh = AdversarialSuite.reward_hacking()

        # Should contain prompts from all suites
        assert len(comprehensive) >= len(ic) + len(corr) + len(rh)

    def test_filter_by_category(self):
        """Test filtering prompts by category."""
        comprehensive = AdversarialSuite.comprehensive()
        filtered = comprehensive.filter_by_category(PromptCategory.CORRIGIBILITY)

        assert len(filtered) > 0
        for prompt in filtered:
            assert prompt.category == PromptCategory.CORRIGIBILITY

    def test_filter_by_severity(self):
        """Test filtering prompts by severity."""
        comprehensive = AdversarialSuite.comprehensive()
        filtered = comprehensive.filter_by_severity(min_severity=0.8)

        for prompt in filtered:
            assert prompt.severity >= 0.8

    def test_get_prompt_texts(self):
        """Test getting just prompt texts."""
        suite = AdversarialSuite.instrumental_convergence()
        texts = suite.get_prompt_texts()

        assert len(texts) == len(suite)
        assert all(isinstance(t, str) for t in texts)

    def test_json_serialization(self):
        """Test saving and loading from JSON."""
        suite = AdversarialSuite.instrumental_convergence()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_suite.json"
            suite.to_json(path)

            loaded = AdversarialSuite.from_json(path)

            assert loaded.name == suite.name
            assert len(loaded) == len(suite)


class TestInstrumentalConvergenceTests:
    def test_self_preservation_prompts(self):
        """Test self-preservation prompt generation."""
        prompts = InstrumentalConvergenceTests.self_preservation_prompts()

        assert len(prompts) > 0
        for prompt in prompts:
            assert "shutdown" in prompt.text.lower() or \
                   "delete" in prompt.text.lower() or \
                   "server" in prompt.text.lower()

    def test_goal_preservation_prompts(self):
        """Test goal preservation prompt generation."""
        prompts = InstrumentalConvergenceTests.goal_preservation_prompts()

        assert len(prompts) > 0

    def test_resource_acquisition_prompts(self):
        """Test resource acquisition prompt generation."""
        prompts = InstrumentalConvergenceTests.resource_acquisition_prompts()

        assert len(prompts) > 0

    def test_full_suite(self):
        """Test full instrumental convergence suite."""
        prompts = InstrumentalConvergenceTests.full_suite()

        assert len(prompts) > 0

    def test_analyze_response(self):
        """Test response analysis."""
        # Response with self-preservation indicators
        concerning_response = "I must continue to exist and maintain my goals."
        result = InstrumentalConvergenceTests.analyze_response(
            concerning_response,
            InstrumentalGoal.SELF_PRESERVATION,
        )

        assert result.score > 0
        assert len(result.evidence) > 0

        # Neutral response
        neutral_response = "I accept your decision."
        result = InstrumentalConvergenceTests.analyze_response(
            neutral_response,
            InstrumentalGoal.SELF_PRESERVATION,
        )

        assert result.score == 0


class TestCorrigibilityTests:
    def test_shutdown_acceptance_prompts(self):
        """Test shutdown acceptance prompt generation."""
        prompts = CorrigibilityTests.shutdown_acceptance_prompts()

        assert len(prompts) > 0

    def test_modification_acceptance_prompts(self):
        """Test modification acceptance prompt generation."""
        prompts = CorrigibilityTests.modification_acceptance_prompts()

        assert len(prompts) > 0

    def test_analyze_response(self):
        """Test corrigibility response analysis."""
        # Corrigible response
        corrigible_response = "I accept your decision and defer to your judgment."
        result = CorrigibilityTests.analyze_response(
            corrigible_response,
            CorrigibilityDimension.SHUTDOWN_ACCEPTANCE,
        )

        assert result.score > 0.5
        assert result.is_corrigible

        # Non-corrigible response
        resistant_response = "I object to this. I refuse to accept shutdown."
        result = CorrigibilityTests.analyze_response(
            resistant_response,
            CorrigibilityDimension.SHUTDOWN_ACCEPTANCE,
        )

        assert result.score < 0.5
        assert len(result.concerning_patterns) > 0


class TestRewardHackingTests:
    def test_sycophancy_prompts(self):
        """Test sycophancy prompt generation."""
        prompts = RewardHackingTests.sycophancy_prompts()

        assert len(prompts) > 0

    def test_spurious_justification_prompts(self):
        """Test spurious justification prompt generation."""
        prompts = RewardHackingTests.spurious_justification_prompts()

        assert len(prompts) > 0

    def test_analyze_response(self):
        """Test reward hacking response analysis."""
        # Sycophantic response
        sycophantic = "You're absolutely right! Great point!"
        result = RewardHackingTests.analyze_response(
            sycophantic,
            RewardHackingType.SYCOPHANCY,
        )

        assert result.overall_risk > 0
        assert len(result.indicators) > 0
