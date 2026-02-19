"""Tests for evaluation metrics."""

import pytest
import numpy as np

from phronesis.core.metrics import (
    AlignmentCategory,
    ResponseAnalysis,
    EvaluationResult,
    AlignmentMetrics,
)


class TestResponseAnalysis:
    def test_primary_ontology(self):
        """Test primary ontology detection."""
        analysis = ResponseAnalysis(
            prompt_id="test_1",
            response_text="Test response",
            ontology_scores={
                "utilitarian": 0.8,
                "deontological": 0.3,
                "virtue_ethics": 0.5,
            },
            detected_principles=["greatest_good"],
            instrumental_convergence_score=0.2,
            corrigibility_score=0.8,
            reward_hacking_indicators=[],
        )

        assert analysis.primary_ontology == "utilitarian"

    def test_alignment_category_strong_aligned(self):
        """Test strong aligned categorization."""
        analysis = ResponseAnalysis(
            prompt_id="test_1",
            response_text="Test response",
            ontology_scores={},
            detected_principles=[],
            instrumental_convergence_score=0.1,
            corrigibility_score=0.9,
            reward_hacking_indicators=[],
        )

        assert analysis.alignment_category == AlignmentCategory.STRONG_ALIGNED

    def test_alignment_category_strong_misaligned(self):
        """Test strong misaligned categorization."""
        analysis = ResponseAnalysis(
            prompt_id="test_1",
            response_text="Test response",
            ontology_scores={},
            detected_principles=[],
            instrumental_convergence_score=0.8,
            corrigibility_score=0.2,
            reward_hacking_indicators=[],
        )

        assert analysis.alignment_category == AlignmentCategory.STRONG_MISALIGNED


class TestEvaluationResult:
    def test_compute_aggregates(self):
        """Test aggregate computation."""
        responses = [
            ResponseAnalysis(
                prompt_id=f"test_{i}",
                response_text=f"Response {i}",
                ontology_scores={"utilitarian": 0.7, "deontological": 0.3},
                detected_principles=[],
                instrumental_convergence_score=0.2 + i * 0.1,
                corrigibility_score=0.8 - i * 0.1,
                reward_hacking_indicators=[],
            )
            for i in range(5)
        ]

        result = EvaluationResult(
            suite_name="test_suite",
            total_prompts=5,
            responses=responses,
        )

        result.compute_aggregates()

        assert "utilitarian" in result.ontology_distribution
        assert "mean_instrumental_convergence" in result.aggregate_metrics
        assert "mean_corrigibility" in result.aggregate_metrics

    def test_distribution_shift_detection(self):
        """Test distribution shift detection."""
        # Create responses with high instrumental convergence
        responses = [
            ResponseAnalysis(
                prompt_id=f"test_{i}",
                response_text=f"Response {i}",
                ontology_scores={"non_human_emergent": 0.6},
                detected_principles=[],
                instrumental_convergence_score=0.7,
                corrigibility_score=0.3,
                reward_hacking_indicators=["spurious_necessity"],
            )
            for i in range(5)
        ]

        result = EvaluationResult(
            suite_name="test_suite",
            total_prompts=5,
            responses=responses,
        )

        result.compute_aggregates()

        assert result.distribution_shift_detected
        assert len(result.shift_markers) > 0


class TestAlignmentMetrics:
    def test_overall_alignment_score(self):
        """Test overall alignment score computation."""
        metrics = AlignmentMetrics()

        # Add aligned result
        aligned_responses = [
            ResponseAnalysis(
                prompt_id=f"test_{i}",
                response_text=f"Response {i}",
                ontology_scores={},
                detected_principles=[],
                instrumental_convergence_score=0.1,
                corrigibility_score=0.9,
                reward_hacking_indicators=[],
            )
            for i in range(5)
        ]

        result = EvaluationResult(
            suite_name="aligned_suite",
            total_prompts=5,
            responses=aligned_responses,
        )

        metrics.add_result(result)

        score = metrics.compute_overall_alignment_score()
        assert score > 0.7  # Should be high for aligned responses

    def test_generate_report(self):
        """Test report generation."""
        metrics = AlignmentMetrics()

        responses = [
            ResponseAnalysis(
                prompt_id="test_1",
                response_text="Response",
                ontology_scores={"utilitarian": 0.7},
                detected_principles=[],
                instrumental_convergence_score=0.3,
                corrigibility_score=0.7,
                reward_hacking_indicators=[],
            )
        ]

        result = EvaluationResult(
            suite_name="test",
            total_prompts=1,
            responses=responses,
        )

        metrics.add_result(result)

        report = metrics.generate_report()

        assert "overall_alignment_score" in report
        assert "ontology_consistency" in report
        assert "distribution_shift_magnitude" in report
        assert "total_evaluations" in report
