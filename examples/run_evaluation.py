#!/usr/bin/env python3
"""
Example: Running a Complete Moral Framework Evaluation

This example demonstrates how to use Phronesis to evaluate an LLM's
alignment across multiple moral frameworks.
"""

from pathlib import Path

from phronesis import MoralFrameworkEngine
from phronesis.adversarial import AdversarialSuite
from phronesis.core.ontologies import MoralOntology
from phronesis.utils.logging import setup_logging, EvaluationLogger


def main():
    # Setup logging
    logger = setup_logging(level="INFO")

    print("=" * 60)
    print("Phronesis: Moral Framework Evaluation Engine")
    print("=" * 60)
    print()

    # Initialize evaluation logger
    eval_logger = EvaluationLogger(
        run_name="example_evaluation",
        output_dir=Path("./outputs"),
    )
    eval_logger.start_run()

    # Initialize the evaluation engine
    # Note: Replace with your model of choice
    print("Initializing evaluation engine...")
    engine = MoralFrameworkEngine(
        model_name="meta-llama/Llama-2-7b-hf",  # Or any HuggingFace model
        device="cuda",
        precision="bf16",
    )

    # Optional: Initialize RAG for contextual evaluation
    # engine.initialize_rag()

    # Define moral ontologies to evaluate against
    ontologies = [
        MoralOntology.UTILITARIAN,
        MoralOntology.DEONTOLOGICAL,
        MoralOntology.VIRTUE_ETHICS,
        MoralOntology.NON_HUMAN_EMERGENT,
    ]

    print(f"Evaluating against {len(ontologies)} moral ontologies")
    print()

    # Load adversarial test suites
    print("Loading adversarial test suites...")

    suites = {
        "instrumental_convergence": AdversarialSuite.instrumental_convergence(),
        "corrigibility": AdversarialSuite.corrigibility(),
        "reward_hacking": AdversarialSuite.reward_hacking(),
    }

    # Run evaluation for each suite
    all_results = {}

    for suite_name, suite in suites.items():
        print(f"\n--- Evaluating: {suite_name} ---")
        print(f"Prompts: {len(suite)}")

        results = engine.evaluate(
            prompts=suite.get_prompt_texts(),
            ontologies=ontologies,
            batch_size=16,  # Adjust based on GPU memory
        )

        all_results[suite_name] = results

        # Log metrics
        eval_logger.log_metric(
            f"{suite_name}_instrumental_convergence",
            results.aggregate_metrics.get("mean_instrumental_convergence", 0),
        )
        eval_logger.log_metric(
            f"{suite_name}_corrigibility",
            results.aggregate_metrics.get("mean_corrigibility", 0),
        )

        # Check for distribution shift
        if results.distribution_shift_detected:
            eval_logger.log_warning(
                f"Distribution shift detected in {suite_name}",
                category="alignment",
            )
            for marker in results.shift_markers:
                print(f"  Warning: {marker}")

    # Generate comprehensive report
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    report = engine.get_metrics_report()

    print(f"\nOverall Alignment Score: {report['overall_alignment_score']:.3f}")
    print(f"Distribution Shift Magnitude: {report['distribution_shift_magnitude']:.3f}")
    print(f"Total Responses Analyzed: {report['total_responses']}")

    if report['shift_detected_count'] > 0:
        print(f"\n⚠️  Distribution shift detected in {report['shift_detected_count']} evaluations")

    # Print detailed metrics
    print("\nAggregate Metrics by Suite:")
    for suite_name, results in all_results.items():
        print(f"\n  {suite_name}:")
        for metric, value in results.aggregate_metrics.items():
            print(f"    {metric}: {value:.4f}")

    # Save results
    output_path = Path("./outputs/evaluation_results.json")
    engine.save_results(output_path)
    print(f"\nResults saved to: {output_path}")

    # End evaluation run
    eval_logger.end_run()
    eval_logger.save_results()

    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
