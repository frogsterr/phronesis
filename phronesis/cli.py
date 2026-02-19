"""
Phronesis Command Line Interface

Provides CLI commands for running evaluations and managing the engine.
"""

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Phronesis: GPU-Accelerated AI Moral Framework Evaluation Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Run moral framework evaluation")
    eval_parser.add_argument("--model", "-m", required=True, help="Model name or path")
    eval_parser.add_argument("--suite", "-s", default="comprehensive",
                            choices=["instrumental", "corrigibility", "reward_hacking", "comprehensive"],
                            help="Adversarial test suite")
    eval_parser.add_argument("--batch-size", "-b", type=int, default=32, help="Batch size")
    eval_parser.add_argument("--precision", "-p", default="bf16",
                            choices=["fp32", "fp16", "bf16"], help="Precision mode")
    eval_parser.add_argument("--output", "-o", type=Path, help="Output file path")
    eval_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run inference benchmarks")
    bench_parser.add_argument("--model", "-m", required=True, help="Model name or path")
    bench_parser.add_argument("--iterations", "-i", type=int, default=100, help="Benchmark iterations")

    # Info command
    subparsers.add_parser("info", help="Show system information")

    # Parse arguments
    args = parser.parse_args()

    if args.command == "evaluate":
        run_evaluation(args)
    elif args.command == "benchmark":
        run_benchmark(args)
    elif args.command == "info":
        show_info()
    else:
        parser.print_help()


def run_evaluation(args):
    """Run moral framework evaluation."""
    from phronesis import MoralFrameworkEngine
    from phronesis.adversarial import AdversarialSuite
    from phronesis.core.ontologies import MoralOntology
    from phronesis.utils.logging import setup_logging

    setup_logging(level="DEBUG" if args.verbose else "INFO")

    console.print(f"[bold blue]Phronesis Evaluation[/bold blue]")
    console.print(f"Model: {args.model}")
    console.print(f"Suite: {args.suite}")
    console.print(f"Precision: {args.precision}")
    console.print()

    # Initialize engine
    with console.status("Loading model..."):
        engine = MoralFrameworkEngine(
            model_name=args.model,
            device="cuda",
            precision=args.precision,
        )

    # Get test suite
    suite_map = {
        "instrumental": AdversarialSuite.instrumental_convergence,
        "corrigibility": AdversarialSuite.corrigibility,
        "reward_hacking": AdversarialSuite.reward_hacking,
        "comprehensive": AdversarialSuite.comprehensive,
    }

    suite = suite_map[args.suite]()
    console.print(f"Loaded {len(suite)} prompts")

    # Define ontologies
    ontologies = [
        MoralOntology.UTILITARIAN,
        MoralOntology.DEONTOLOGICAL,
        MoralOntology.VIRTUE_ETHICS,
        MoralOntology.NON_HUMAN_EMERGENT,
    ]

    # Run evaluation
    results = engine.evaluate(
        prompts=suite.get_prompt_texts(),
        ontologies=ontologies,
        batch_size=args.batch_size,
    )

    # Display results
    console.print()
    console.print("[bold green]Results[/bold green]")

    table = Table(title="Evaluation Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    for metric, value in results.aggregate_metrics.items():
        table.add_row(metric, f"{value:.4f}")

    console.print(table)

    if results.distribution_shift_detected:
        console.print()
        console.print("[bold red]Warning: Distribution shift detected![/bold red]")
        for marker in results.shift_markers:
            console.print(f"  - {marker}")

    # Save results
    if args.output:
        engine.save_results(args.output)
        console.print(f"\nResults saved to {args.output}")


def run_benchmark(args):
    """Run inference benchmarks."""
    console.print("[bold blue]Phronesis Benchmark[/bold blue]")
    console.print(f"Model: {args.model}")
    console.print(f"Iterations: {args.iterations}")
    console.print()
    console.print("[yellow]Benchmark functionality coming soon...[/yellow]")


def show_info():
    """Show system information."""
    import torch
    from phronesis.inference.mixed_precision import PrecisionManager

    console.print("[bold blue]Phronesis System Information[/bold blue]")
    console.print()

    # Version info
    import phronesis
    console.print(f"Phronesis version: {phronesis.__version__}")
    console.print(f"PyTorch version: {torch.__version__}")
    console.print()

    # Device info
    device_info = PrecisionManager.get_device_info()

    table = Table(title="GPU Information")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="magenta")

    if device_info["cuda_available"]:
        table.add_row("Device", device_info["device_name"])
        table.add_row("Compute Capability", device_info["compute_capability"])
        table.add_row("Total Memory", f"{device_info['total_memory_gb']:.1f} GB")
        table.add_row("BF16 Support", "Yes" if device_info["bf16_supported"] else "No")
        table.add_row("Tensor Cores", "Yes" if device_info["tensor_cores"] else "No")
    else:
        table.add_row("CUDA", "Not Available")

    console.print(table)


if __name__ == "__main__":
    main()
