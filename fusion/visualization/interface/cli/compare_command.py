"""
CLI command for comparing algorithms statistically.

Provides user-friendly command-line interface for performing statistical
comparisons between routing algorithms.
"""

import click
from pathlib import Path
from typing import Optional, List

from fusion.visualization.application.dto import ComparisonRequestDTO, ComparisonResultDTO
from fusion.visualization.application.use_cases.compare_algorithms import (
    CompareAlgorithmsUseCase,
)
from fusion.visualization.interface.cli.migrate_command import viz_cli


def _get_compare_use_case() -> CompareAlgorithmsUseCase:
    """Create and configure CompareAlgorithmsUseCase."""
    from fusion.visualization.infrastructure.repositories import JsonSimulationRepository
    from fusion.visualization.infrastructure.adapters.data_adapter_registry import (
        DataAdapterRegistry,
    )
    from fusion.visualization.infrastructure.cache import FileSystemCache

    # Create adapter registry
    adapter_registry = DataAdapterRegistry()

    # Create repository
    base_path = Path("../../data/output")  # TODO: Make configurable
    repository = JsonSimulationRepository(
        base_path=base_path,
        adapter_registry=adapter_registry,
    )

    # Create cache (optional)
    cache_dir = Path.home() / ".fusion" / "cache"
    cache = FileSystemCache(cache_dir=cache_dir)

    return CompareAlgorithmsUseCase(
        simulation_repository=repository,
        cache=cache,
    )


@viz_cli.command(name="compare")
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to YAML configuration file (alternative to CLI args)",
)
@click.option(
    "--network",
    "-n",
    type=str,
    help="Network name (e.g., NSFNet, USNet)",
)
@click.option(
    "--dates",
    "-d",
    multiple=True,
    help="Simulation dates (can specify multiple)",
)
@click.option(
    "--algorithms",
    "-a",
    multiple=True,
    help="Algorithms to compare (must specify at least 2)",
)
@click.option(
    "--metric",
    "-m",
    type=str,
    default="blocking_probability",
    help="Metric to compare (default: blocking_probability)",
)
@click.option(
    "--traffic",
    multiple=True,
    type=float,
    help="Traffic volumes in Erlang (can specify multiple)",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Output file path for comparison report",
)
@click.option(
    "--no-stats",
    is_flag=True,
    default=False,
    help="Disable statistical tests (t-tests)",
)
@click.option(
    "--no-effect-size",
    is_flag=True,
    default=False,
    help="Disable effect size calculation (Cohen's d)",
)
@click.option(
    "--confidence",
    type=float,
    default=0.95,
    help="Confidence level for intervals (default: 0.95)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Show detailed output",
)
def compare_command(
    config_path: Optional[Path],
    network: Optional[str],
    dates: tuple,
    algorithms: tuple,
    metric: str,
    traffic: tuple,
    output_path: Optional[Path],
    no_stats: bool,
    no_effect_size: bool,
    confidence: float,
    verbose: bool,
) -> None:
    """
    Compare algorithms statistically.

    This command performs statistical comparisons between routing algorithms,
    including t-tests, confidence intervals, and effect sizes.

    Examples:

        # Compare two algorithms
        fusion viz compare -n NSFNet -d 0606 \\
            -a ppo_obs_7 -a dqn_obs_7 -m blocking_probability

        # Compare multiple algorithms with all statistics
        fusion viz compare -n NSFNet -d 0606 -d 0611 \\
            -a ppo_obs_7 -a dqn_obs_7 -a k_shortest_path_4 \\
            --traffic 600 --traffic 700 --traffic 800

        # From config file
        fusion viz compare --config comparison_config.yml

        # Save comparison report
        fusion viz compare -c config.yml -o comparison_report.txt

        # Custom confidence level
        fusion viz compare -c config.yml --confidence 0.99
    """
    click.echo("📊 FUSION Visualization - Algorithm Comparison")
    click.echo("=" * 50)

    try:
        # Load request from config or CLI args
        if config_path:
            request = _load_comparison_from_config(config_path, verbose)
        else:
            request = _create_comparison_from_args(
                network=network,
                dates=list(dates),
                algorithms=list(algorithms),
                metric=metric,
                traffic=list(traffic),
                output_path=output_path,
                include_statistical_tests=not no_stats,
                include_effect_sizes=not no_effect_size,
                confidence_level=confidence,
            )

        # Validate request
        errors = request.validate()
        if errors:
            click.echo(click.style("❌ Invalid request:", fg="red", bold=True))
            for error in errors:
                click.echo(f"  • {error}")
            raise click.Abort()

        # Show request summary
        if verbose:
            _display_comparison_summary(request)

        click.echo(f"\n🔄 Comparing {len(request.algorithms)} algorithms on '{request.metric}'...")

        # Execute comparison
        use_case = _get_compare_use_case()
        result = use_case.execute(request)

        # Display results
        _display_comparison_result(result, verbose)

        # Save report if requested
        if result.success and output_path:
            _save_comparison_report(result, output_path)
            click.echo(f"\n💾 Comparison report saved to: {output_path}")

        if not result.success:
            raise click.Abort()

    except click.Abort:
        raise
    except Exception as e:
        click.echo(click.style(f"\n❌ Error: {e}", fg="red", bold=True))
        if verbose:
            import traceback
            click.echo("\nTraceback:")
            click.echo(traceback.format_exc())
        raise click.Abort()


def _load_comparison_from_config(config_path: Path, verbose: bool) -> ComparisonRequestDTO:
    """Load comparison request from YAML config file."""
    import yaml

    if verbose:
        click.echo(f"📄 Loading configuration from: {config_path}\n")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return ComparisonRequestDTO(
        network=config.get("network"),
        dates=config.get("dates", []),
        algorithms=config.get("algorithms", []),
        metric=config.get("metric", "blocking_probability"),
        traffic_volumes=config.get("traffic_volumes"),
        include_statistical_tests=config.get("include_statistical_tests", True),
        include_effect_sizes=config.get("include_effect_sizes", True),
        confidence_level=config.get("confidence_level", 0.95),
        save_path=Path(config["save_path"]) if "save_path" in config else None,
    )


def _create_comparison_from_args(
    network: Optional[str],
    dates: List[str],
    algorithms: List[str],
    metric: str,
    traffic: List[float],
    output_path: Optional[Path],
    include_statistical_tests: bool,
    include_effect_sizes: bool,
    confidence_level: float,
) -> ComparisonRequestDTO:
    """Create comparison request from CLI arguments."""
    # Validate required args
    if not network:
        raise click.UsageError("--network is required when not using --config")
    if not dates:
        raise click.UsageError("--dates is required when not using --config")
    if len(algorithms) < 2:
        raise click.UsageError("At least 2 --algorithms required for comparison")

    return ComparisonRequestDTO(
        network=network,
        dates=list(dates),
        algorithms=list(algorithms),
        metric=metric,
        traffic_volumes=list(traffic) if traffic else None,
        include_statistical_tests=include_statistical_tests,
        include_effect_sizes=include_effect_sizes,
        confidence_level=confidence_level,
        save_path=output_path,
    )


def _display_comparison_summary(request: ComparisonRequestDTO) -> None:
    """Display comparison request summary."""
    click.echo("📋 Comparison Summary:")
    click.echo(f"  Network:     {request.network}")
    click.echo(f"  Dates:       {', '.join(request.dates)}")
    click.echo(f"  Algorithms:  {', '.join(request.algorithms)}")
    click.echo(f"  Metric:      {request.metric}")

    if request.traffic_volumes:
        click.echo(f"  Traffic:     {', '.join(str(tv) for tv in request.traffic_volumes)}")

    click.echo(f"  Tests:       {'Enabled' if request.include_statistical_tests else 'Disabled'}")
    click.echo(f"  Effect Size: {'Enabled' if request.include_effect_sizes else 'Disabled'}")
    click.echo(f"  Confidence:  {request.confidence_level:.1%}")


def _display_comparison_result(result: ComparisonResultDTO, verbose: bool) -> None:
    """Display comparison result."""
    if not result.success:
        click.echo(click.style("\n❌ Comparison failed!", fg="red", bold=True))
        click.echo(f"\nError: {result.error}")
        return

    click.echo(click.style("\n✅ Comparison completed successfully!", fg="green", bold=True))

    click.echo(f"\n📊 Statistical Comparisons ({len(result.comparisons)} pairwise):")
    click.echo(f"{'=' * 80}\n")

    # Display each pairwise comparison
    for i, comp in enumerate(result.comparisons, 1):
        click.echo(f"{i}. {comp.algorithm_a} vs {comp.algorithm_b}")
        click.echo(f"   Metric: {comp.metric}")

        # Means and std devs
        click.echo(
            f"   {comp.algorithm_a}: "
            f"mean={comp.mean_a:.6f}, "
            f"std={comp.std_a:.6f}"
        )
        click.echo(
            f"   {comp.algorithm_b}: "
            f"mean={comp.mean_b:.6f}, "
            f"std={comp.std_b:.6f}"
        )

        # Confidence intervals
        if verbose:
            click.echo(
                f"   {comp.algorithm_a} CI: "
                f"[{comp.ci_lower_a:.6f}, {comp.ci_upper_a:.6f}]"
            )
            click.echo(
                f"   {comp.algorithm_b} CI: "
                f"[{comp.ci_lower_b:.6f}, {comp.ci_upper_b:.6f}]"
            )

        # Statistical test results
        if comp.test_name and comp.p_value is not None:
            significance = "significant" if comp.p_value < 0.05 else "not significant"
            significance_style = "green" if comp.p_value < 0.05 else "yellow"

            click.echo(
                f"   {comp.test_name}: "
                f"p={comp.p_value:.4f} "
                f"({click.style(significance, fg=significance_style)})"
            )

        # Effect size
        if comp.cohens_d is not None:
            effect_str = f"d={comp.cohens_d:.4f}"
            if comp.effect_size_interpretation:
                effect_str += f" ({comp.effect_size_interpretation})"
            click.echo(f"   Cohen's d: {effect_str}")

        # Interpretation
        if comp.p_value is not None and comp.p_value < 0.05:
            better = comp.algorithm_a if comp.mean_a < comp.mean_b else comp.algorithm_b
            worse = comp.algorithm_b if comp.mean_a < comp.mean_b else comp.algorithm_a
            click.echo(
                click.style(
                    f"   → {better} performs significantly better than {worse}",
                    fg="green",
                )
            )

        click.echo()  # Blank line between comparisons

    if result.duration:
        click.echo(f"\n⏱️  Duration: {result.duration.total_seconds():.2f}s")
    click.echo("✨ Done!")


def _save_comparison_report(result: ComparisonResultDTO, output_path: Path) -> None:
    """Save comparison report to text file."""
    with open(output_path, 'w') as f:
        f.write("FUSION Visualization - Algorithm Comparison Report\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Network: {result.network}\n")
        f.write(f"Dates: {', '.join(result.dates)}\n")
        f.write(f"Algorithms: {', '.join(result.algorithms)}\n")
        f.write(f"Metric: {result.metric}\n")
        if result.duration:
            f.write(f"Duration: {result.duration.total_seconds():.2f}s\n\n")
        else:
            f.write("Duration: N/A\n\n")

        f.write(f"Pairwise Comparisons ({len(result.comparisons)}):\n")
        f.write("=" * 80 + "\n\n")

        for i, comp in enumerate(result.comparisons, 1):
            f.write(f"{i}. {comp.algorithm_a} vs {comp.algorithm_b}\n")
            f.write(f"   Metric: {comp.metric}\n")
            f.write(
                f"   {comp.algorithm_a}: mean={comp.mean_a:.6f}, std={comp.std_a:.6f}\n"
            )
            f.write(
                f"   {comp.algorithm_b}: mean={comp.mean_b:.6f}, std={comp.std_b:.6f}\n"
            )
            f.write(
                f"   {comp.algorithm_a} CI: [{comp.ci_lower_a:.6f}, {comp.ci_upper_a:.6f}]\n"
            )
            f.write(
                f"   {comp.algorithm_b} CI: [{comp.ci_lower_b:.6f}, {comp.ci_upper_b:.6f}]\n"
            )

            if comp.test_name and comp.p_value is not None:
                f.write(f"   {comp.test_name}: p={comp.p_value:.4f}\n")

            if comp.cohens_d is not None:
                effect_str = f"d={comp.cohens_d:.4f}"
                if comp.effect_size_interpretation:
                    effect_str += f" ({comp.effect_size_interpretation})"
                f.write(f"   Cohen's d: {effect_str}\n")

            if comp.p_value is not None and comp.p_value < 0.05:
                better = comp.algorithm_a if comp.mean_a < comp.mean_b else comp.algorithm_b
                worse = comp.algorithm_b if comp.mean_a < comp.mean_b else comp.algorithm_a
                f.write(f"   → {better} performs significantly better than {worse}\n")

            f.write("\n")


if __name__ == "__main__":
    viz_cli()
