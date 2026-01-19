"""
CLI command for batch plot generation.

Provides user-friendly command-line interface for generating multiple plots
in a single command, either sequentially or in parallel.
"""

from pathlib import Path

import click

from fusion.visualization.application.dto import (
    BatchPlotRequestDTO,
    BatchPlotResultDTO,
    PlotRequestDTO,
)
from fusion.visualization.application.use_cases.batch_generate_plots import (
    BatchGeneratePlotsUseCase,
)
from fusion.visualization.interface.cli.migrate_command import viz_cli
from fusion.visualization.interface.cli.plot_command import _get_use_case


@viz_cli.command(name="batch")
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to YAML configuration file with multiple plots",
)
@click.option(
    "--sequential/--parallel",
    default=False,
    help="Run plots sequentially or in parallel (default: parallel)",
)
@click.option(
    "--workers",
    "-w",
    type=int,
    default=4,
    help="Number of parallel workers (default: 4)",
)
@click.option(
    "--stop-on-error",
    is_flag=True,
    default=False,
    help="Stop batch generation on first error",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for all plots",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Show detailed output",
)
def batch_command(
    config_path: Path,
    sequential: bool,
    workers: int,
    stop_on_error: bool,
    output_dir: Path | None,
    verbose: bool,
) -> None:
    """
    Generate multiple plots in batch mode.

    This command reads a YAML configuration file containing multiple plot
    specifications and generates them all, either sequentially or in parallel.

    Examples:

        # Generate all plots from config in parallel
        fusion viz batch --config batch_plots.yml

        # Generate sequentially (useful for debugging)
        fusion viz batch -c batch.yml --sequential

        # Use 8 parallel workers
        fusion viz batch -c batch.yml --workers 8

        # Stop on first error
        fusion viz batch -c batch.yml --stop-on-error

        # Save all plots to specific directory
        fusion viz batch -c batch.yml -o ./figures/

    Configuration File Format:

        network: NSFNet
        dates:
          - "0606"
          - "0611"

        plots:
          - type: line
            algorithms: [ppo_obs_7, dqn_obs_7]
            save_path: ./figures/blocking.png

          - type: scatter
            algorithms: [ppo_obs_7]
            save_path: ./figures/rewards.png
    """
    click.echo("📊 FUSION Visualization - Batch Plot Generation")
    click.echo("=" * 50)

    try:
        # Load batch request from config
        request = _load_batch_request_from_config(
            config_path=config_path,
            parallel=not sequential,
            max_workers=workers,
            stop_on_error=stop_on_error,
            output_dir=output_dir,
            verbose=verbose,
        )

        # Validate request
        errors = request.validate()
        if errors:
            click.echo(click.style("❌ Invalid batch request:", fg="red", bold=True))
            for error in errors:
                click.echo(f"  • {error}")
            raise click.Abort()

        # Show batch summary
        _display_batch_summary(request, verbose)

        # Execute batch generation
        use_case = _get_batch_use_case()
        result = use_case.execute(request)

        # Handle legacy list return type
        if isinstance(result, list):
            # Convert to BatchPlotResultDTO for display
            from datetime import datetime

            batch_result = BatchPlotResultDTO(results=result, started_at=datetime.now(), completed_at=datetime.now())
            _display_batch_result(batch_result, verbose)
            if any(not r.success for r in result):
                raise click.Abort()
        else:
            # Display results
            _display_batch_result(result, verbose)

            # Exit with error if any plots failed
            if any(not r.success for r in result.results):
                raise click.Abort()

    except click.Abort:
        raise
    except Exception as e:
        click.echo(click.style(f"\n❌ Error: {e}", fg="red", bold=True))
        if verbose:
            import traceback

            click.echo("\nTraceback:")
            click.echo(traceback.format_exc())
        raise click.Abort() from e


def _get_batch_use_case() -> BatchGeneratePlotsUseCase:
    """Create and configure BatchGeneratePlotsUseCase."""
    generate_plot_use_case = _get_use_case()
    return BatchGeneratePlotsUseCase(generate_plot_use_case=generate_plot_use_case)


def _load_batch_request_from_config(
    config_path: Path,
    parallel: bool,
    max_workers: int,
    stop_on_error: bool,
    output_dir: Path | None,
    verbose: bool,
) -> BatchPlotRequestDTO:
    """Load batch request from YAML config file."""
    import yaml

    from fusion.visualization.domain.value_objects.plot_specification import PlotType

    if verbose:
        click.echo(f"📄 Loading batch configuration from: {config_path}\n")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Extract network and dates (common to all plots)
    network = config.get("network")
    dates = config.get("dates", [])

    if not network:
        raise click.UsageError("Configuration must specify 'network'")
    if not dates:
        raise click.UsageError("Configuration must specify 'dates'")

    # Parse plot specifications
    plot_configs = config.get("plots", [])
    if not plot_configs:
        raise click.UsageError("Configuration must specify at least one plot in 'plots'")

    # Create PlotRequestDTO for each plot
    plot_requests = []
    for i, plot_config in enumerate(plot_configs):
        try:
            # Parse plot type
            plot_type_str = plot_config.get("type", "line")
            try:
                plot_type = PlotType(plot_type_str)
            except ValueError as exc:
                raise ValueError(f"Invalid plot type: {plot_type_str}. Supported: {', '.join(pt.value for pt in PlotType)}") from exc

            # Determine output path
            save_path = None
            if "save_path" in plot_config:
                save_path = Path(plot_config["save_path"])
            elif output_dir:
                # Auto-generate filename in output_dir
                filename = f"{plot_type_str}_{i + 1}.png"
                save_path = Path(output_dir) / filename

            # Create request DTO
            request = PlotRequestDTO(
                network=network,
                dates=dates,
                plot_type=plot_type,
                algorithms=plot_config.get("algorithms"),
                traffic_volumes=plot_config.get("traffic_volumes"),
                run_ids=plot_config.get("run_ids"),
                title=plot_config.get("title"),
                x_label=plot_config.get("x_label"),
                y_label=plot_config.get("y_label"),
                include_ci=plot_config.get("include_ci", True),
                include_baselines=plot_config.get("include_baselines", False),
                save_path=save_path,
                dpi=config.get("defaults", {}).get("dpi", 300),
                figsize=tuple(plot_config.get("figsize", (10, 6))),
                format=config.get("defaults", {}).get("format", "png"),
                cache_enabled=config.get("defaults", {}).get("cache_enabled", True),
            )

            plot_requests.append(request)

        except Exception as e:
            raise click.UsageError(f"Error in plot {i + 1}: {e}") from e

    # Create batch request
    return BatchPlotRequestDTO(
        network=network,
        dates=dates,
        plots=plot_requests,
        parallel=parallel,
        max_workers=max_workers,
        stop_on_error=stop_on_error,
        output_dir=output_dir,
    )


def _display_batch_summary(request: BatchPlotRequestDTO, verbose: bool) -> None:
    """Display batch request summary."""
    click.echo("📋 Batch Summary:")
    click.echo(f"  Network:     {request.network}")
    click.echo(f"  Dates:       {', '.join(request.dates)}")
    click.echo(f"  Total Plots: {len(request.plots)}")
    click.echo(f"  Mode:        {'Parallel' if request.parallel else 'Sequential'}")

    if request.parallel:
        click.echo(f"  Workers:     {request.max_workers}")

    if request.output_dir:
        click.echo(f"  Output Dir:  {request.output_dir}")

    if verbose:
        click.echo("\n  Plot Types:")
        for i, plot in enumerate(request.plots, 1):
            algos_str = f" ({len(plot.algorithms)} algo(s))" if plot.algorithms else ""
            click.echo(f"    {i}. {plot.plot_type.value}{algos_str}")

    click.echo()


def _display_batch_result(result: BatchPlotResultDTO, verbose: bool) -> None:
    """Display batch generation result."""
    successful = sum(1 for r in result.results if r.success)
    failed = len(result.results) - successful

    click.echo(f"\n{'=' * 50}")
    click.echo("📊 Batch Generation Complete")
    click.echo(f"{'=' * 50}\n")

    # Summary statistics
    if successful == len(result.results):
        click.echo(
            click.style(
                f"✅ All {successful} plots generated successfully!",
                fg="green",
                bold=True,
            )
        )
    else:
        click.echo(
            click.style(
                f"⚠️  {successful}/{len(result.results)} plots succeeded, {failed} failed",
                fg="yellow" if successful > 0 else "red",
                bold=True,
            )
        )

    duration_seconds = result.duration.total_seconds() if result.duration else 0.0
    click.echo(f"\n⏱️  Total Duration: {duration_seconds:.2f}s")

    # List results
    if verbose or failed > 0:
        click.echo("\n📋 Individual Results:")

        for i, plot_result in enumerate(result.results, 1):
            if plot_result.success:
                status = click.style("✓", fg="green")
                info = f"{plot_result.output_path}"
            else:
                status = click.style("✗", fg="red")
                info = f"Error: {plot_result.error}"

            click.echo(f"  {status} Plot {i}: {info}")

            if verbose and plot_result.success:
                click.echo(f"      Type: {plot_result.plot_type}")
                duration = plot_result.duration.total_seconds() if plot_result.duration else 0.0
                click.echo(f"      Duration: {duration:.2f}s")

    # Output directory summary
    if successful > 0:
        click.echo("\n📁 Generated plots:")
        for plot_result in result.results:
            if plot_result.success and plot_result.output_path:
                click.echo(f"  • {plot_result.output_path}")

    click.echo("\n✨ Done!")


if __name__ == "__main__":
    viz_cli()
