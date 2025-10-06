"""
CLI command for generating individual plots.

Provides user-friendly command-line interface for generating single plots
from simulation data.
"""

from pathlib import Path

import click

from fusion.visualization.application.dto import PlotRequestDTO, PlotResultDTO
from fusion.visualization.application.use_cases.generate_plot import GeneratePlotUseCase
from fusion.visualization.domain.value_objects.plot_specification import PlotType
from fusion.visualization.interface.cli.migrate_command import viz_cli


def _get_use_case() -> GeneratePlotUseCase:
    """
    Create and configure GeneratePlotUseCase with dependencies.

    Returns:
        Configured GeneratePlotUseCase instance
    """
    from fusion.visualization.infrastructure.adapters.data_adapter_registry import (
        DataAdapterRegistry,
    )
    from fusion.visualization.infrastructure.cache import FileSystemCache
    from fusion.visualization.infrastructure.processors import MultiMetricProcessor
    from fusion.visualization.infrastructure.renderers import MatplotlibRenderer
    from fusion.visualization.infrastructure.repositories import (
        JsonSimulationRepository,
    )

    # Create adapter registry
    adapter_registry = DataAdapterRegistry()

    # Create repository
    base_path = Path("../../data/output")  # TODO: Make configurable
    repository = JsonSimulationRepository(
        base_path=base_path,
        adapter_registry=adapter_registry,
    )

    # Create processor and renderer
    processor = MultiMetricProcessor()
    renderer = MatplotlibRenderer()

    # Create cache (optional)
    cache_dir = Path.home() / ".fusion" / "cache"
    cache = FileSystemCache(cache_dir=cache_dir)

    # Create use case
    return GeneratePlotUseCase(
        simulation_repository=repository,
        data_processor=processor,
        plot_renderer=renderer,
        cache=cache,
    )


@viz_cli.command(name="plot")
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
    "--type",
    "-t",
    "plot_type",
    type=click.Choice([pt.value for pt in PlotType], case_sensitive=False),
    help="Plot type",
)
@click.option(
    "--algorithms",
    "-a",
    multiple=True,
    help="Algorithms to plot (can specify multiple)",
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
    help="Output file path (default: auto-generated)",
)
@click.option(
    "--title",
    type=str,
    default=None,
    help="Plot title (default: auto-generated)",
)
@click.option(
    "--dpi",
    type=int,
    default=300,
    help="Output resolution in DPI",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["png", "pdf", "svg", "jpg"]),
    default="png",
    help="Output format",
)
@click.option(
    "--no-ci",
    is_flag=True,
    default=False,
    help="Disable confidence intervals",
)
@click.option(
    "--baselines/--no-baselines",
    default=False,
    help="Include baseline algorithms",
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Disable caching",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Show detailed output",
)
def plot_command(
    config_path: Path | None,
    network: str | None,
    dates: tuple,
    plot_type: str | None,
    algorithms: tuple,
    traffic: tuple,
    output_path: Path | None,
    title: str | None,
    dpi: int,
    output_format: str,
    no_ci: bool,
    baselines: bool,
    no_cache: bool,
    verbose: bool,
) -> None:
    """
    Generate a single plot from simulation data.

    This command loads simulation data and generates a visualization plot.
    You can either provide a YAML configuration file or specify parameters
    via command-line arguments.

    Examples:

        # Generate blocking plot from config file
        fusion viz plot --config plot_config.yml

        # Generate plot with CLI arguments
        fusion viz plot -n NSFNet -d 0606 -t line \\
            -a ppo_obs_7 -a dqn_obs_7 --traffic 600 --traffic 700

        # Generate with custom output
        fusion viz plot -c config.yml -o ./figures/blocking.png --dpi 600

        # Generate without confidence intervals
        fusion viz plot -c config.yml --no-ci
    """
    click.echo("📊 FUSION Visualization - Plot Generation")
    click.echo("=" * 50)

    try:
        # Load request from config or CLI args
        if config_path:
            request = _load_request_from_config(config_path, verbose)
        else:
            request = _create_request_from_args(
                network=network,
                dates=list(dates),
                plot_type=plot_type,
                algorithms=list(algorithms),
                traffic=list(traffic),
                output_path=output_path,
                title=title,
                dpi=dpi,
                output_format=output_format,
                include_ci=not no_ci,
                include_baselines=baselines,
                cache_enabled=not no_cache,
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
            _display_request_summary(request)

        click.echo(f"\n🔄 Generating {request.plot_type.value} plot...")

        # Execute plot generation
        use_case = _get_use_case()
        result = use_case.execute(request)

        # Display results
        _display_result(result, verbose)

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
        raise click.Abort() from e


def _load_request_from_config(config_path: Path, verbose: bool) -> PlotRequestDTO:
    """Load plot request from YAML config file."""
    import yaml

    if verbose:
        click.echo(f"📄 Loading configuration from: {config_path}\n")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Extract plot configuration (assume first plot if multiple)
    plot_config = config.get("plots", [{}])[0] if "plots" in config else config

    # Parse plot type
    plot_type_str = plot_config.get("type", config.get("type", "line"))
    try:
        plot_type = PlotType(plot_type_str)
    except ValueError as exc:
        raise click.UsageError(
            f"Invalid plot type: {plot_type_str}. "
            f"Supported: {', '.join(pt.value for pt in PlotType)}"
        ) from exc

    # Create request DTO
    return PlotRequestDTO(
        network=config.get("network"),
        dates=config.get("dates", []),
        plot_type=plot_type,
        algorithms=plot_config.get("algorithms"),
        traffic_volumes=plot_config.get("traffic_volumes"),
        run_ids=plot_config.get("run_ids"),
        title=plot_config.get("title"),
        x_label=plot_config.get("x_label"),
        y_label=plot_config.get("y_label"),
        include_ci=plot_config.get("include_ci", True),
        include_baselines=plot_config.get("include_baselines", False),
        save_path=Path(plot_config["save_path"])
        if "save_path" in plot_config
        else None,
        dpi=config.get("defaults", {}).get("dpi", 300),
        figsize=tuple(plot_config.get("figsize", (10, 6))),
        format=config.get("defaults", {}).get("format", "png"),
        cache_enabled=config.get("defaults", {}).get("cache_enabled", True),
        metadata=config.get("metadata", {}),
    )


def _create_request_from_args(
    network: str | None,
    dates: list[str],
    plot_type: str | None,
    algorithms: list[str],
    traffic: list[float],
    output_path: Path | None,
    title: str | None,
    dpi: int,
    output_format: str,
    include_ci: bool,
    include_baselines: bool,
    cache_enabled: bool,
) -> PlotRequestDTO:
    """Create plot request from CLI arguments."""
    # Validate required args
    if not network:
        raise click.UsageError("--network is required when not using --config")
    if not dates:
        raise click.UsageError("--dates is required when not using --config")
    if not plot_type:
        raise click.UsageError("--type is required when not using --config")

    # Parse plot type
    try:
        plot_type_enum = PlotType(plot_type)
    except ValueError as exc:
        raise click.UsageError(
            f"Invalid plot type: {plot_type}. "
            f"Supported: {', '.join(pt.value for pt in PlotType)}"
        ) from exc

    return PlotRequestDTO(
        network=network,
        dates=list(dates),
        plot_type=plot_type_enum,
        algorithms=list(algorithms) if algorithms else None,
        traffic_volumes=list(traffic) if traffic else None,
        save_path=output_path,
        title=title,
        include_ci=include_ci,
        include_baselines=include_baselines,
        dpi=dpi,
        format=output_format,
        cache_enabled=cache_enabled,
    )


def _display_request_summary(request: PlotRequestDTO) -> None:
    """Display request summary."""
    click.echo("📋 Request Summary:")
    click.echo(f"  Network:     {request.network}")
    click.echo(f"  Dates:       {', '.join(request.dates)}")
    click.echo(f"  Plot Type:   {request.plot_type.value}")

    if request.algorithms:
        click.echo(f"  Algorithms:  {', '.join(request.algorithms)}")

    if request.traffic_volumes:
        click.echo(
            f"  Traffic:     {', '.join(str(tv) for tv in request.traffic_volumes)}"
        )

    click.echo(f"  Output DPI:  {request.dpi}")
    click.echo(f"  Format:      {request.format}")
    click.echo(f"  Include CI:  {request.include_ci}")


def _display_result(result: PlotResultDTO, verbose: bool) -> None:
    """Display plot generation result."""
    if result.success:
        click.echo(
            click.style("\n✅ Plot generated successfully!", fg="green", bold=True)
        )
        click.echo(f"\n📁 Output: {result.output_path}")

        if verbose:
            click.echo("\n📊 Plot Details:")
            click.echo(f"  Plot ID:     {result.plot_id}")
            click.echo(f"  Plot Type:   {result.plot_type}")
            click.echo(f"  Algorithms:  {', '.join(result.algorithms)}")
            click.echo(f"  Runs Used:   {result.num_runs}")
            if result.duration:
                click.echo(f"  Duration:    {result.duration.total_seconds():.2f}s")

            if result.traffic_volumes:
                click.echo(f"  Traffic:     {len(result.traffic_volumes)} point(s)")

        click.echo("\n✨ Done!")
    else:
        click.echo(click.style("\n❌ Plot generation failed!", fg="red", bold=True))
        click.echo(f"\nError: {result.error}")

        if verbose and result.plot_id:
            click.echo(f"\nPlot ID: {result.plot_id}")
            if result.duration:
                click.echo(f"Duration: {result.duration.total_seconds():.2f}s")


if __name__ == "__main__":
    viz_cli()
