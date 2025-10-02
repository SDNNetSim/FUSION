"""
CLI command for migrating visualization configurations.

Provides user-friendly command-line interface for migrating old configuration
files to the new format.
"""

from pathlib import Path

import click

from fusion.visualization.migration import ConfigMigrator, MigrationResult


@click.group(name="viz")
def viz_cli() -> None:
    """Visualization system commands."""
    pass


@viz_cli.command(name="migrate")
@click.option(
    "--input",
    "-i",
    "input_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to old configuration file",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Path for new configuration file (default: <input>.new.yml)",
)
@click.option(
    "--no-backup",
    is_flag=True,
    default=False,
    help="Don't create a backup of the original file",
)
@click.option(
    "--validate",
    is_flag=True,
    default=False,
    help="Validate the new configuration after migration",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Show what would be migrated without writing files",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Show detailed output",
)
def migrate_config(
    input_path: Path,
    output_path: Path | None,
    no_backup: bool,
    validate: bool,
    dry_run: bool,
    verbose: bool,
) -> None:
    """
    Migrate old visualization configuration to new format.

    This command converts legacy YAML configuration files to the new format,
    handling deprecated fields and structural changes automatically.

    Examples:

        # Migrate a configuration file
        fusion viz migrate --input old_config.yml --output new_config.yml

        # Migrate with validation
        fusion viz migrate -i old.yml -o new.yml --validate

        # Preview migration without writing files
        fusion viz migrate -i old.yml --dry-run -v

        # Migrate without creating backup
        fusion viz migrate -i old.yml --no-backup
    """
    click.echo(f"🔄 Migrating configuration: {input_path}")

    # Create migrator
    migrator = ConfigMigrator()

    # Perform migration
    if dry_run:
        # Load and migrate without writing
        import yaml

        with open(input_path) as f:
            old_config = yaml.safe_load(f)

        new_config = migrator.migrate_config(old_config)

        result = MigrationResult(
            success=len(migrator.errors) == 0,
            new_config=new_config,
            warnings=migrator.warnings,
            errors=migrator.errors,
            deprecated_fields=migrator.deprecated_fields,
        )

        click.echo("\n📋 Dry run - no files will be written\n")
    else:
        result = migrator.migrate_file(
            old_config_path=input_path,
            new_config_path=output_path,
            backup=not no_backup,
        )

    # Display results
    _display_migration_result(result, verbose=verbose)

    # Validate if requested
    if validate and result.success and result.new_config:
        click.echo("\n🔍 Validating new configuration...")
        validation_errors = migrator.validate_config(result.new_config)

        if validation_errors:
            click.echo(click.style("❌ Validation failed:", fg="red", bold=True))
            for error in validation_errors:
                click.echo(f"  • {error}")
            raise click.Abort()
        else:
            click.echo(click.style("✅ Validation passed!", fg="green", bold=True))

    # Show output path
    if result.success and not dry_run:
        actual_output = output_path or input_path.with_suffix(".new.yml")
        click.echo(f"\n✨ Migration complete! New config saved to: {actual_output}")

        # Show next steps
        click.echo("\n📚 Next steps:")
        click.echo("  1. Review the new configuration file")
        click.echo("  2. Test with: fusion viz validate --config <new-config>.yml")
        click.echo("  3. Update your scripts to use the new config")
        if not no_backup:
            backup_path = input_path.with_suffix(".bak.yml")
            click.echo(
                f"  4. Once confirmed working, you can delete the backup: "
                f"{backup_path}"
            )
    elif not result.success:
        click.echo(click.style("\n❌ Migration failed!", fg="red", bold=True))
        raise click.Abort()


@viz_cli.command(name="validate")
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to configuration file to validate",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Show detailed validation results",
)
def validate_config(config_path: Path, verbose: bool) -> None:
    """
    Validate a visualization configuration file.

    Checks that the configuration file has valid syntax and all required
    fields are present with correct types.

    Examples:

        # Validate a configuration
        fusion viz validate --config plot_config.yml

        # Validate with detailed output
        fusion viz validate -c config.yml -v
    """
    click.echo(f"🔍 Validating configuration: {config_path}")

    # Load configuration
    import yaml

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except Exception as e:
        click.echo(click.style(f"❌ Failed to load config: {e}", fg="red", bold=True))
        raise click.Abort() from e

    # Validate
    migrator = ConfigMigrator()
    errors = migrator.validate_config(config)

    # Display results
    if not errors:
        click.echo(click.style("✅ Configuration is valid!", fg="green", bold=True))

        if verbose:
            click.echo("\n📋 Configuration summary:")
            click.echo(f"  Network: {config.get('network', 'N/A')}")
            click.echo(f"  Dates: {len(config.get('dates', []))} date(s)")
            click.echo(f"  Plots: {len(config.get('plots', []))} plot(s)")

            if "plots" in config:
                click.echo("\n  Plot types:")
                for i, plot in enumerate(config["plots"], 1):
                    plot_type = plot.get("type", "unknown")
                    algorithms = plot.get("algorithms", [])
                    click.echo(f"    {i}. {plot_type} ({len(algorithms)} algorithm(s))")
    else:
        click.echo(click.style("❌ Configuration is invalid!", fg="red", bold=True))
        click.echo("\nErrors found:")
        for error in errors:
            click.echo(f"  • {error}")
        raise click.Abort()


@viz_cli.command(name="info")
def show_info() -> None:
    """
    Display visualization system information.

    Shows version, available plot types, and configuration options.
    """
    click.echo("📊 FUSION Visualization System\n")
    click.echo("Version: 6.0.0 (New Architecture)")
    click.echo("Documentation: docs/visualization/\n")

    click.echo("Available commands:")
    click.echo("  plot       - Generate a single plot")
    click.echo("  batch      - Generate multiple plots")
    click.echo("  compare    - Compare algorithms")
    click.echo("  migrate    - Migrate old configuration")
    click.echo("  validate   - Validate configuration")
    click.echo("  list-plots - List available plot types")
    click.echo("  info       - Show this information")

    click.echo("\nSupported plot types:")
    click.echo("  • blocking       - Blocking probability vs traffic")
    click.echo("  • rewards        - RL rewards over time")
    click.echo("  • memory         - Memory usage")
    click.echo("  • computation    - Computation time")
    click.echo("  • heatmap        - Spectrum utilization heatmap")
    click.echo("  • custom         - Custom plot types via plugins")

    click.echo("\nFor more information:")
    click.echo("  fusion viz --help")
    click.echo("  fusion viz <command> --help")


@viz_cli.command(name="list-plots")
@click.option(
    "--plugin",
    "-p",
    default=None,
    help="Filter by plugin name",
)
def list_plot_types(plugin: str | None) -> None:
    """
    List available plot types.

    Shows all registered plot types and their descriptions.
    """
    click.echo("📊 Available Plot Types:\n")

    # Built-in plot types
    builtin_plots = {
        "blocking": "Blocking probability vs traffic volume",
        "rewards": "RL training rewards over time",
        "memory": "Memory usage over time",
        "computation": "Computation time analysis",
        "hops": "Average hop count vs traffic",
        "lengths": "Average path length vs traffic",
    }

    click.echo(click.style("Built-in plots:", fg="blue", bold=True))
    for plot_type, description in builtin_plots.items():
        click.echo(f"  • {plot_type:15s} - {description}")

    # Plugin plots (would be loaded dynamically in real implementation)
    click.echo(f"\n{click.style('Plugin plots:', fg='blue', bold=True)}")
    click.echo("  (Use --plugin to filter)")

    click.echo("\nTo use a plot type:")
    click.echo("  fusion viz plot --type <plot-type> --config <config>.yml")


def _display_migration_result(result: MigrationResult, verbose: bool = False) -> None:
    """Display migration result in a user-friendly format."""
    if result.deprecated_fields:
        click.echo(f"\n⚠️  Found {len(result.deprecated_fields)} deprecated field(s):")
        for field in result.deprecated_fields:
            click.echo(f"  • {field}")

    if result.warnings:
        click.echo(f"\n⚠️  {len(result.warnings)} warning(s):")
        for warning in result.warnings:
            click.echo(f"  • {warning}")

    if result.errors:
        click.echo(
            click.style(f"\n❌ {len(result.errors)} error(s):", fg="red", bold=True)
        )
        for error in result.errors:
            click.echo(f"  • {error}")

    if verbose and result.new_config:
        click.echo("\n📄 New configuration preview:")
        import yaml

        config_preview = yaml.dump(
            result.new_config, default_flow_style=False, sort_keys=False
        )
        # Show first 30 lines
        lines = config_preview.split("\n")[:30]
        click.echo("  " + "\n  ".join(lines))
        if len(config_preview.split("\n")) > 30:
            click.echo("  ... (truncated)")


if __name__ == "__main__":
    viz_cli()
