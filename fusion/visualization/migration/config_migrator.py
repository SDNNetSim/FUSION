"""
Configuration migration from old to new visualization system.

This module provides tools to convert legacy YAML configurations to the
new format, ensuring backward compatibility during the transition period.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


@dataclass
class MigrationResult:
    """Result of a configuration migration."""

    success: bool
    new_config: dict[str, Any] | None
    warnings: list[str]
    errors: list[str]
    deprecated_fields: list[str]

    def __str__(self) -> str:
        """String representation of migration result."""
        lines = [f"Migration {'succeeded' if self.success else 'failed'}"]

        if self.deprecated_fields:
            lines.append(f"\nDeprecated fields found: {len(self.deprecated_fields)}")
            for field in self.deprecated_fields:
                lines.append(f"  - {field}")

        if self.warnings:
            lines.append(f"\nWarnings: {len(self.warnings)}")
            for warning in self.warnings:
                lines.append(f"  - {warning}")

        if self.errors:
            lines.append(f"\nErrors: {len(self.errors)}")
            for error in self.errors:
                lines.append(f"  - {error}")

        return "\n".join(lines)


class ConfigMigrator:
    """
    Migrates legacy visualization configurations to new format.

    Example:
        >>> migrator = ConfigMigrator()
        >>> result = migrator.migrate_file("old_config.yml", "new_config.yml")
        >>> if result.success:
        ...     print("Migration successful!")
        ... else:
        ...     print(f"Migration failed: {result.errors}")
    """

    # Mapping of old field names to new field names
    FIELD_MAPPING = {
        "observation_spaces": "algorithms",  # obs_7 → ppo_obs_7
        "runs": "run_filters",
        "plots": "plots",
        "network": "network",
        "dates": "dates",
    }

    # Deprecated fields that should be removed
    DEPRECATED_FIELDS = [
        "use_cache",  # Now under defaults.cache_enabled
        "plot_style",  # Now under defaults.style
        "save_format",  # Now under defaults.format
    ]

    # Fields that require transformation
    TRANSFORMATION_FIELDS = {
        "observation_spaces": "algorithms",
        "runs": "run_filters",
    }

    def __init__(self) -> None:
        """Initialize configuration migrator."""
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self.deprecated_fields: list[str] = []

    def migrate_file(
        self,
        old_config_path: Path,
        new_config_path: Path | None = None,
        backup: bool = True,
    ) -> MigrationResult:
        """
        Migrate a configuration file from old to new format.

        Args:
            old_config_path: Path to old configuration file
            new_config_path: Path for new configuration
                (default: old_path with .new.yml suffix)
            backup: Whether to create a backup of the old config

        Returns:
            MigrationResult with details of the migration
        """
        old_config_path = Path(old_config_path)

        if not old_config_path.exists():
            return MigrationResult(
                success=False,
                new_config=None,
                warnings=[],
                errors=[f"Config file not found: {old_config_path}"],
                deprecated_fields=[],
            )

        # Load old configuration
        try:
            with open(old_config_path) as f:
                old_config = yaml.safe_load(f)
        except Exception as e:
            return MigrationResult(
                success=False,
                new_config=None,
                warnings=[],
                errors=[f"Failed to load config: {str(e)}"],
                deprecated_fields=[],
            )

        # Migrate configuration
        new_config = self.migrate_config(old_config)

        # Determine output path
        if new_config_path is None:
            new_config_path = old_config_path.with_suffix(".new.yml")
        else:
            new_config_path = Path(new_config_path)

        # Create backup if requested
        if backup and old_config_path.exists():
            backup_path = old_config_path.with_suffix(".bak.yml")
            import shutil

            shutil.copy2(old_config_path, backup_path)
            self.warnings.append(f"Created backup at {backup_path}")

        # Write new configuration
        try:
            with open(new_config_path, "w") as f:
                yaml.dump(new_config, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            return MigrationResult(
                success=False,
                new_config=new_config,
                warnings=self.warnings,
                errors=self.errors + [f"Failed to write config: {str(e)}"],
                deprecated_fields=self.deprecated_fields,
            )

        return MigrationResult(
            success=len(self.errors) == 0,
            new_config=new_config,
            warnings=self.warnings,
            errors=self.errors,
            deprecated_fields=self.deprecated_fields,
        )

    def migrate_config(self, old_config: dict[str, Any]) -> dict[str, Any]:
        """
        Migrate a configuration dictionary from old to new format.

        Args:
            old_config: Old configuration dictionary

        Returns:
            New configuration dictionary
        """
        self.warnings = []
        self.errors = []
        self.deprecated_fields = []

        new_config: dict[str, Any] = {}

        # Migrate basic fields
        if "network" in old_config:
            new_config["network"] = old_config["network"]
        else:
            self.errors.append("Missing required field: network")

        if "dates" in old_config:
            # Ensure dates are strings
            dates = old_config["dates"]
            if isinstance(dates, list):
                new_config["dates"] = [str(d) for d in dates]
            else:
                new_config["dates"] = [str(dates)]
        else:
            self.errors.append("Missing required field: dates")

        # Migrate plots
        if "plots" in old_config:
            new_config["plots"] = self._migrate_plots(old_config["plots"], old_config)
        else:
            self.errors.append("Missing required field: plots")

        # Migrate defaults
        new_config["defaults"] = self._migrate_defaults(old_config)

        # Check for deprecated fields
        for field in self.DEPRECATED_FIELDS:
            if field in old_config:
                self.deprecated_fields.append(field)
                self.warnings.append(
                    f"Deprecated field '{field}' found. "
                    f"It has been migrated to the appropriate location."
                )

        # Add migration metadata
        new_config["_migration"] = {
            "migrated_at": datetime.now().isoformat(),
            "original_format": "legacy",
            "migrator_version": "1.0",
        }

        return new_config

    def _migrate_plots(
        self, plots: Any, full_config: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """
        Migrate plot configurations.

        Old format examples:
        - plots: ["blocking", "rewards"]
        - plots: [{"type": "blocking", "algorithms": [...]}]

        New format:
        - plots: [{"type": "blocking", "algorithms": [...], ...}]
        """
        migrated_plots = []

        # Handle simple list format (just plot type names)
        if isinstance(plots, list) and len(plots) > 0 and isinstance(plots[0], str):
            for plot_type in plots:
                migrated_plot = {
                    "type": plot_type,
                }

                # Add global algorithms if available
                if "algorithms" in full_config:
                    migrated_plot["algorithms"] = self._migrate_algorithms(
                        full_config["algorithms"],
                        full_config.get("observation_spaces", []),
                    )
                elif "observation_spaces" in full_config:
                    # Old format: separate algorithm and obs_space
                    migrated_plot["algorithms"] = self._migrate_algorithms(
                        full_config.get("algorithms", []),
                        full_config["observation_spaces"],
                    )
                    self.deprecated_fields.append("observation_spaces")

                # Add traffic volumes if available
                if "traffic_volumes" in full_config:
                    migrated_plot["traffic_volumes"] = full_config["traffic_volumes"]

                migrated_plots.append(migrated_plot)

        # Handle dict format
        elif isinstance(plots, list):
            for plot in plots:
                if isinstance(plot, dict):
                    migrated_plot = self._migrate_single_plot(plot, full_config)
                    migrated_plots.append(migrated_plot)
                else:
                    migrated_plot = {"type": str(plot)}
                    migrated_plots.append(migrated_plot)
        else:
            self.errors.append(f"Invalid plots format: {type(plots)}")

        return migrated_plots

    def _migrate_single_plot(
        self, plot: dict[str, Any], full_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Migrate a single plot configuration."""
        migrated = {}

        # Copy type
        if "type" in plot:
            migrated["type"] = plot["type"]
        else:
            self.errors.append("Plot missing required field: type")

        # Migrate algorithms
        if "algorithms" in plot:
            migrated["algorithms"] = self._migrate_algorithms(
                plot["algorithms"],
                plot.get(
                    "observation_spaces", full_config.get("observation_spaces", [])
                ),
            )
        elif "observation_spaces" in plot:
            migrated["algorithms"] = self._migrate_algorithms(
                plot.get("algorithms", full_config.get("algorithms", [])),
                plot["observation_spaces"],
            )
            self.deprecated_fields.append("observation_spaces")
        elif "algorithms" in full_config:
            migrated["algorithms"] = self._migrate_algorithms(
                full_config["algorithms"], full_config.get("observation_spaces", [])
            )

        # Copy other fields
        for field in ["traffic_volumes", "title", "x_label", "y_label", "save_path"]:
            if field in plot:
                migrated[field] = plot[field]
            elif field in full_config:
                migrated[field] = full_config[field]

        # Migrate include flags
        if "include_baselines" in plot:
            migrated["include_baselines"] = plot["include_baselines"]

        if "include_ci" in plot:
            migrated["include_ci"] = plot["include_ci"]
        elif "confidence_intervals" in plot:
            migrated["include_ci"] = plot["confidence_intervals"]
            self.deprecated_fields.append("confidence_intervals")

        return migrated

    def _migrate_algorithms(
        self, algorithms: list[str], observation_spaces: list[str]
    ) -> list[str]:
        """
        Migrate algorithm specifications.

        Old format: algorithms=["ppo"], observation_spaces=["obs_7"]
        New format: algorithms=["ppo_obs_7"]
        """
        migrated = []

        if observation_spaces:
            # Combine algorithm and observation space
            for algo in algorithms:
                for obs in observation_spaces:
                    # Handle case where it's already combined
                    if "_obs_" in algo:
                        migrated.append(algo)
                    else:
                        migrated.append(f"{algo}_{obs}")
        else:
            # No observation spaces, use algorithms as-is
            migrated = algorithms.copy()

        return migrated

    def _migrate_defaults(self, old_config: dict[str, Any]) -> dict[str, Any]:
        """Migrate default settings."""
        defaults = {}

        # Migrate format
        if "save_format" in old_config:
            defaults["format"] = old_config["save_format"]
            self.deprecated_fields.append("save_format")
        elif "format" in old_config:
            defaults["format"] = old_config["format"]
        else:
            defaults["format"] = "png"

        # Migrate DPI
        if "dpi" in old_config:
            defaults["dpi"] = old_config["dpi"]
        else:
            defaults["dpi"] = 300

        # Migrate style
        if "plot_style" in old_config:
            defaults["style"] = old_config["plot_style"]
            self.deprecated_fields.append("plot_style")
        elif "style" in old_config:
            defaults["style"] = old_config["style"]
        else:
            defaults["style"] = "default"

        # Migrate cache settings
        if "use_cache" in old_config:
            defaults["cache_enabled"] = old_config["use_cache"]
            self.deprecated_fields.append("use_cache")
        elif "cache_enabled" in old_config:
            defaults["cache_enabled"] = old_config["cache_enabled"]
        else:
            defaults["cache_enabled"] = True

        return defaults

    def validate_config(self, config: dict[str, Any]) -> list[str]:
        """
        Validate a configuration dictionary.

        Args:
            config: Configuration to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check required fields
        required_fields = ["network", "dates", "plots"]
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")

        # Validate network
        if "network" in config:
            if not isinstance(config["network"], str) or not config["network"]:
                errors.append("Network must be a non-empty string")

        # Validate dates
        if "dates" in config:
            if not isinstance(config["dates"], list) or len(config["dates"]) == 0:
                errors.append("Dates must be a non-empty list")

        # Validate plots
        if "plots" in config:
            if not isinstance(config["plots"], list) or len(config["plots"]) == 0:
                errors.append("Plots must be a non-empty list")
            else:
                for i, plot in enumerate(config["plots"]):
                    if not isinstance(plot, dict):
                        errors.append(f"Plot {i} must be a dictionary")
                    elif "type" not in plot:
                        errors.append(f"Plot {i} missing required field: type")

        return errors


def migrate_config_file(
    old_path: str,
    new_path: str | None = None,
    backup: bool = True,
    verbose: bool = False,
) -> bool:
    """
    Convenience function to migrate a configuration file.

    Args:
        old_path: Path to old configuration file
        new_path: Path for new configuration (optional)
        backup: Whether to create a backup
        verbose: Whether to print detailed output

    Returns:
        True if migration succeeded, False otherwise
    """
    migrator = ConfigMigrator()
    result = migrator.migrate_file(
        Path(old_path),
        Path(new_path) if new_path else None,
        backup=backup,
    )

    if verbose:
        print(result)

    return result.success
