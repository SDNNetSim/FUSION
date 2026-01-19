"""Configuration validation service."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of configuration validation."""

    is_valid: bool
    errors: list[str]
    warnings: list[str]

    @property
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0

    def __str__(self) -> str:
        """String representation."""
        lines = []
        if self.errors:
            lines.append("Errors:")
            for error in self.errors:
                lines.append(f"  - {error}")
        if self.warnings:
            lines.append("Warnings:")
            for warning in self.warnings:
                lines.append(f"  - {warning}")
        return "\n".join(lines) if lines else "Validation passed"


class ValidationService:
    """
    Service for validating configurations.

    This provides comprehensive validation of plot configurations,
    checking for common errors and providing helpful warnings.
    """

    VALID_NETWORKS = ["NSFNet", "USNet", "Pan-European", "dt_network"]
    VALID_PLOT_TYPES = [
        "blocking",
        "rewards",
        "memory",
        "convergence",
        "heatmap",
        "comparison",
    ]
    VALID_FORMATS = ["png", "pdf", "svg", "jpg"]

    def validate_plot_config(self, config: dict[str, Any]) -> ValidationResult:
        """
        Validate a plot configuration dictionary.

        Args:
            config: Configuration dictionary

        Returns:
            ValidationResult with errors and warnings
        """
        errors = []
        warnings = []

        # Required fields
        if "network" not in config:
            errors.append("Missing required field: network")
        elif config["network"] not in self.VALID_NETWORKS:
            warnings.append(f"Unknown network: {config['network']}. Valid networks: {', '.join(self.VALID_NETWORKS)}")

        if "dates" not in config:
            errors.append("Missing required field: dates")
        elif not isinstance(config["dates"], list) or not config["dates"]:
            errors.append("dates must be a non-empty list")
        else:
            # Validate date format
            for date in config["dates"]:
                if not isinstance(date, str) or len(date) != 4:
                    errors.append(f"Invalid date format: {date} (expected MMDD)")

        if "plot_type" not in config:
            errors.append("Missing required field: plot_type")
        elif config["plot_type"] not in self.VALID_PLOT_TYPES:
            errors.append(f"Invalid plot_type: {config['plot_type']}. Valid types: {', '.join(self.VALID_PLOT_TYPES)}")

        # Optional fields validation
        if "algorithms" in config:
            if not isinstance(config["algorithms"], list):
                errors.append("algorithms must be a list")
            elif not config["algorithms"]:
                warnings.append("Empty algorithms list - will use all available")

        if "traffic_volumes" in config:
            if not isinstance(config["traffic_volumes"], list):
                errors.append("traffic_volumes must be a list")
            elif not config["traffic_volumes"]:
                warnings.append("Empty traffic_volumes list - will use all available")
            else:
                # Check for valid values
                for tv in config["traffic_volumes"]:
                    if not isinstance(tv, (int, float)) or tv <= 0:
                        errors.append(f"Invalid traffic volume: {tv} (must be positive number)")

        # Output configuration
        if "save_path" in config:
            save_path = Path(config["save_path"])
            parent = save_path.parent
            if not parent.exists():
                warnings.append(f"Output directory does not exist: {parent} (will be created automatically)")

        if "format" in config:
            fmt = config["format"]
            if fmt not in self.VALID_FORMATS:
                errors.append(f"Invalid format: {fmt}. Valid formats: {', '.join(self.VALID_FORMATS)}")

        if "dpi" in config:
            dpi = config["dpi"]
            if not isinstance(dpi, int) or dpi <= 0:
                errors.append(f"Invalid dpi: {dpi} (must be positive integer)")
            elif dpi < 72:
                warnings.append(f"Low dpi value: {dpi} (recommended: 300 for print)")

        # Advanced options
        if "figsize" in config:
            figsize = config["figsize"]
            if (
                not isinstance(figsize, (list, tuple))
                or len(figsize) != 2
                or not all(isinstance(x, (int, float)) and x > 0 for x in figsize)
            ):
                errors.append("Invalid figsize: must be (width, height) with positive numbers")

        is_valid = len(errors) == 0

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
        )

    def validate_batch_config(self, config: dict[str, Any]) -> ValidationResult:
        """
        Validate a batch plot configuration.

        Args:
            config: Batch configuration dictionary

        Returns:
            ValidationResult with errors and warnings
        """
        errors = []
        warnings = []

        # Required fields for batch
        if "network" not in config:
            errors.append("Missing required field: network")

        if "dates" not in config:
            errors.append("Missing required field: dates")

        if "plots" not in config:
            errors.append("Missing required field: plots")
        elif not isinstance(config["plots"], list):
            errors.append("plots must be a list")
        elif not config["plots"]:
            errors.append("plots list cannot be empty")
        else:
            # Validate each plot
            for i, plot_config in enumerate(config["plots"]):
                # Add network and dates if not present
                if "network" not in plot_config:
                    plot_config["network"] = config["network"]
                if "dates" not in plot_config:
                    plot_config["dates"] = config["dates"]

                plot_result = self.validate_plot_config(plot_config)
                for error in plot_result.errors:
                    errors.append(f"Plot {i + 1}: {error}")
                for warning in plot_result.warnings:
                    warnings.append(f"Plot {i + 1}: {warning}")

        # Batch-specific options
        if "parallel" in config:
            if not isinstance(config["parallel"], bool):
                errors.append("parallel must be boolean")

        if "max_workers" in config:
            max_workers = config["max_workers"]
            if not isinstance(max_workers, int) or max_workers < 1:
                errors.append("max_workers must be positive integer")
            elif max_workers > 16:
                warnings.append(f"High max_workers value: {max_workers} (may cause resource contention)")

        is_valid = len(errors) == 0

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
        )

    def validate_comparison_config(self, config: dict[str, Any]) -> ValidationResult:
        """
        Validate an algorithm comparison configuration.

        Args:
            config: Comparison configuration dictionary

        Returns:
            ValidationResult with errors and warnings
        """
        errors: list[str] = []
        warnings: list[str] = []

        # Required fields
        if "network" not in config:
            errors.append("Missing required field: network")

        if "dates" not in config:
            errors.append("Missing required field: dates")

        if "algorithms" not in config:
            errors.append("Missing required field: algorithms")
        elif not isinstance(config["algorithms"], list):
            errors.append("algorithms must be a list")
        elif len(config["algorithms"]) < 2:
            errors.append("At least 2 algorithms required for comparison")

        if "metric" not in config:
            errors.append("Missing required field: metric")
        elif not isinstance(config["metric"], str):
            errors.append("metric must be a string")

        # Optional fields
        if "confidence_level" in config:
            cl = config["confidence_level"]
            if not isinstance(cl, (int, float)) or not (0 < cl < 1):
                errors.append("confidence_level must be between 0 and 1 (e.g., 0.95 for 95%)")

        is_valid = len(errors) == 0

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
        )

    def suggest_fixes(self, errors: list[str]) -> dict[str, str]:
        """
        Suggest fixes for common validation errors.

        Args:
            errors: List of error messages

        Returns:
            Dictionary mapping error keywords to suggested fixes
        """
        suggestions = {}

        for error in errors:
            if "network" in error.lower():
                suggestions["network"] = f"Use one of: {', '.join(self.VALID_NETWORKS)}"
            elif "plot_type" in error.lower():
                suggestions["plot_type"] = f"Use one of: {', '.join(self.VALID_PLOT_TYPES)}"
            elif "dates" in error.lower():
                suggestions["dates"] = 'Use MMDD format, e.g., ["0606", "0611"]'
            elif "traffic_volumes" in error.lower():
                suggestions["traffic_volumes"] = "Use list of positive numbers, e.g., [600, 700, 800]"

        return suggestions
