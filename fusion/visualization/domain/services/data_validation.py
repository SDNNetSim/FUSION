"""Data validation service for the domain layer."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ValidationResult:
    """Result of data validation."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)

    def __bool__(self) -> bool:
        """Allow using result in boolean context."""
        return self.is_valid

    def __repr__(self) -> str:
        """Return representation."""
        status = "valid" if self.is_valid else "invalid"
        error_count = len(self.errors)
        warning_count = len(self.warnings)
        return (
            f"ValidationResult({status}, "
            f"errors={error_count}, warnings={warning_count})"
        )


class DataValidationService:
    """
    Service for validating data against schemas and requirements.

    This service checks data structure, required fields, types, and values
    to ensure they meet expected formats.
    """

    def validate(
        self,
        data: dict[str, Any],
        required_fields: list[str] | None = None,
        schema: dict[str, Any] | None = None,
    ) -> ValidationResult:
        """
        Validate data against requirements.

        Args:
            data: Data to validate
            required_fields: List of required field names
            schema: Optional schema dict with field types

        Returns:
            ValidationResult with errors and warnings
        """
        result = ValidationResult(is_valid=True)

        # Check for None or empty data
        if data is None:
            result.add_error("Data is None")
            return result

        if not isinstance(data, dict):
            result.add_error(f"Data must be a dict, got {type(data).__name__}")
            return result

        # Validate required fields
        if required_fields:
            for field in required_fields:
                if field not in data:
                    result.add_error(f"Required field '{field}' is missing")

        # Validate against schema if provided
        if schema:
            self._validate_schema(data, schema, result)

        return result

    def validate_simulation_data(self, data: dict[str, Any]) -> ValidationResult:
        """
        Validate simulation output data.

        Args:
            data: Simulation data to validate

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)

        # Check for common simulation fields
        common_fields = [
            "blocking_mean",
            "iter_stats",
            "iterations",
            "metrics",
        ]

        # At least one of these should be present
        if not any(field in data for field in common_fields):
            result.add_warning(
                "Data does not contain any common simulation fields. "
                "Expected at least one of: " + ", ".join(common_fields)
            )

        # Check deprecated fields
        deprecated_fields = {
            "blocking_mean": "Use metrics.blocking_probability instead",
            "observation_spaces": "Use algorithm name with format 'ppo_obs_7'",
        }

        for field_name, message in deprecated_fields.items():
            if field_name in data:
                result.add_warning(f"Field '{field_name}' is deprecated. {message}")

        # Validate iter_stats structure if present
        if "iter_stats" in data:
            iter_stats = data["iter_stats"]
            if not isinstance(iter_stats, (dict, list)):
                result.add_error("'iter_stats' must be a dict or list")
            elif isinstance(iter_stats, dict):
                # V1 format validation
                for key, value in iter_stats.items():
                    if not isinstance(value, dict):
                        result.add_error(
                            f"iter_stats[{key}] should be a dict, "
                            f"got {type(value).__name__}"
                        )

        # Validate iterations structure if present
        if "iterations" in data:
            iterations = data["iterations"]
            if not isinstance(iterations, list):
                result.add_error("'iterations' must be a list")

        return result

    def _validate_schema(
        self,
        data: dict[str, Any],
        schema: dict[str, Any],
        result: ValidationResult,
    ) -> None:
        """
        Validate data against a schema.

        Args:
            data: Data to validate
            schema: Schema definition
            result: ValidationResult to populate
        """
        for field_name, expected_type in schema.items():
            if field_name not in data:
                continue  # Optional field

            value = data[field_name]
            actual_type = type(value).__name__

            # Handle type checking
            if isinstance(expected_type, type):
                if not isinstance(value, expected_type):
                    result.add_error(
                        f"Field '{field_name}' should be {expected_type.__name__}, "
                        f"got {actual_type}"
                    )

    def validate_metric_path(self, path: str) -> ValidationResult:
        """
        Validate a metric path expression.

        Args:
            path: JSONPath expression

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)

        if not path:
            result.add_error("Metric path cannot be empty")
            return result

        # Basic validation (can be expanded)
        if not path.startswith("$") and not path.split(".")[0].isalnum():
            result.add_warning(
                f"Metric path '{path}' does not look like a valid JSONPath"
            )

        return result

    def suggest_missing_fields(
        self,
        data: dict[str, Any],
        expected_fields: list[str],
    ) -> list[str]:
        """
        Suggest fields that might be missing.

        Args:
            data: Data to check
            expected_fields: List of expected field names

        Returns:
            List of missing field names
        """
        return [field for field in expected_fields if field not in data]
