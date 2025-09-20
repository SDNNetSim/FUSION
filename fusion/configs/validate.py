"""Schema validation for FUSION configuration files."""

import json
import os
from pathlib import Path
from typing import Any


class ValidationError(Exception):
    """Custom exception for configuration validation errors."""


class SchemaValidator:
    """Schema validator for configuration files."""

    def __init__(self, schema_dir: str | None = None):
        """Initialize schema validator.

        Args:
            schema_dir: Directory containing schema files
        """
        self.schema_dir = schema_dir or os.path.join(
            os.path.dirname(__file__), "schemas"
        )
        self.schemas: dict[str, dict[str, Any]] = {}
        self._load_schemas()

    def _load_schemas(self) -> None:
        """Load all schema files from schema directory."""
        if not os.path.exists(self.schema_dir):
            return

        for schema_file in Path(self.schema_dir).glob("*.json"):
            schema_name = schema_file.stem
            try:
                with open(schema_file, encoding="utf-8") as f:
                    self.schemas[schema_name] = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                print(f"Warning: Could not load schema {schema_file}: {e}")

    def validate(self, config: dict[str, Any], schema_name: str = "main") -> None:
        """Validate configuration against schema.

        Args:
            config: Configuration dictionary to validate
            schema_name: Name of schema to use for validation

        Raises:
            ValidationError: If configuration is invalid
        """
        if schema_name not in self.schemas:
            print(f"Warning: Schema '{schema_name}' not found, skipping validation")
            return

        schema = self.schemas[schema_name]
        errors = self._validate_recursive(config, schema, "")

        if errors:
            raise ValidationError(
                "Configuration validation failed:\n" + "\n".join(errors)
            )

    def _validate_recursive(self, config: Any, schema: Any, path: str) -> list[str]:
        errors_list: list[str] = []

        if isinstance(schema, dict):
            if "type" in schema:
                errors_list.extend(self._validate_type(config, schema, path))

            if "required" in schema and isinstance(config, dict):
                errors_list.extend(
                    self._validate_required_fields(config, schema["required"], path)
                )

            if "properties" in schema and isinstance(config, dict):
                for prop, prop_schema in schema["properties"].items():
                    if prop in config:
                        prop_path = f"{path}.{prop}" if path else prop
                        errors_list.extend(
                            self._validate_recursive(
                                config[prop], prop_schema, prop_path
                            )
                        )

            if "items" in schema and isinstance(config, list):
                for i, item in enumerate(config):
                    item_path = f"{path}[{i}]" if path else f"[{i}]"
                    errors_list.extend(
                        self._validate_recursive(item, schema["items"], item_path)
                    )

        return errors_list

    def _validate_type(
        self, value: Any, schema: dict[str, Any], path: str
    ) -> list[str]:
        errors_list: list[str] = []
        expected_type = schema["type"]

        type_map = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "object": dict,
            "array": list,
            "null": type(None),
        }

        if expected_type in type_map:
            expected_python_type = type_map[expected_type]
            if not isinstance(value, expected_python_type):
                actual_type = type(value).__name__
                errors_list.append(
                    f"{path}: Expected {expected_type}, got {actual_type}"
                )

        # Validate numeric constraints
        if expected_type == "number" and "minimum" in schema:
            if value < schema["minimum"]:
                errors_list.append(
                    f"{path}: Value {value} is below minimum {schema['minimum']}"
                )

        if expected_type == "number" and "maximum" in schema:
            if value > schema["maximum"]:
                errors_list.append(
                    f"{path}: Value {value} is above maximum {schema['maximum']}"
                )

        # Validate string enumeration
        if expected_type == "string" and "enum" in schema:
            if value not in schema["enum"]:
                errors_list.append(
                    f"{path}: Value '{value}' not in allowed values: {schema['enum']}"
                )

        return errors_list

    def _validate_required_fields(
        self, config: dict[str, Any], required: list[str], path: str
    ) -> list[str]:
        errors_list: list[str] = []
        for field in required:
            if field not in config:
                field_path = f"{path}.{field}" if path else field
                errors_list.append(f"{field_path}: Required field missing")
        return errors_list

    def get_default_config(self, schema_name: str = "main") -> dict[str, Any]:
        """Generate default configuration from schema.

        Args:
            schema_name: Name of schema to use

        Returns:
            Default configuration dictionary
        """
        if schema_name not in self.schemas:
            return {}

        return self._generate_defaults(self.schemas[schema_name])

    def _generate_defaults(self, schema: dict[str, Any]) -> Any:
        """Generate default values from schema.

        Args:
            schema: Schema to generate defaults from

        Returns:
            Default value based on schema
        """
        if "default" in schema:
            return schema["default"]

        if "type" not in schema:
            return None

        schema_type = schema["type"]

        if schema_type == "object" and "properties" in schema:
            result = {}
            for prop, prop_schema in schema["properties"].items():
                result[prop] = self._generate_defaults(prop_schema)
            return result

        # Type defaults mapping
        type_defaults = {
            "array": [],
            "string": "",
            "number": 0.0,
            "integer": 0,
            "boolean": False,
            "null": None,
        }

        return type_defaults.get(schema_type, None)
