"""Schema validation for FUSION configuration files."""

import os
import json
from typing import Dict, Any, Optional, List
from pathlib import Path


class ValidationError(Exception):
    """Custom exception for configuration validation errors."""


class SchemaValidator:
    """Schema validator for configuration files."""

    def __init__(self, schema_dir: Optional[str] = None):
        """Initialize schema validator.
        
        Args:
            schema_dir: Directory containing schema files
        """
        self.schema_dir = schema_dir or os.path.join(os.path.dirname(__file__), 'schemas')
        self.schemas: Dict[str, Dict[str, Any]] = {}
        self._load_schemas()

    def _load_schemas(self) -> None:
        """Load all schema files from schema directory."""
        if not os.path.exists(self.schema_dir):
            return

        for schema_file in Path(self.schema_dir).glob('*.json'):
            schema_name = schema_file.stem
            try:
                with open(schema_file, 'r', encoding='utf-8') as f:
                    self.schemas[schema_name] = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load schema {schema_file}: {e}")

    def validate(self, config: Dict[str, Any], schema_name: str = 'main') -> None:
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
        errors = self._validate_recursive(config, schema, '')

        if errors:
            raise ValidationError("Configuration validation failed:\n" + "\n".join(errors))

    def _validate_recursive(self, config: Any, schema: Any, path: str) -> List[str]:
        """Recursively validate configuration against schema.
        
        Args:
            config: Configuration value to validate
            schema: Schema to validate against
            path: Current path in configuration (for error reporting)
            
        Returns:
            List of validation errors
        """
        errors = []

        if isinstance(schema, dict):
            if 'type' in schema:
                errors.extend(self._validate_type(config, schema, path))

            if 'required' in schema and isinstance(config, dict):
                errors.extend(self._validate_required_fields(config, schema['required'], path))

            if 'properties' in schema and isinstance(config, dict):
                for prop, prop_schema in schema['properties'].items():
                    if prop in config:
                        prop_path = f"{path}.{prop}" if path else prop
                        errors.extend(self._validate_recursive(config[prop], prop_schema, prop_path))

            if 'items' in schema and isinstance(config, list):
                for i, item in enumerate(config):
                    item_path = f"{path}[{i}]" if path else f"[{i}]"
                    errors.extend(self._validate_recursive(item, schema['items'], item_path))

        return errors

    def _validate_type(self, value: Any, schema: Dict[str, Any], path: str) -> List[str]:
        """Validate value type against schema.
        
        Args:
            value: Value to validate
            schema: Schema containing type information
            path: Current path in configuration
            
        Returns:
            List of validation errors
        """
        errors = []
        expected_type = schema['type']

        type_map = {
            'string': str,
            'number': (int, float),
            'integer': int,
            'boolean': bool,
            'object': dict,
            'array': list,
            'null': type(None)
        }

        if expected_type in type_map:
            expected_python_type = type_map[expected_type]
            if not isinstance(value, expected_python_type):
                actual_type = type(value).__name__
                errors.append(f"{path}: Expected {expected_type}, got {actual_type}")

        # Additional type-specific validations
        if expected_type == 'number' and 'minimum' in schema:
            if value < schema['minimum']:
                errors.append(f"{path}: Value {value} is below minimum {schema['minimum']}")

        if expected_type == 'number' and 'maximum' in schema:
            if value > schema['maximum']:
                errors.append(f"{path}: Value {value} is above maximum {schema['maximum']}")

        if expected_type == 'string' and 'enum' in schema:
            if value not in schema['enum']:
                errors.append(f"{path}: Value '{value}' not in allowed values: {schema['enum']}")

        return errors

    def _validate_required_fields(self, config: Dict[str, Any], required: List[str], path: str) -> List[str]:
        """Validate that required fields are present.
        
        Args:
            config: Configuration dictionary
            required: List of required field names
            path: Current path in configuration
            
        Returns:
            List of validation errors
        """
        errors = []
        for field in required:
            if field not in config:
                field_path = f"{path}.{field}" if path else field
                errors.append(f"{field_path}: Required field missing")
        return errors

    def get_default_config(self, schema_name: str = 'main') -> Dict[str, Any]:
        """Generate default configuration from schema.
        
        Args:
            schema_name: Name of schema to use
            
        Returns:
            Default configuration dictionary
        """
        if schema_name not in self.schemas:
            return {}

        return self._generate_defaults(self.schemas[schema_name])

    def _generate_defaults(self, schema: Dict[str, Any]) -> Any:
        """Generate default values from schema.
        
        Args:
            schema: Schema to generate defaults from
            
        Returns:
            Default value based on schema
        """
        if 'default' in schema:
            return schema['default']

        if 'type' not in schema:
            return None

        schema_type = schema['type']

        if schema_type == 'object' and 'properties' in schema:
            result = {}
            for prop, prop_schema in schema['properties'].items():
                result[prop] = self._generate_defaults(prop_schema)
            return result

        # Type defaults mapping
        type_defaults = {
            'array': [],
            'string': "",
            'number': 0.0,
            'integer': 0,
            'boolean': False,
            'null': None
        }

        return type_defaults.get(schema_type, None)
