"""Unit tests for fusion.configs.validate module."""

import json
import os
import tempfile
from unittest.mock import patch

import pytest

from fusion.configs.validate import SchemaValidator, ValidationError


class TestValidationError:
    """Tests for ValidationError exception class."""

    def test_validation_error_inheritance(self) -> None:
        """Test that ValidationError inherits from Exception."""
        error = ValidationError("Test validation error")

        assert isinstance(error, Exception)
        assert str(error) == "Test validation error"

    def test_validation_error_can_be_raised(self) -> None:
        """Test that ValidationError can be raised and caught."""
        with pytest.raises(ValidationError) as exc_info:
            raise ValidationError("Validation failed")

        assert str(exc_info.value) == "Validation failed"


class TestSchemaValidator:
    """Tests for SchemaValidator class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.schema_dir = os.path.join(self.temp_dir, 'schemas')
        os.makedirs(self.schema_dir)

    def teardown_method(self) -> None:
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_init_with_default_schema_dir(self) -> None:
        """Test SchemaValidator initialization with default schema directory."""
        validator = SchemaValidator()

        assert validator.schema_dir.endswith('schemas')
        assert isinstance(validator.schemas, dict)

    def test_init_with_custom_schema_dir(self) -> None:
        """Test SchemaValidator initialization with custom schema directory."""
        validator = SchemaValidator(self.schema_dir)

        assert validator.schema_dir == self.schema_dir
        assert isinstance(validator.schemas, dict)

    def test_load_schemas_empty_directory(self) -> None:
        """Test loading schemas from empty directory."""
        validator = SchemaValidator(self.schema_dir)

        assert validator.schemas == {}

    def test_load_schemas_with_valid_json(self) -> None:
        """Test loading schemas with valid JSON files."""
        # Create test schema files
        schema1 = {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            }
        }
        schema2 = {
            "type": "object",
            "properties": {
                "value": {"type": "number"}
            }
        }

        schema1_path = os.path.join(self.schema_dir, 'schema1.json')
        schema2_path = os.path.join(self.schema_dir, 'schema2.json')

        with open(schema1_path, 'w') as f:
            json.dump(schema1, f)
        with open(schema2_path, 'w') as f:
            json.dump(schema2, f)

        validator = SchemaValidator(self.schema_dir)

        assert 'schema1' in validator.schemas
        assert 'schema2' in validator.schemas
        assert validator.schemas['schema1'] == schema1
        assert validator.schemas['schema2'] == schema2

    def test_load_schemas_invalid_json(self) -> None:
        """Test loading schemas with invalid JSON files."""
        # Create invalid JSON file
        invalid_schema_path = os.path.join(self.schema_dir, 'invalid.json')
        with open(invalid_schema_path, 'w') as f:
            f.write('{"invalid": json}')

        with patch('fusion.configs.validate.logger') as mock_logger:
            validator = SchemaValidator(self.schema_dir)

            assert 'invalid' not in validator.schemas
            mock_logger.warning.assert_called()

    def test_load_schemas_nonexistent_directory(self) -> None:
        """Test loading schemas from non-existent directory."""
        nonexistent_dir = os.path.join(self.temp_dir, 'nonexistent')
        validator = SchemaValidator(nonexistent_dir)

        assert validator.schemas == {}

    def test_validate_success(self) -> None:
        """Test successful configuration validation."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            },
            "required": ["name"]
        }

        schema_path = os.path.join(self.schema_dir, 'test.json')
        with open(schema_path, 'w') as f:
            json.dump(schema, f)

        validator = SchemaValidator(self.schema_dir)
        config = {"name": "test", "age": 25}

        # Should not raise any exception
        validator.validate(config, 'test')

    def test_validate_failure(self) -> None:
        """Test configuration validation failure."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "required": ["name"]
        }

        schema_path = os.path.join(self.schema_dir, 'test.json')
        with open(schema_path, 'w') as f:
            json.dump(schema, f)

        validator = SchemaValidator(self.schema_dir)
        config = {"age": 25}  # Missing required "name" field

        with pytest.raises(ValidationError) as exc_info:
            validator.validate(config, 'test')

        assert "Configuration validation failed" in str(exc_info.value)

    def test_validate_schema_not_found(self) -> None:
        """Test validation with non-existent schema."""
        validator = SchemaValidator(self.schema_dir)
        config = {"name": "test"}

        with patch('fusion.configs.validate.logger') as mock_logger:
            # Should not raise exception, just log warning
            validator.validate(config, 'nonexistent')

            mock_logger.warning.assert_called_with(
                "Schema 'nonexistent' not found, skipping validation"
            )

    def test_validate_type_string(self) -> None:
        """Test type validation for string values."""
        validator = SchemaValidator(self.schema_dir)
        schema = {"type": "string"}

        # Valid string
        errors = validator._validate_type("test", schema, "field")
        assert errors == []

        # Invalid type
        errors = validator._validate_type(123, schema, "field")
        assert len(errors) == 1
        assert "Expected string, got int" in errors[0]

    def test_validate_type_number(self) -> None:
        """Test type validation for number values."""
        validator = SchemaValidator(self.schema_dir)
        schema = {"type": "number"}

        # Valid numbers
        errors = validator._validate_type(123, schema, "field")
        assert errors == []

        errors = validator._validate_type(123.45, schema, "field")
        assert errors == []

        # Invalid type
        errors = validator._validate_type("123", schema, "field")
        assert len(errors) == 1
        assert "Expected number, got str" in errors[0]

    def test_validate_type_integer(self) -> None:
        """Test type validation for integer values."""
        validator = SchemaValidator(self.schema_dir)
        schema = {"type": "integer"}

        # Valid integer
        errors = validator._validate_type(123, schema, "field")
        assert errors == []

        # Invalid types
        errors = validator._validate_type(123.45, schema, "field")
        assert len(errors) == 1
        assert "Expected integer, got float" in errors[0]

    def test_validate_type_boolean(self) -> None:
        """Test type validation for boolean values."""
        validator = SchemaValidator(self.schema_dir)
        schema = {"type": "boolean"}

        # Valid booleans
        errors = validator._validate_type(True, schema, "field")
        assert errors == []

        errors = validator._validate_type(False, schema, "field")
        assert errors == []

        # Invalid type
        errors = validator._validate_type("true", schema, "field")
        assert len(errors) == 1
        assert "Expected boolean, got str" in errors[0]

    def test_validate_type_object(self) -> None:
        """Test type validation for object values."""
        validator = SchemaValidator(self.schema_dir)
        schema = {"type": "object"}

        # Valid object
        errors = validator._validate_type({"key": "value"}, schema, "field")
        assert errors == []

        # Invalid type
        errors = validator._validate_type("not an object", schema, "field")
        assert len(errors) == 1
        assert "Expected object, got str" in errors[0]

    def test_validate_type_array(self) -> None:
        """Test type validation for array values."""
        validator = SchemaValidator(self.schema_dir)
        schema = {"type": "array"}

        # Valid array
        errors = validator._validate_type([1, 2, 3], schema, "field")
        assert errors == []

        # Invalid type
        errors = validator._validate_type("not an array", schema, "field")
        assert len(errors) == 1
        assert "Expected array, got str" in errors[0]

    def test_validate_type_null(self) -> None:
        """Test type validation for null values."""
        validator = SchemaValidator(self.schema_dir)
        schema = {"type": "null"}

        # Valid null
        errors = validator._validate_type(None, schema, "field")
        assert errors == []

        # Invalid type
        errors = validator._validate_type("not null", schema, "field")
        assert len(errors) == 1
        assert "Expected null, got str" in errors[0]

    def test_validate_numeric_constraints(self) -> None:
        """Test validation of numeric constraints."""
        validator = SchemaValidator(self.schema_dir)

        # Test minimum constraint
        schema = {"type": "number", "minimum": 10}
        errors = validator._validate_type(15, schema, "field")
        assert errors == []

        errors = validator._validate_type(5, schema, "field")
        assert len(errors) == 1
        assert "below minimum" in errors[0]

        # Test maximum constraint
        schema = {"type": "number", "maximum": 100}
        errors = validator._validate_type(50, schema, "field")
        assert errors == []

        errors = validator._validate_type(150, schema, "field")
        assert len(errors) == 1
        assert "above maximum" in errors[0]

    def test_validate_string_enum(self) -> None:
        """Test validation of string enumeration constraints."""
        validator = SchemaValidator(self.schema_dir)
        schema = {"type": "string", "enum": ["red", "green", "blue"]}

        # Valid enum value
        errors = validator._validate_type("red", schema, "field")
        assert errors == []

        # Invalid enum value
        errors = validator._validate_type("yellow", schema, "field")
        assert len(errors) == 1
        assert "not in allowed values" in errors[0]

    def test_validate_required_fields(self) -> None:
        """Test validation of required fields."""
        validator = SchemaValidator(self.schema_dir)
        config = {"name": "test", "age": 25}
        required = ["name", "email"]

        errors = validator._validate_required_fields(config, required, "")

        assert len(errors) == 1
        assert "email: Required field missing" in errors[0]

    def test_validate_recursive_object_properties(self) -> None:
        """Test recursive validation of object properties."""
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "number"}
                    },
                    "required": ["name"]
                }
            }
        }

        schema_path = os.path.join(self.schema_dir, 'nested.json')
        with open(schema_path, 'w') as f:
            json.dump(schema, f)

        validator = SchemaValidator(self.schema_dir)

        # Valid nested object
        config = {"user": {"name": "John", "age": 30}}
        validator.validate(config, 'nested')  # Should not raise

        # Invalid nested object
        config = {"user": {"age": 30}}  # Missing required "name"
        with pytest.raises(ValidationError):
            validator.validate(config, 'nested')

    def test_validate_recursive_array_items(self) -> None:
        """Test recursive validation of array items."""
        schema = {
            "type": "object",
            "properties": {
                "numbers": {
                    "type": "array",
                    "items": {"type": "number"}
                }
            }
        }

        schema_path = os.path.join(self.schema_dir, 'array.json')
        with open(schema_path, 'w') as f:
            json.dump(schema, f)

        validator = SchemaValidator(self.schema_dir)

        # Valid array
        config = {"numbers": [1, 2, 3.5]}
        validator.validate(config, 'array')  # Should not raise

        # Invalid array items - contains string which violates number schema
        from typing import Any
        invalid_numbers: list[Any] = [1, "two", 3.0]
        config = {"numbers": invalid_numbers}
        with pytest.raises(ValidationError):
            validator.validate(config, 'array')

    def test_get_default_config_success(self) -> None:
        """Test generating default configuration from schema."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "default": "default_name"},
                "age": {"type": "number", "default": 0},
                "active": {"type": "boolean", "default": True}
            }
        }

        schema_path = os.path.join(self.schema_dir, 'defaults.json')
        with open(schema_path, 'w') as f:
            json.dump(schema, f)

        validator = SchemaValidator(self.schema_dir)
        defaults = validator.get_default_config('defaults')

        assert defaults['name'] == "default_name"
        assert defaults['age'] == 0
        assert defaults['active'] is True

    def test_get_default_config_schema_not_found(self) -> None:
        """Test getting default config for non-existent schema."""
        validator = SchemaValidator(self.schema_dir)
        defaults = validator.get_default_config('nonexistent')

        assert defaults == {}

    def test_generate_defaults_type_defaults(self) -> None:
        """Test generating defaults based on types when no explicit defaults."""
        validator = SchemaValidator(self.schema_dir)

        type_schemas = [
            ({"type": "string"}, ""),
            ({"type": "number"}, 0.0),
            ({"type": "integer"}, 0),
            ({"type": "boolean"}, False),
            ({"type": "array"}, []),
            ({"type": "null"}, None),
        ]

        for schema, expected_default in type_schemas:
            result = validator._generate_defaults(schema)
            assert result == expected_default

    def test_generate_defaults_no_type(self) -> None:
        """Test generating defaults when no type is specified."""
        validator = SchemaValidator(self.schema_dir)
        schema = {"description": "A field without type"}

        result = validator._generate_defaults(schema)

        assert result is None

    def test_path_handling_in_errors(self) -> None:
        """Test that error messages include proper field paths."""
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}
                    }
                }
            }
        }

        schema_path = os.path.join(self.schema_dir, 'paths.json')
        with open(schema_path, 'w') as f:
            json.dump(schema, f)

        validator = SchemaValidator(self.schema_dir)
        config = {"user": {"name": 123}}  # Wrong type for name

        with pytest.raises(ValidationError) as exc_info:
            validator.validate(config, 'paths')

        error_message = str(exc_info.value)
        assert "user.name" in error_message

