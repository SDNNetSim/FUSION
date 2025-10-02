"""Unit tests for fusion.configs.errors module."""

import pytest

from fusion.configs.errors import (
    ConfigError,
    ConfigFileNotFoundError,
    ConfigParseError,
    ConfigTypeConversionError,
    MissingRequiredOptionError,
)


class TestConfigError:
    """Tests for ConfigError exception class."""

    def test_config_error_inheritance(self) -> None:
        """Test that ConfigError inherits from Exception."""
        error = ConfigError("Test error")

        assert isinstance(error, Exception)
        assert str(error) == "Test error"

    def test_config_error_with_message(self) -> None:
        """Test ConfigError with custom message."""
        message = "Custom configuration error message"
        error = ConfigError(message)

        assert str(error) == message

    def test_config_error_can_be_raised(self) -> None:
        """Test that ConfigError can be raised and caught."""
        with pytest.raises(ConfigError) as exc_info:
            raise ConfigError("Test error")

        assert str(exc_info.value) == "Test error"


class TestConfigFileNotFoundError:
    """Tests for ConfigFileNotFoundError exception class."""

    def test_inheritance_from_config_error(self) -> None:
        """Test that ConfigFileNotFoundError inherits from ConfigError."""
        error = ConfigFileNotFoundError("File not found")

        assert isinstance(error, ConfigError)
        assert isinstance(error, Exception)

    def test_error_with_file_path(self) -> None:
        """Test error with file path information."""
        file_path = "/path/to/config.ini"
        message = f"Configuration file not found: {file_path}"
        error = ConfigFileNotFoundError(message)

        assert str(error) == message

    def test_can_be_caught_as_config_error(self) -> None:
        """Test that ConfigFileNotFoundError can be caught as ConfigError."""
        with pytest.raises(ConfigError):
            raise ConfigFileNotFoundError("File not found")

    def test_can_be_caught_specifically(self) -> None:
        """Test that ConfigFileNotFoundError can be caught specifically."""
        with pytest.raises(ConfigFileNotFoundError) as exc_info:
            raise ConfigFileNotFoundError("Specific file error")

        assert str(exc_info.value) == "Specific file error"


class TestConfigParseError:
    """Tests for ConfigParseError exception class."""

    def test_inheritance_from_config_error(self) -> None:
        """Test that ConfigParseError inherits from ConfigError."""
        error = ConfigParseError("Parse error")

        assert isinstance(error, ConfigError)
        assert isinstance(error, Exception)

    def test_error_with_parse_details(self) -> None:
        """Test error with parsing details."""
        message = "Invalid JSON format in 'config.json': Expecting ',' delimiter"
        error = ConfigParseError(message)

        assert str(error) == message

    def test_can_be_caught_as_config_error(self) -> None:
        """Test that ConfigParseError can be caught as ConfigError."""
        with pytest.raises(ConfigError):
            raise ConfigParseError("Parse error")

    def test_can_be_caught_specifically(self) -> None:
        """Test that ConfigParseError can be caught specifically."""
        with pytest.raises(ConfigParseError) as exc_info:
            raise ConfigParseError("Specific parse error")

        assert str(exc_info.value) == "Specific parse error"


class TestMissingRequiredOptionError:
    """Tests for MissingRequiredOptionError exception class."""

    def test_inheritance_from_config_error(self) -> None:
        """Test that MissingRequiredOptionError inherits from ConfigError."""
        error = MissingRequiredOptionError("Missing option")

        assert isinstance(error, ConfigError)
        assert isinstance(error, Exception)

    def test_error_with_option_details(self) -> None:
        """Test error with missing option details."""
        message = (
            "Required option 'holding_time' is missing from section 'general_settings'"
        )
        error = MissingRequiredOptionError(message)

        assert str(error) == message

    def test_can_be_caught_as_config_error(self) -> None:
        """Test that MissingRequiredOptionError can be caught as ConfigError."""
        with pytest.raises(ConfigError):
            raise MissingRequiredOptionError("Missing option")

    def test_can_be_caught_specifically(self) -> None:
        """Test that MissingRequiredOptionError can be caught specifically."""
        with pytest.raises(MissingRequiredOptionError) as exc_info:
            raise MissingRequiredOptionError("Specific missing option error")

        assert str(exc_info.value) == "Specific missing option error"


class TestConfigTypeConversionError:
    """Tests for ConfigTypeConversionError exception class."""

    def test_inheritance_from_config_error(self) -> None:
        """Test that ConfigTypeConversionError inherits from ConfigError."""
        error = ConfigTypeConversionError("Conversion error")

        assert isinstance(error, ConfigError)
        assert isinstance(error, Exception)

    def test_error_with_conversion_details(self) -> None:
        """Test error with type conversion details."""
        message = "Cannot convert 'invalid_number' to float for option 'holding_time'"
        error = ConfigTypeConversionError(message)

        assert str(error) == message

    def test_can_be_caught_as_config_error(self) -> None:
        """Test that ConfigTypeConversionError can be caught as ConfigError."""
        with pytest.raises(ConfigError):
            raise ConfigTypeConversionError("Conversion error")

    def test_can_be_caught_specifically(self) -> None:
        """Test that ConfigTypeConversionError can be caught specifically."""
        with pytest.raises(ConfigTypeConversionError) as exc_info:
            raise ConfigTypeConversionError("Specific conversion error")

        assert str(exc_info.value) == "Specific conversion error"


class TestExceptionHierarchy:
    """Tests for exception hierarchy and polymorphism."""

    def test_all_config_exceptions_inherit_from_config_error(self) -> None:
        """Test that all config exceptions inherit from ConfigError."""
        exceptions = [
            ConfigFileNotFoundError("test"),
            ConfigParseError("test"),
            MissingRequiredOptionError("test"),
            ConfigTypeConversionError("test"),
        ]

        for exception in exceptions:
            assert isinstance(exception, ConfigError)

    def test_can_catch_all_with_base_exception(self) -> None:
        """Test that all config exceptions can be caught with ConfigError."""
        exceptions_to_test = [
            ConfigFileNotFoundError("file error"),
            ConfigParseError("parse error"),
            MissingRequiredOptionError("missing error"),
            ConfigTypeConversionError("conversion error"),
        ]

        for exception in exceptions_to_test:
            with pytest.raises(ConfigError):
                raise exception

    def test_specific_exception_catching(self) -> None:
        """Test that specific exceptions can be caught individually."""
        # Test that each exception type can be caught specifically
        with pytest.raises(ConfigFileNotFoundError):
            raise ConfigFileNotFoundError("file not found")

        with pytest.raises(ConfigParseError):
            raise ConfigParseError("parse failed")

        with pytest.raises(MissingRequiredOptionError):
            raise MissingRequiredOptionError("option missing")

        with pytest.raises(ConfigTypeConversionError):
            raise ConfigTypeConversionError("conversion failed")

    def test_exception_message_preservation(self) -> None:
        """Test that exception messages are preserved through inheritance."""
        test_message = "Test error message"

        exceptions = [
            ConfigError(test_message),
            ConfigFileNotFoundError(test_message),
            ConfigParseError(test_message),
            MissingRequiredOptionError(test_message),
            ConfigTypeConversionError(test_message),
        ]

        for exception in exceptions:
            assert str(exception) == test_message

    def test_exception_with_cause(self) -> None:
        """Test exceptions with cause chains."""
        original_error = ValueError("Original error")

        try:
            try:
                raise original_error
            except ValueError as e:
                raise ConfigError("Config error") from e
        except ConfigError as config_error:
            assert config_error.__cause__ is original_error

    def test_empty_message_handling(self) -> None:
        """Test exception handling with empty messages."""
        exceptions = [
            ConfigError(""),
            ConfigFileNotFoundError(""),
            ConfigParseError(""),
            MissingRequiredOptionError(""),
            ConfigTypeConversionError(""),
        ]

        for exception in exceptions:
            assert str(exception) == ""
