"""Unit tests for fusion.configs.cli_to_config module."""

import argparse
from unittest.mock import Mock

import pytest

from fusion.configs.cli_to_config import CLIToConfigMapper
from fusion.configs.errors import ConfigTypeConversionError


class TestCLIToConfigMapper:
    """Tests for CLIToConfigMapper class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mapper = CLIToConfigMapper()

    def test_initialization(self) -> None:
        """Test CLIToConfigMapper initialization."""
        mapper = CLIToConfigMapper()

        assert isinstance(mapper.arg_mapping, dict)
        assert "holding_time" in mapper.arg_mapping
        assert mapper.arg_mapping["holding_time"] == (
            "general_settings",
            "holding_time",
        )

    def test_map_args_to_config_basic(self) -> None:
        """Test basic argument mapping to configuration."""
        args = {"holding_time": 10, "network": "nsfnet", "c_band": 1}

        result = self.mapper.map_args_to_config(args)

        assert result["general_settings"]["holding_time"] == 10
        assert result["topology_settings"]["network"] == "nsfnet"
        assert result["spectrum_settings"]["c_band"] == 1

    def test_map_args_to_config_none_values_ignored(self) -> None:
        """Test that None values are ignored during mapping."""
        args = {"holding_time": 10, "network": None, "c_band": 1}

        result = self.mapper.map_args_to_config(args)

        assert result["general_settings"]["holding_time"] == 10
        assert "network" not in result.get("topology_settings", {})
        assert result["spectrum_settings"]["c_band"] == 1

    def test_map_args_to_config_unmapped_arguments(self) -> None:
        """Test handling of unmapped arguments."""
        args = {"holding_time": 10, "unknown_arg": "value"}

        result = self.mapper.map_args_to_config(args)

        assert result["general_settings"]["holding_time"] == 10
        assert result["general_settings"]["unknown_arg"] == "value"

    def test_map_args_to_config_empty_args(self) -> None:
        """Test mapping empty arguments dictionary."""
        result = self.mapper.map_args_to_config({})

        assert result == {}

    def test_map_args_to_config_all_sections(self) -> None:
        """Test mapping arguments to all configuration sections."""
        args = {
            "holding_time": 10,  # general_settings
            "network": "nsfnet",  # topology_settings
            "c_band": 1,  # spectrum_settings
            "snr_type": "gn",  # snr_settings
            "device": "cpu",  # rl_settings
            "deploy_model": True,  # ml_settings
            "file_type": "csv",  # file_settings
        }

        result = self.mapper.map_args_to_config(args)

        assert "general_settings" in result
        assert "topology_settings" in result
        assert "spectrum_settings" in result
        assert "snr_settings" in result
        assert "rl_settings" in result
        assert "ml_settings" in result
        assert "file_settings" in result

    def test_map_args_to_config_error_handling(self) -> None:
        """Test error handling during argument mapping."""
        # Mock an exception during processing
        with pytest.raises(ConfigTypeConversionError) as exc_info:
            # Create a scenario that would cause an exception
            # by patching the internal method to raise an exception
            args = {"test_arg": "value"}

            # Temporarily modify the mapper to cause an exception
            original_mapping = self.mapper.arg_mapping
            self.mapper.arg_mapping = Mock()
            self.mapper.arg_mapping.__contains__ = Mock(side_effect=Exception("Test error"))

            try:
                self.mapper.map_args_to_config(args)
            finally:
                # Restore original mapping
                self.mapper.arg_mapping = original_mapping

        assert "Failed to map argument" in str(exc_info.value)

    def test_map_namespace_to_config_success(self) -> None:
        """Test mapping argparse Namespace to configuration."""
        namespace = argparse.Namespace()
        namespace.holding_time = 10
        namespace.network = "nsfnet"

        result = self.mapper.map_namespace_to_config(namespace)

        assert result["general_settings"]["holding_time"] == 10
        assert result["topology_settings"]["network"] == "nsfnet"

    def test_map_namespace_to_config_invalid_namespace(self) -> None:
        """Test mapping invalid Namespace object."""

        # Create an object that doesn't have __dict__ attribute
        class InvalidNamespace:
            __slots__ = ["attr"]

        invalid_namespace = InvalidNamespace()

        with pytest.raises(TypeError) as exc_info:
            self.mapper.map_namespace_to_config(invalid_namespace)  # type: ignore

        assert "vars() argument must have __dict__ attribute" in str(exc_info.value)

    def test_get_cli_override_config_basic(self) -> None:
        """Test getting configuration with CLI overrides."""
        base_config = {
            "general_settings": {"holding_time": 5, "num_requests": 100},
            "topology_settings": {"network": "oldnet"},
        }
        cli_args = {"holding_time": 10, "network": "nsfnet"}

        result = self.mapper.get_cli_override_config(cli_args, base_config)

        assert result["general_settings"]["holding_time"] == 10  # Overridden
        assert result["general_settings"]["num_requests"] == 100  # Preserved
        assert result["topology_settings"]["network"] == "nsfnet"  # Overridden

    def test_get_cli_override_config_new_sections(self) -> None:
        """Test CLI overrides creating new configuration sections."""
        base_config = {"general_settings": {"holding_time": 5}}
        cli_args = {
            "c_band": 1  # Should create spectrum_settings section
        }

        result = self.mapper.get_cli_override_config(cli_args, base_config)

        assert result["general_settings"]["holding_time"] == 5
        assert result["spectrum_settings"]["c_band"] == 1

    def test_get_cli_override_config_invalid_base_config(self) -> None:
        """Test CLI override with invalid base config type."""
        cli_args = {"holding_time": 10}
        invalid_base = "not a dict"

        with pytest.raises(TypeError) as exc_info:
            self.mapper.get_cli_override_config(cli_args, invalid_base)  # type: ignore

        assert "base_config must be a dictionary" in str(exc_info.value)

    def test_get_cli_override_config_nested_dict_copy(self) -> None:
        """Test that nested dictionaries are properly copied."""
        base_config = {"general_settings": {"nested": {"key": "value"}}}
        cli_args = {"holding_time": 10}

        result = self.mapper.get_cli_override_config(cli_args, base_config)

        # Verify the copy was made correctly first
        assert result["general_settings"]["nested"]["key"] == "value"

        # The implementation does a shallow copy, so nested dicts are shared
        # This test verifies the current behavior
        original_value = base_config["general_settings"]["nested"]["key"]
        result["general_settings"]["nested"]["key"] = "modified"

        # With shallow copy, nested dicts are shared, so both change
        assert base_config["general_settings"]["nested"]["key"] == "modified"
        # Restore for cleanup
        base_config["general_settings"]["nested"]["key"] = original_value

    def test_get_reverse_mapping(self) -> None:
        """Test getting reverse mapping from config paths to CLI arguments."""
        reverse_map = self.mapper.get_reverse_mapping()

        assert isinstance(reverse_map, dict)
        assert "general_settings.holding_time" in reverse_map
        assert reverse_map["general_settings.holding_time"] == "holding_time"
        assert "topology_settings.network" in reverse_map
        assert reverse_map["topology_settings.network"] == "network"

    def test_get_reverse_mapping_completeness(self) -> None:
        """Test that reverse mapping contains all forward mappings."""
        reverse_map = self.mapper.get_reverse_mapping()

        # Check that all forward mappings have reverse mappings
        for cli_arg, (section, key) in self.mapper.arg_mapping.items():
            config_path = f"{section}.{key}"
            assert config_path in reverse_map
            assert reverse_map[config_path] == cli_arg

    def test_mapping_consistency(self) -> None:
        """Test consistency of argument mappings."""
        # Verify that all mapped arguments point to valid sections
        valid_sections = {
            # Core settings (used by both legacy engine and orchestrator)
            "general_settings",
            "topology_settings",
            "spectrum_settings",
            "snr_settings",
            "file_settings",
            # Legacy settings (to be phased out)
            "rl_settings",
            "ml_settings",
            # Orchestrator settings (v6.0+)
            "policy_settings",
            "heuristic_settings",
            "protection_settings",
            "routing_settings",
        }

        for cli_arg, (section, key) in self.mapper.arg_mapping.items():
            assert section in valid_sections, f"Invalid section '{section}' for arg '{cli_arg}'"
            assert isinstance(key, str), f"Key must be string for arg '{cli_arg}'"
            assert len(key) > 0, f"Key cannot be empty for arg '{cli_arg}'"

    def test_specific_argument_mappings(self) -> None:
        """Test specific critical argument mappings."""
        # Test a few important mappings explicitly
        assert self.mapper.arg_mapping["holding_time"] == (
            "general_settings",
            "holding_time",
        )
        assert self.mapper.arg_mapping["network"] == ("topology_settings", "network")
        assert self.mapper.arg_mapping["snr_type"] == ("snr_settings", "snr_type")
        assert self.mapper.arg_mapping["device"] == ("rl_settings", "device")
        assert self.mapper.arg_mapping["deploy_model"] == (
            "ml_settings",
            "deploy_model",
        )
        assert self.mapper.arg_mapping["file_type"] == ("file_settings", "file_type")

    def test_complex_cli_args_mapping(self) -> None:
        """Test mapping complex CLI arguments."""
        args = {
            "holding_time": 10.5,
            "thread_erlangs": True,
            "erlang_start": 100,
            "cores_per_link": 4,
            "bi_directional": False,
            "alpha_start": 0.1,
            "epsilon_end": 0.05,
        }

        result = self.mapper.map_args_to_config(args)

        # Verify correct section placement
        assert result["general_settings"]["holding_time"] == 10.5
        assert result["general_settings"]["thread_erlangs"] is True
        assert result["general_settings"]["erlang_start"] == 100
        assert result["topology_settings"]["cores_per_link"] == 4
        assert result["snr_settings"]["bi_directional"] is False
        assert result["rl_settings"]["alpha_start"] == 0.1
        assert result["rl_settings"]["epsilon_end"] == 0.05
