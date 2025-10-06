"""Unit tests for fusion.configs.config module."""

import json
import os
import tempfile
from unittest.mock import Mock, patch

import pytest

from fusion.configs.config import ConfigManager, SimulationConfig
from fusion.configs.errors import (
    ConfigError,
    ConfigFileNotFoundError,
    ConfigParseError,
    ConfigTypeConversionError,
)


class TestSimulationConfig:
    """Tests for SimulationConfig dataclass."""

    def test_simulation_config_initialization(self) -> None:
        """Test SimulationConfig can be created with all sections."""
        config = SimulationConfig(
            general={"holding_time": 10},
            topology={"network": "nsfnet"},
            spectrum={"c_band": 1},
            snr={"snr_type": "gn"},
            rl={"device": "cpu"},
            ml={"deploy_model": False},
            file={"file_type": "csv"},
        )

        assert config.general == {"holding_time": 10}
        assert config.topology == {"network": "nsfnet"}
        assert config.spectrum == {"c_band": 1}
        assert config.snr == {"snr_type": "gn"}
        assert config.rl == {"device": "cpu"}
        assert config.ml == {"deploy_model": False}
        assert config.file == {"file_type": "csv"}


class TestConfigManager:
    """Tests for ConfigManager class."""

    def test_init_without_config_path(self) -> None:
        """Test ConfigManager initialization without config path."""
        manager = ConfigManager()

        assert manager.config_path is None
        assert manager.schema_validator is None
        assert manager._config is None
        assert manager._raw_config == {}

    def test_init_with_schema_dir(self) -> None:
        """Test ConfigManager initialization with schema directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConfigManager(schema_dir=temp_dir)

            assert manager.schema_validator is not None

    def test_init_with_existing_config_path(self) -> None:
        """Test ConfigManager initialization with existing config file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ini", delete=False) as f:
            f.write("[general_settings]\nholding_time = 10\n")
            config_path = f.name

        try:
            with patch.object(ConfigManager, "load_config") as mock_load:
                ConfigManager(config_path=config_path)
                mock_load.assert_called_once_with(config_path)
        finally:
            os.unlink(config_path)

    def test_load_config_file_not_found(self) -> None:
        """Test loading non-existent configuration file."""
        manager = ConfigManager()

        with pytest.raises(ConfigFileNotFoundError) as exc_info:
            manager.load_config("nonexistent.ini")

        assert "Configuration file not found" in str(exc_info.value)
        assert "nonexistent.ini" in str(exc_info.value)

    def test_load_config_unsupported_format(self) -> None:
        """Test loading configuration file with unsupported format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("some content")
            config_path = f.name

        try:
            manager = ConfigManager()

            with pytest.raises(ConfigParseError) as exc_info:
                manager.load_config(config_path)

            assert "Unsupported configuration file format" in str(exc_info.value)
        finally:
            os.unlink(config_path)

    def test_load_ini_config_success(self) -> None:
        """Test successful loading of INI configuration."""
        ini_content = """[general_settings]
holding_time = 10
thread_erlangs = true
num_requests = 100

[topology_settings]
network = nsfnet
cores_per_link = 1
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ini", delete=False) as f:
            f.write(ini_content)
            config_path = f.name

        try:
            manager = ConfigManager()
            config = manager.load_config(config_path)

            assert isinstance(config, SimulationConfig)
            assert config.general["holding_time"] == 10
            assert config.general["thread_erlangs"] is True
            assert config.general["num_requests"] == 100
            assert config.topology["network"] == "nsfnet"
            assert config.topology["cores_per_link"] == 1
        finally:
            os.unlink(config_path)

    def test_load_ini_config_with_json_values(self) -> None:
        """Test loading INI config with JSON values."""
        ini_content = """[general_settings]
request_distribution = {"uniform": 0.5, "poisson": 0.5}
phi = [1, 2, 3]
none_value = null
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ini", delete=False) as f:
            f.write(ini_content)
            config_path = f.name

        try:
            manager = ConfigManager()
            config = manager.load_config(config_path)

            assert config.general["request_distribution"] == {
                "uniform": 0.5,
                "poisson": 0.5,
            }
            assert config.general["phi"] == [1, 2, 3]
            assert config.general["none_value"] is None
        finally:
            os.unlink(config_path)

    def test_load_json_config_success(self) -> None:
        """Test successful loading of JSON configuration."""
        json_content = {
            "general_settings": {"holding_time": 10},
            "topology_settings": {"network": "nsfnet"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(json_content, f)
            config_path = f.name

        try:
            manager = ConfigManager()
            config = manager.load_config(config_path)

            assert config.general["holding_time"] == 10
            assert config.topology["network"] == "nsfnet"
        finally:
            os.unlink(config_path)

    def test_load_json_config_invalid(self) -> None:
        """Test loading invalid JSON configuration."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"invalid": json}')
            config_path = f.name

        try:
            manager = ConfigManager()

            with pytest.raises(ConfigParseError) as exc_info:
                manager.load_config(config_path)

            assert "Invalid JSON format" in str(exc_info.value)
        finally:
            os.unlink(config_path)

    @patch("fusion.configs.config.yaml")
    def test_load_yaml_config_success(self, mock_yaml: Mock) -> None:
        """Test successful loading of YAML configuration."""
        yaml_content = {
            "general_settings": {"holding_time": 10},
            "topology_settings": {"network": "nsfnet"},
        }
        mock_yaml.safe_load.return_value = yaml_content

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("content")
            config_path = f.name

        try:
            manager = ConfigManager()
            config = manager.load_config(config_path)

            assert config.general["holding_time"] == 10
            assert config.topology["network"] == "nsfnet"
        finally:
            os.unlink(config_path)

    def test_load_yaml_config_without_yaml_library(self) -> None:
        """Test loading YAML config when PyYAML is not available."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("content")
            config_path = f.name

        try:
            with patch("fusion.configs.config.yaml", None):
                manager = ConfigManager()

                with pytest.raises(ConfigParseError) as exc_info:
                    manager.load_config(config_path)

                assert "PyYAML is required" in str(exc_info.value)
        finally:
            os.unlink(config_path)

    def test_load_config_with_schema_validation(self) -> None:
        """Test loading config with schema validation."""
        ini_content = "[general_settings]\nholding_time = 10\n"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".ini", delete=False) as f:
            f.write(ini_content)
            config_path = f.name

        try:
            mock_validator = Mock()
            manager = ConfigManager()
            manager.schema_validator = mock_validator

            manager.load_config(config_path)

            mock_validator.validate.assert_called_once()
        finally:
            os.unlink(config_path)

    def test_load_config_schema_validation_failure(self) -> None:
        """Test config loading with schema validation failure."""
        ini_content = "[general_settings]\nholding_time = 10\n"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".ini", delete=False) as f:
            f.write(ini_content)
            config_path = f.name

        try:
            mock_validator = Mock()
            mock_validator.validate.side_effect = Exception("Validation failed")
            manager = ConfigManager()
            manager.schema_validator = mock_validator

            with pytest.raises(ConfigError) as exc_info:
                manager.load_config(config_path)

            assert "Configuration validation failed" in str(exc_info.value)
        finally:
            os.unlink(config_path)

    def test_get_config_when_loaded(self) -> None:
        """Test getting configuration when one is loaded."""
        manager = ConfigManager()
        test_config = SimulationConfig(
            general={}, topology={}, spectrum={}, snr={}, rl={}, ml={}, file={}
        )
        manager._config = test_config

        result = manager.get_config()

        assert result is test_config

    def test_get_config_when_not_loaded(self) -> None:
        """Test getting configuration when none is loaded."""
        manager = ConfigManager()

        result = manager.get_config()

        assert result is None

    def test_get_module_config_routing(self) -> None:
        """Test getting routing module configuration."""
        manager = ConfigManager()
        manager._config = SimulationConfig(
            general={"k_paths": 3},
            topology={"network": "nsfnet"},
            spectrum={},
            snr={},
            rl={},
            ml={},
            file={},
        )

        result = manager.get_module_config("routing")

        assert result["k_paths"] == 3
        assert result["network"] == "nsfnet"

    def test_get_module_config_spectrum(self) -> None:
        """Test getting spectrum module configuration."""
        manager = ConfigManager()
        manager._config = SimulationConfig(
            general={},
            topology={},
            spectrum={"c_band": 1},
            snr={},
            rl={},
            ml={},
            file={},
        )

        result = manager.get_module_config("spectrum")

        assert result["c_band"] == 1

    def test_get_module_config_unknown_module(self) -> None:
        """Test getting configuration for unknown module."""
        manager = ConfigManager()
        manager._config = SimulationConfig(
            general={}, topology={}, spectrum={}, snr={}, rl={}, ml={}, file={}
        )

        result = manager.get_module_config("unknown")

        assert result == {}

    def test_get_module_config_no_config_loaded(self) -> None:
        """Test getting module configuration when no config is loaded."""
        manager = ConfigManager()

        result = manager.get_module_config("spectrum")

        assert result == {}

    def test_save_config_no_config_loaded(self) -> None:
        """Test saving configuration when none is loaded."""
        manager = ConfigManager()

        with pytest.raises(ValueError) as exc_info:
            manager.save_config("output.ini")

        assert "No configuration loaded to save" in str(exc_info.value)

    def test_save_config_ini_format(self) -> None:
        """Test saving configuration in INI format."""
        manager = ConfigManager()
        manager._raw_config = {
            "general_settings": {"holding_time": 10, "complex_data": [1, 2, 3]},
            "topology_settings": {"network": "nsfnet"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".ini", delete=False) as f:
            config_path = f.name

        try:
            manager.save_config(config_path, "ini")

            assert os.path.exists(config_path)

            # Verify content
            import configparser

            saved_config = configparser.ConfigParser()
            saved_config.read(config_path)

            assert saved_config["general_settings"]["holding_time"] == "10"
            assert json.loads(saved_config["general_settings"]["complex_data"]) == [
                1,
                2,
                3,
            ]
        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)

    def test_save_config_json_format(self) -> None:
        """Test saving configuration in JSON format."""
        manager = ConfigManager()
        manager._raw_config = {"general_settings": {"holding_time": 10}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config_path = f.name

        try:
            manager.save_config(config_path, "json")

            assert os.path.exists(config_path)

            with open(config_path) as f:
                saved_data = json.load(f)

            assert saved_data["general_settings"]["holding_time"] == 10
        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)

    @patch("fusion.configs.config.yaml")
    def test_save_config_yaml_format(self, mock_yaml: Mock) -> None:
        """Test saving configuration in YAML format."""
        manager = ConfigManager()
        manager._raw_config = {"general_settings": {"holding_time": 10}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config_path = f.name

        try:
            manager.save_config(config_path, "yaml")

            mock_yaml.dump.assert_called_once()
        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)

    def test_save_config_yaml_without_library(self) -> None:
        """Test saving YAML config when PyYAML is not available."""
        manager = ConfigManager()
        manager._raw_config = {"test": "data"}

        with patch("fusion.configs.config.yaml", None):
            with pytest.raises(ConfigError) as exc_info:
                manager.save_config("output.yaml", "yaml")

            assert "PyYAML is required" in str(exc_info.value)

    def test_save_config_unsupported_format(self) -> None:
        """Test saving configuration with unsupported format."""
        manager = ConfigManager()
        manager._raw_config = {"test": "data"}

        with pytest.raises(ConfigError) as exc_info:
            manager.save_config("output.txt", "txt")

        assert "Unsupported format" in str(exc_info.value)

    def test_update_config_new_section(self) -> None:
        """Test updating configuration with new section."""
        manager = ConfigManager()
        manager._raw_config = {}

        manager.update_config("new_section", "key", "value")

        assert manager._raw_config["new_section"]["key"] == "value"

    def test_update_config_existing_section(self) -> None:
        """Test updating configuration in existing section."""
        manager = ConfigManager()
        manager._raw_config = {"general_settings": {"old_key": "old_value"}}
        manager._config = SimulationConfig(
            general={"old_key": "old_value"},
            topology={},
            spectrum={},
            snr={},
            rl={},
            ml={},
            file={},
        )

        manager.update_config("general_settings", "new_key", "new_value")

        assert manager._raw_config["general_settings"]["new_key"] == "new_value"
        assert manager._raw_config["general_settings"]["old_key"] == "old_value"
        assert manager._config.general["new_key"] == "new_value"

    @patch("fusion.configs.config.CLIToConfigMapper")
    def test_merge_cli_args_success(self, mock_mapper_class: Mock) -> None:
        """Test successful merging of CLI arguments."""
        mock_mapper = Mock()
        mock_mapper.map_args_to_config.return_value = {
            "general_settings": {"holding_time": 15}
        }
        mock_mapper_class.return_value = mock_mapper

        manager = ConfigManager()
        manager._raw_config = {"general_settings": {"network": "nsfnet"}}

        manager.merge_cli_args({"holding_time": 15})

        assert manager._raw_config["general_settings"]["holding_time"] == 15
        assert manager._raw_config["general_settings"]["network"] == "nsfnet"

    @patch("fusion.configs.config.CLIToConfigMapper")
    def test_merge_cli_args_failure(self, mock_mapper_class: Mock) -> None:
        """Test CLI argument merging failure."""
        mock_mapper = Mock()
        mock_mapper.map_args_to_config.side_effect = Exception("Mapping failed")
        mock_mapper_class.return_value = mock_mapper

        manager = ConfigManager()

        with pytest.raises(ConfigTypeConversionError) as exc_info:
            manager.merge_cli_args({"invalid": "arg"})

        assert "Failed to merge CLI arguments" in str(exc_info.value)

    def test_create_config_object(self) -> None:
        """Test creating SimulationConfig object from raw config."""
        manager = ConfigManager()
        raw_config = {
            "general_settings": {"holding_time": 10},
            "topology_settings": {"network": "nsfnet"},
            "spectrum_settings": {"c_band": 1},
            "snr_settings": {"snr_type": "gn"},
            "rl_settings": {"device": "cpu"},
            "ml_settings": {"deploy_model": False},
            "file_settings": {"file_type": "csv"},
        }

        config = manager._create_config_object(raw_config)

        assert isinstance(config, SimulationConfig)
        assert config.general == {"holding_time": 10}
        assert config.topology == {"network": "nsfnet"}
        assert config.spectrum == {"c_band": 1}
        assert config.snr == {"snr_type": "gn"}
        assert config.rl == {"device": "cpu"}
        assert config.ml == {"deploy_model": False}
        assert config.file == {"file_type": "csv"}

    def test_create_config_object_missing_sections(self) -> None:
        """Test creating config object with missing sections."""
        manager = ConfigManager()
        raw_config = {"general_settings": {"holding_time": 10}}

        config = manager._create_config_object(raw_config)

        assert config.general == {"holding_time": 10}
        assert config.topology == {}
        assert config.spectrum == {}
        assert config.snr == {}
        assert config.rl == {}
        assert config.ml == {}
        assert config.file == {}
