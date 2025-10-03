"""Unit tests for config_setup module."""

from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

from fusion.cli.config_setup import (
    ConfigManager,
    _copy_dict_vals,
    _find_category,
    _process_optional_options,
    _process_required_options,
    _read_config_file,
    _resolve_config_path,
    _setup_threads,
    _validate_config_structure,
    load_and_validate_config,
    load_config,
    normalize_config_path,
    setup_config_from_cli,
)
from fusion.configs.constants import DEFAULT_THREAD_NAME, REQUIRED_SECTION
from fusion.configs.errors import (
    ConfigError,
    ConfigFileNotFoundError,
    ConfigParseError,
    MissingRequiredOptionError,
)


class TestNormalizeConfigPath:
    """Tests for normalize_config_path function."""

    def test_normalize_config_path_with_absolute_path(self, tmp_path: Path) -> None:
        """Test normalize_config_path with absolute path."""
        config_file = tmp_path / "config.ini"
        config_file.touch()

        result = normalize_config_path(str(config_file))

        assert Path(result).is_absolute()
        assert Path(result).exists()

    def test_normalize_config_path_with_relative_path(self, tmp_path: Path) -> None:
        """Test normalize_config_path converts relative to absolute path."""
        config_file = tmp_path / "config.ini"
        config_file.touch()

        with patch("os.path.isabs", return_value=False):
            with patch("os.path.expanduser", return_value="config.ini"):
                with patch("os.path.abspath", return_value=str(config_file)):
                    with patch("os.path.exists", return_value=True):
                        result = normalize_config_path("config.ini")

                        assert Path(result).is_absolute()

    def test_normalize_config_path_with_nonexistent_file_raises_error(self) -> None:
        """Test normalize_config_path raises error for nonexistent file."""
        with pytest.raises(ConfigFileNotFoundError) as exc_info:
            normalize_config_path("/path/to/nonexistent.ini")
        assert "Could not find config file" in str(exc_info.value)

    def test_normalize_config_path_expands_user_path(self, tmp_path: Path) -> None:
        """Test normalize_config_path expands ~ in path."""
        config_file = tmp_path / "config.ini"
        config_file.touch()

        with patch("os.path.expanduser") as mock_expand:
            mock_expand.return_value = str(config_file)

            result = normalize_config_path("~/config.ini")

            mock_expand.assert_called_once_with("~/config.ini")
            assert Path(result).exists()


class TestSetupConfigFromCli:
    """Tests for setup_config_from_cli function."""

    @patch("fusion.cli.config_setup.load_config")
    def test_setup_config_from_cli_with_valid_config(
        self, mock_load_config: Any, sample_args: Any, valid_config_dict: dict[str, Any]
    ) -> None:
        """Test setup_config_from_cli returns config dict on success."""
        mock_load_config.return_value = valid_config_dict

        result = setup_config_from_cli(sample_args)

        assert result == valid_config_dict
        mock_load_config.assert_called_once()

    @patch("fusion.cli.config_setup.load_config")
    def test_setup_config_from_cli_handles_config_errors(
        self, mock_load_config: Any, sample_args: Any, mock_logger: Any
    ) -> None:
        """Test setup_config_from_cli handles configuration errors gracefully."""
        mock_load_config.side_effect = ConfigFileNotFoundError("File not found")

        result = setup_config_from_cli(sample_args)

        assert result == {}

    @patch("fusion.cli.config_setup.load_config")
    def test_setup_config_from_cli_handles_unexpected_errors(
        self, mock_load_config: Any, sample_args: Any, mock_logger: Any
    ) -> None:
        """Test setup_config_from_cli handles unexpected errors."""
        mock_load_config.side_effect = ValueError("Unexpected error")

        result = setup_config_from_cli(sample_args)

        assert result == {}


class TestProcessRequiredOptions:
    """Tests for _process_required_options function."""

    def test_process_required_options_with_valid_config(
        self,
        mock_config_parser: Any,
        valid_config_dict: dict[str, Any],
        sample_args_dict: dict[str, Any],
    ) -> None:
        """Test _process_required_options processes all required options."""
        config_dict: dict[str, dict[str, Any]] = {DEFAULT_THREAD_NAME: {}}
        required_dict = {"sim": {"test_option": str}}
        optional_dict: dict[str, dict[str, Any]] = {}

        _process_required_options(
            mock_config_parser,
            config_dict,
            required_dict,
            optional_dict,
            sample_args_dict,
        )

        assert "test_option" in config_dict[DEFAULT_THREAD_NAME]

    def test_process_required_options_with_missing_option_raises_error(
        self, mock_config_parser: Any, sample_args_dict: dict[str, Any]
    ) -> None:
        """Test _process_required_options raises error for missing required option."""
        mock_config_parser.has_option.return_value = False
        config_dict: dict[str, dict[str, Any]] = {DEFAULT_THREAD_NAME: {}}
        required_dict = {"sim": {"missing_option": str}}

        with pytest.raises(MissingRequiredOptionError) as exc_info:
            _process_required_options(
                mock_config_parser, config_dict, required_dict, {}, sample_args_dict
            )
        assert "Missing required option" in str(exc_info.value)


class TestProcessOptionalOptions:
    """Tests for _process_optional_options function."""

    def test_process_optional_options_with_valid_config(
        self, mock_config_parser: Any, sample_args_dict: dict[str, Any]
    ) -> None:
        """Test _process_optional_options processes optional options correctly."""
        config_dict: dict[str, dict[str, Any]] = {DEFAULT_THREAD_NAME: {}}
        optional_dict = {"sim": {"optional_test": str}}
        mock_config_parser.__getitem__.return_value = {"optional_test": "test_value"}

        _process_optional_options(
            mock_config_parser, config_dict, optional_dict, sample_args_dict
        )

        assert config_dict[DEFAULT_THREAD_NAME]["optional_test"] == "test_value"

    def test_process_optional_options_with_missing_option_sets_none(
        self, mock_config_parser: Any, sample_args_dict: dict[str, Any]
    ) -> None:
        """Test _process_optional_options sets None for missing optional options."""
        config_dict: dict[str, dict[str, Any]] = {DEFAULT_THREAD_NAME: {}}
        optional_dict = {"sim": {"missing_optional": str}}
        mock_config_parser.__getitem__.return_value = {}

        _process_optional_options(
            mock_config_parser, config_dict, optional_dict, sample_args_dict
        )

        assert config_dict[DEFAULT_THREAD_NAME]["missing_optional"] is None


class TestValidateConfigStructure:
    """Tests for _validate_config_structure function."""

    def test_validate_config_structure_with_valid_config(
        self, mock_config_parser: Any
    ) -> None:
        """Test _validate_config_structure passes with valid structure."""
        mock_config_parser.has_section.return_value = True

        _validate_config_structure(mock_config_parser)

    def test_validate_config_structure_with_missing_section_raises_error(
        self, mock_config_parser: Any
    ) -> None:
        """Test _validate_config_structure raises error for missing required section."""
        mock_config_parser.has_section.return_value = False

        with pytest.raises(ConfigParseError) as exc_info:
            _validate_config_structure(mock_config_parser)
        assert f"Missing '{REQUIRED_SECTION}' section" in str(exc_info.value)


class TestReadConfigFile:
    """Tests for _read_config_file function."""

    def test_read_config_file_with_valid_file(self, temp_config_file: Any) -> None:
        """Test _read_config_file reads valid config file successfully."""
        config = _read_config_file(str(temp_config_file))

        assert config.has_section("general_settings")
        assert config["general_settings"]["run_id"] == "test_run"

    def test_read_config_file_with_parse_error_raises_exception(self) -> None:
        """Test _read_config_file raises ConfigParseError on parse failure."""
        with patch("configparser.ConfigParser.read") as mock_read:
            mock_read.side_effect = Exception("Parse error")

            with pytest.raises(ConfigParseError) as exc_info:
                _read_config_file("bad_config.ini")
            assert "Failed to parse config file" in str(exc_info.value)


class TestResolveConfigPath:
    """Tests for _resolve_config_path function."""

    def test_resolve_config_path_with_none_returns_default(self) -> None:
        """Test _resolve_config_path returns default path when None provided."""
        with patch("os.path.exists") as mock_exists:
            mock_exists.return_value = True

            result = _resolve_config_path(None)

            assert "config.ini" in result

    def test_resolve_config_path_with_valid_path(self, temp_config_file: Any) -> None:
        """Test _resolve_config_path returns normalized path for valid file."""
        with patch("fusion.cli.config_setup.normalize_config_path") as mock_normalize:
            mock_normalize.return_value = str(temp_config_file)

            result = _resolve_config_path(str(temp_config_file))

            assert result == str(temp_config_file)

    def test_resolve_config_path_with_nonexistent_file_raises_error(self) -> None:
        """Test _resolve_config_path raises error for nonexistent file."""
        with patch("fusion.cli.config_setup.normalize_config_path") as mock_normalize:
            mock_normalize.return_value = "/nonexistent/path.ini"

            with pytest.raises(ConfigFileNotFoundError):
                _resolve_config_path("/nonexistent/path.ini")


class TestLoadConfig:
    """Tests for load_config function."""

    @patch("fusion.cli.config_setup._process_optional_options")
    @patch("fusion.cli.config_setup._process_required_options")
    @patch("fusion.cli.config_setup._validate_config_structure")
    @patch("fusion.cli.config_setup._read_config_file")
    @patch("fusion.cli.config_setup._resolve_config_path")
    def test_load_config_with_valid_file_returns_dict(
        self,
        mock_resolve: Any,
        mock_read: Any,
        _mock_validate: Any,
        _mock_process_req: Any,
        _mock_process_opt: Any,
        temp_config_file: Any,
        sample_args_dict: dict[str, Any],
    ) -> None:
        """Test load_config returns valid configuration dictionary."""
        mock_resolve.return_value = str(temp_config_file)
        mock_config = Mock()
        mock_config.sections.return_value = []
        mock_read.return_value = mock_config

        result = load_config(str(temp_config_file), sample_args_dict)

        assert isinstance(result, dict)
        assert DEFAULT_THREAD_NAME in result

    def test_load_config_with_invalid_file_returns_empty_dict(
        self, invalid_config_file: Any, sample_args_dict: dict[str, Any]
    ) -> None:
        """Test load_config returns empty dict for invalid config."""
        result = load_config(str(invalid_config_file), sample_args_dict)

        assert result == {}

    @patch("fusion.cli.config_setup._setup_threads")
    @patch("fusion.cli.config_setup._process_optional_options")
    @patch("fusion.cli.config_setup._process_required_options")
    @patch("fusion.cli.config_setup._validate_config_structure")
    @patch("fusion.cli.config_setup._read_config_file")
    @patch("fusion.cli.config_setup._resolve_config_path")
    def test_load_config_with_multi_thread_config(
        self,
        mock_resolve: Any,
        mock_read: Any,
        _mock_validate: Any,
        _mock_process_req: Any,
        _mock_process_opt: Any,
        mock_setup_threads: Any,
        multi_thread_config_file: Any,
        sample_args_dict: dict[str, Any],
    ) -> None:
        """Test load_config handles multi-threaded configurations."""
        mock_resolve.return_value = str(multi_thread_config_file)
        mock_config = Mock()
        mock_config.sections.return_value = ["s2", "s3"]
        mock_read.return_value = mock_config

        # Mock the setup_threads function to return expected multi-thread structure
        mock_setup_threads.return_value = {
            DEFAULT_THREAD_NAME: {"run_id": "test_run"},
            "s2": {"simulation_time": 2000},
            "s3": {"simulation_time": 3000},
        }

        result = load_config(str(multi_thread_config_file), sample_args_dict)

        assert DEFAULT_THREAD_NAME in result
        assert "s2" in result
        assert "s3" in result
        assert result["s2"]["simulation_time"] == 2000
        assert result["s3"]["simulation_time"] == 3000


class TestSetupThreads:
    """Tests for _setup_threads function."""

    def test_setup_threads_copies_default_values(self) -> None:
        """Test _setup_threads copies default thread values to new threads."""
        config = Mock()
        config.items.return_value = [("simulation_time", "2000")]
        config_dict = {
            DEFAULT_THREAD_NAME: {"run_id": "default", "simulation_time": 1000}
        }
        section_list = ["s2"]
        types_dict = {"sim": {"simulation_time": int}}

        result = _setup_threads(config, config_dict, section_list, types_dict, {}, {})

        assert "s2" in result
        assert result["s2"]["run_id"] == "default"


class TestCopyDictVals:
    """Tests for _copy_dict_vals function."""

    def test_copy_dict_vals_creates_copy_with_new_key(self) -> None:
        """Test _copy_dict_vals creates a copy of values under new key."""
        dictionary = {DEFAULT_THREAD_NAME: {"key1": "value1", "key2": "value2"}}

        result = _copy_dict_vals("s2", dictionary)

        assert "s2" in result
        assert result["s2"]["key1"] == "value1"
        assert result["s2"]["key2"] == "value2"
        assert result["s2"] is not result[DEFAULT_THREAD_NAME]


class TestFindCategory:
    """Tests for _find_category function."""

    def test_find_category_returns_correct_category(self) -> None:
        """Test _find_category finds the correct category for a key."""
        category_dict: dict[str, dict[str, Any]] = {
            "network": {"topology": str, "nodes": int},
            "simulation": {"time": int, "steps": int},
        }

        assert _find_category(category_dict, "topology") == "network"
        assert _find_category(category_dict, "steps") == "simulation"
        assert _find_category(category_dict, "unknown") is None


class TestLoadAndValidateConfig:
    """Tests for load_and_validate_config function."""

    @patch("fusion.cli.config_setup.load_config")
    def test_load_and_validate_config_returns_config_dict(
        self, mock_load_config: Any, temp_config_file: Any
    ) -> None:
        """Test load_and_validate_config returns validated config dictionary."""
        mock_load_config.return_value = {DEFAULT_THREAD_NAME: {"run_id": "test"}}
        args = Mock(config_path=str(temp_config_file))

        result = load_and_validate_config(args)

        assert isinstance(result, dict)
        assert DEFAULT_THREAD_NAME in result


class TestConfigManager:
    """Tests for ConfigManager class."""

    def test_config_manager_initialization(
        self, valid_config_dict: dict[str, Any], sample_args: Any
    ) -> None:
        """Test ConfigManager initializes correctly with config and args."""
        manager = ConfigManager(valid_config_dict, sample_args)

        assert manager._config == valid_config_dict
        assert manager._args == sample_args

    @patch("fusion.cli.config_setup.load_config")
    def test_config_manager_from_args_creates_instance(
        self, mock_load_config: Any, temp_config_file: Any
    ) -> None:
        """Test ConfigManager.from_args creates instance from arguments."""
        mock_load_config.return_value = {DEFAULT_THREAD_NAME: {"run_id": "test"}}
        args = Mock(config_path=str(temp_config_file))

        manager = ConfigManager.from_args(args)

        assert isinstance(manager, ConfigManager)
        assert DEFAULT_THREAD_NAME in manager._config

    def test_config_manager_from_args_handles_config_error(self) -> None:
        """Test ConfigManager.from_args re-raises ConfigError."""
        args = Mock(config_path="/nonexistent/file.ini")

        with patch("fusion.cli.config_setup.load_config") as mock_load:
            mock_load.side_effect = ConfigError("Test error")

            with pytest.raises(ConfigError):
                ConfigManager.from_args(args)

    def test_config_manager_from_file_creates_instance(
        self, temp_config_file: Any, sample_args_dict: dict[str, Any]
    ) -> None:
        """Test ConfigManager.from_file creates instance from file path."""
        manager = ConfigManager.from_file(str(temp_config_file), sample_args_dict)

        assert isinstance(manager, ConfigManager)
        assert hasattr(manager._args, "config_path")

    def test_config_manager_as_dict_returns_config(
        self, valid_config_dict: dict[str, Any], sample_args: Any
    ) -> None:
        """Test ConfigManager.as_dict returns configuration dictionary."""
        manager = ConfigManager(valid_config_dict, sample_args)

        result = manager.as_dict()

        assert result == valid_config_dict

    def test_config_manager_get_returns_thread_config(
        self, valid_config_dict: dict[str, Any], sample_args: Any
    ) -> None:
        """Test ConfigManager.get returns configuration for specified thread."""
        manager = ConfigManager(valid_config_dict, sample_args)

        result = manager.get(DEFAULT_THREAD_NAME)

        assert result == valid_config_dict[DEFAULT_THREAD_NAME]

    def test_config_manager_get_with_nonexistent_thread_returns_empty(
        self, valid_config_dict: dict[str, Any], sample_args: Any
    ) -> None:
        """Test ConfigManager.get returns empty dict for nonexistent thread."""
        manager = ConfigManager(valid_config_dict, sample_args)

        result = manager.get("nonexistent_thread")

        assert result == {}

    def test_config_manager_get_value_returns_correct_value(
        self, valid_config_dict: dict[str, Any], sample_args: Any
    ) -> None:
        """Test ConfigManager.get_value returns specific configuration value."""
        manager = ConfigManager(valid_config_dict, sample_args)

        result = manager.get_value("run_id", DEFAULT_THREAD_NAME)

        assert result == "test_run_001"

    def test_config_manager_get_value_with_default(
        self, valid_config_dict: dict[str, Any], sample_args: Any
    ) -> None:
        """Test ConfigManager.get_value returns default for missing key."""
        manager = ConfigManager(valid_config_dict, sample_args)

        result = manager.get_value(
            "nonexistent_key", DEFAULT_THREAD_NAME, "default_value"
        )

        assert result == "default_value"

    def test_config_manager_has_thread_returns_correct_boolean(
        self, valid_config_dict: dict[str, Any], sample_args: Any
    ) -> None:
        """Test ConfigManager.has_thread returns correct boolean values."""
        manager = ConfigManager(valid_config_dict, sample_args)

        assert manager.has_thread(DEFAULT_THREAD_NAME) is True
        assert manager.has_thread("nonexistent_thread") is False

    def test_config_manager_get_threads_returns_all_threads(self) -> None:
        """Test ConfigManager.get_threads returns list of all threads."""
        config_dict: dict[str, dict[str, Any]] = {
            DEFAULT_THREAD_NAME: {},
            "thread_1": {},
            "thread_2": {},
        }
        manager = ConfigManager(config_dict, Mock())

        threads = manager.get_threads()

        assert len(threads) == 3
        assert DEFAULT_THREAD_NAME in threads
        assert "thread_1" in threads
        assert "thread_2" in threads

    def test_config_manager_get_args_returns_args(
        self, valid_config_dict: dict[str, Any], sample_args: Any
    ) -> None:
        """Test ConfigManager.get_args returns stored arguments."""
        manager = ConfigManager(valid_config_dict, sample_args)

        result = manager.get_args()

        assert result == sample_args
