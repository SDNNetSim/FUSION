"""Unit tests for fusion.configs.registry module."""

import os
import tempfile
from unittest.mock import Mock, patch

import pytest

from fusion.configs.config import ConfigManager
from fusion.configs.errors import ConfigError
from fusion.configs.registry import ConfigRegistry


class TestConfigRegistry:
    """Tests for ConfigRegistry class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.templates_dir = os.path.join(self.temp_dir, "templates")
        self.schemas_dir = os.path.join(self.temp_dir, "schemas")
        os.makedirs(self.templates_dir)
        os.makedirs(self.schemas_dir)

    def teardown_method(self) -> None:
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_init_with_default_directories(self) -> None:
        """Test ConfigRegistry initialization with default directories."""
        registry = ConfigRegistry()

        assert registry.templates_dir.endswith("templates")
        assert registry.schemas_dir.endswith("schemas")
        assert isinstance(registry.validator, object)
        assert isinstance(registry._templates, dict)

    def test_init_with_custom_directories(self) -> None:
        """Test ConfigRegistry initialization with custom directories."""
        registry = ConfigRegistry(self.templates_dir, self.schemas_dir)

        assert registry.templates_dir == self.templates_dir
        assert registry.schemas_dir == self.schemas_dir

    def test_load_templates_empty_directory(self) -> None:
        """Test loading templates from empty directory."""
        registry = ConfigRegistry(self.templates_dir, self.schemas_dir)

        assert registry._templates == {}

    def test_load_templates_with_ini_files(self) -> None:
        """Test loading templates with INI files."""
        # Create test template files
        template1_path = os.path.join(self.templates_dir, "template1.ini")
        template2_path = os.path.join(self.templates_dir, "template2.ini")

        with open(template1_path, "w") as f:
            f.write("[general_settings]\nholding_time = 10\n")
        with open(template2_path, "w") as f:
            f.write("[general_settings]\nholding_time = 20\n")

        registry = ConfigRegistry(self.templates_dir, self.schemas_dir)

        assert "template1" in registry._templates
        assert "template2" in registry._templates
        assert registry._templates["template1"] == template1_path
        assert registry._templates["template2"] == template2_path

    def test_load_templates_ignores_non_ini_files(self) -> None:
        """Test that loading templates ignores non-INI files."""
        # Create files with different extensions
        ini_file = os.path.join(self.templates_dir, "template.ini")
        txt_file = os.path.join(self.templates_dir, "readme.txt")
        json_file = os.path.join(self.templates_dir, "config.json")

        with open(ini_file, "w") as f:
            f.write("[general_settings]\nholding_time = 10\n")
        with open(txt_file, "w") as f:
            f.write("This is a readme file")
        with open(json_file, "w") as f:
            f.write('{"key": "value"}')

        registry = ConfigRegistry(self.templates_dir, self.schemas_dir)

        assert "template" in registry._templates
        assert "readme" not in registry._templates
        assert "config" not in registry._templates

    def test_load_templates_nonexistent_directory(self) -> None:
        """Test loading templates from non-existent directory."""
        nonexistent_dir = os.path.join(self.temp_dir, "nonexistent")
        registry = ConfigRegistry(nonexistent_dir, self.schemas_dir)

        assert registry._templates == {}

    def test_list_templates_empty(self) -> None:
        """Test listing templates when none exist."""
        registry = ConfigRegistry(self.templates_dir, self.schemas_dir)

        templates = registry.list_templates()

        assert templates == []

    def test_list_templates_with_templates(self) -> None:
        """Test listing templates when templates exist."""
        # Create test templates
        for i in range(3):
            template_path = os.path.join(self.templates_dir, f"template{i}.ini")
            with open(template_path, "w") as f:
                f.write(f"[general_settings]\nholding_time = {i * 10}\n")

        registry = ConfigRegistry(self.templates_dir, self.schemas_dir)
        templates = registry.list_templates()

        assert len(templates) == 3
        assert "template0" in templates
        assert "template1" in templates
        assert "template2" in templates

    def test_get_template_path_existing(self) -> None:
        """Test getting path for existing template."""
        template_path = os.path.join(self.templates_dir, "test_template.ini")
        with open(template_path, "w") as f:
            f.write("[general_settings]\nholding_time = 10\n")

        registry = ConfigRegistry(self.templates_dir, self.schemas_dir)
        result = registry.get_template_path("test_template")

        assert result == template_path

    def test_get_template_path_nonexistent(self) -> None:
        """Test getting path for non-existent template."""
        registry = ConfigRegistry(self.templates_dir, self.schemas_dir)
        result = registry.get_template_path("nonexistent")

        assert result is None

    @patch.object(ConfigManager, "__init__", return_value=None)
    def test_load_template_success(self, mock_config_manager: Mock) -> None:
        """Test successful template loading."""
        template_path = os.path.join(self.templates_dir, "test_template.ini")
        with open(template_path, "w") as f:
            f.write("[general_settings]\nholding_time = 10\n")

        registry = ConfigRegistry(self.templates_dir, self.schemas_dir)
        result = registry.load_template("test_template")

        assert isinstance(result, ConfigManager)
        mock_config_manager.assert_called_once_with(template_path, self.schemas_dir)

    def test_load_template_not_found(self) -> None:
        """Test loading non-existent template."""
        registry = ConfigRegistry(self.templates_dir, self.schemas_dir)

        with pytest.raises(ConfigError) as exc_info:
            registry.load_template("nonexistent")

        assert "Template 'nonexistent' not found" in str(exc_info.value)

    @patch.object(ConfigRegistry, "load_template")
    def test_create_custom_config_default_template(
        self, mock_load_template: Mock
    ) -> None:
        """Test creating custom config with default template."""
        mock_config_manager = Mock()
        mock_load_template.return_value = mock_config_manager

        registry = ConfigRegistry(self.templates_dir, self.schemas_dir)
        result = registry.create_custom_config()

        mock_load_template.assert_called_once_with("default")
        assert result is mock_config_manager

    @patch.object(ConfigRegistry, "load_template")
    def test_create_custom_config_with_overrides(
        self, mock_load_template: Mock
    ) -> None:
        """Test creating custom config with overrides."""
        mock_config_manager = Mock()
        mock_load_template.return_value = mock_config_manager

        overrides = {
            "general_settings.max_iters": 5,
            "holding_time": 15,  # Unqualified key
        }

        registry = ConfigRegistry(self.templates_dir, self.schemas_dir)
        registry.create_custom_config("test_template", overrides)

        mock_load_template.assert_called_once_with("test_template")
        mock_config_manager.update_config.assert_any_call(
            "general_settings", "max_iters", 5
        )
        mock_config_manager.update_config.assert_any_call(
            "general_settings", "holding_time", 15
        )

    def test_validate_config_success(self) -> None:
        """Test successful configuration validation."""
        config_path = os.path.join(self.temp_dir, "valid_config.ini")
        with open(config_path, "w") as f:
            f.write("[general_settings]\nholding_time = 10\n")

        with patch.object(ConfigManager, "__init__", return_value=None):
            registry = ConfigRegistry(self.templates_dir, self.schemas_dir)
            errors = registry.validate_config(config_path)

        assert errors == []

    def test_validate_config_failure(self) -> None:
        """Test configuration validation failure."""
        config_path = os.path.join(self.temp_dir, "invalid_config.ini")
        with open(config_path, "w") as f:
            f.write("invalid ini content")

        registry = ConfigRegistry(self.templates_dir, self.schemas_dir)
        errors = registry.validate_config(config_path)

        assert len(errors) > 0
        assert isinstance(errors[0], str)

    def test_get_config_profiles(self) -> None:
        """Test getting predefined configuration profiles."""
        registry = ConfigRegistry(self.templates_dir, self.schemas_dir)
        profiles = registry.get_config_profiles()

        assert isinstance(profiles, dict)
        assert "quick_test" in profiles
        assert "development" in profiles
        assert "production" in profiles
        assert "rl_experiment" in profiles
        assert "benchmark" in profiles

        # Test profile structure
        quick_test = profiles["quick_test"]
        assert "description" in quick_test
        assert "template" in quick_test
        assert "overrides" in quick_test

    @patch.object(ConfigRegistry, "create_custom_config")
    def test_create_profile_config_success(self, mock_create_custom: Mock) -> None:
        """Test creating config from profile successfully."""
        mock_config_manager = Mock()
        mock_create_custom.return_value = mock_config_manager

        registry = ConfigRegistry(self.templates_dir, self.schemas_dir)
        result = registry.create_profile_config("quick_test")

        assert result is mock_config_manager
        mock_create_custom.assert_called_once()

    def test_create_profile_config_not_found(self) -> None:
        """Test creating config from non-existent profile."""
        registry = ConfigRegistry(self.templates_dir, self.schemas_dir)

        with pytest.raises(ConfigError) as exc_info:
            registry.create_profile_config("nonexistent_profile")

        assert "Profile 'nonexistent_profile' not found" in str(exc_info.value)

    @patch.object(ConfigRegistry, "create_custom_config")
    def test_create_profile_config_with_additional_overrides(
        self, mock_create_custom: Mock
    ) -> None:
        """Test creating profile config with additional overrides."""
        mock_config_manager = Mock()
        mock_create_custom.return_value = mock_config_manager

        additional_overrides = {"num_requests": 200}

        registry = ConfigRegistry(self.templates_dir, self.schemas_dir)
        registry.create_profile_config("development", additional_overrides)

        # Verify that additional overrides are merged
        call_args = mock_create_custom.call_args
        merged_overrides = call_args[0][1]  # Second argument is overrides
        assert "num_requests" in merged_overrides

    def test_export_config_template(self) -> None:
        """Test exporting configuration as template."""
        mock_config_manager = Mock()

        registry = ConfigRegistry(self.templates_dir, self.schemas_dir)
        result_path = registry.export_config_template(
            mock_config_manager, "exported_template", "Test description"
        )

        expected_path = os.path.join(self.templates_dir, "exported_template.ini")
        assert result_path == expected_path
        mock_config_manager.save_config.assert_called_once_with(expected_path, "ini")

    def test_export_config_template_refreshes_templates(self) -> None:
        """Test that exporting a template refreshes the template list."""
        mock_config_manager = Mock()

        registry = ConfigRegistry(self.templates_dir, self.schemas_dir)
        initial_count = len(registry._templates)

        # Create a real file so _load_templates can find it
        template_path = os.path.join(self.templates_dir, "exported_template.ini")
        with open(template_path, "w") as f:
            f.write("[general_settings]\nholding_time = 10\n")

        registry.export_config_template(mock_config_manager, "exported_template")

        # Templates should be refreshed and include the new template
        assert len(registry._templates) > initial_count

    def test_profile_structure_consistency(self) -> None:
        """Test that all profiles have consistent structure."""
        registry = ConfigRegistry(self.templates_dir, self.schemas_dir)
        profiles = registry.get_config_profiles()

        required_keys = {"description", "template", "overrides"}

        for profile_name, profile in profiles.items():
            assert isinstance(profile, dict), f"Profile '{profile_name}' is not a dict"

            for key in required_keys:
                assert key in profile, f"Profile '{profile_name}' missing key '{key}'"

            assert isinstance(profile["description"], str)
            assert isinstance(profile["template"], str)
            assert isinstance(profile["overrides"], dict)

    def test_template_name_validation(self) -> None:
        """Test template name handling and validation."""
        # Create templates with various valid names
        valid_names = ["default", "test_template", "template-1", "TEMPLATE"]

        for name in valid_names:
            template_path = os.path.join(self.templates_dir, f"{name}.ini")
            with open(template_path, "w") as f:
                f.write("[general_settings]\nholding_time = 10\n")

        registry = ConfigRegistry(self.templates_dir, self.schemas_dir)
        templates = registry.list_templates()

        for name in valid_names:
            assert name in templates
