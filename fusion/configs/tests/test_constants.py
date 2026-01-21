"""Unit tests for fusion.configs.constants module."""

import os
import re

import pytest

from fusion.configs.constants import (
    CONFIG_DIR_PATH,
    DEFAULT_CONFIG_PATH,
    DEFAULT_THREAD_NAME,
    DICT_PARAM_OPTIONS,
    PROJECT_ROOT,
    REQUIRED_SECTION,
    THREAD_SECTION_PATTERN,
)


class TestConstants:
    """Tests for configuration constants."""

    def test_project_root_is_string(self) -> None:
        """Test that PROJECT_ROOT is a string."""
        assert isinstance(PROJECT_ROOT, str)
        assert len(PROJECT_ROOT) > 0

    def test_project_root_is_absolute_path(self) -> None:
        """Test that PROJECT_ROOT is an absolute path."""
        assert os.path.isabs(PROJECT_ROOT)

    def test_default_config_path_is_string(self) -> None:
        """Test that DEFAULT_CONFIG_PATH is a string."""
        assert isinstance(DEFAULT_CONFIG_PATH, str)
        assert len(DEFAULT_CONFIG_PATH) > 0

    def test_default_config_path_is_absolute(self) -> None:
        """Test that DEFAULT_CONFIG_PATH is an absolute path."""
        assert os.path.isabs(DEFAULT_CONFIG_PATH)

    def test_default_config_path_ends_with_ini(self) -> None:
        """Test that DEFAULT_CONFIG_PATH ends with .ini extension."""
        assert DEFAULT_CONFIG_PATH.endswith(".ini")

    def test_default_config_path_contains_project_root(self) -> None:
        """Test that DEFAULT_CONFIG_PATH is based on PROJECT_ROOT."""
        assert DEFAULT_CONFIG_PATH.startswith(PROJECT_ROOT)

    def test_config_dir_path_is_string(self) -> None:
        """Test that CONFIG_DIR_PATH is a string."""
        assert isinstance(CONFIG_DIR_PATH, str)
        assert len(CONFIG_DIR_PATH) > 0

    def test_config_dir_path_value(self) -> None:
        """Test that CONFIG_DIR_PATH has expected value."""
        expected_path = os.path.join("ini", "run_ini")
        assert CONFIG_DIR_PATH == expected_path

    def test_required_section_is_string(self) -> None:
        """Test that REQUIRED_SECTION is a string."""
        assert isinstance(REQUIRED_SECTION, str)
        assert len(REQUIRED_SECTION) > 0

    def test_required_section_value(self) -> None:
        """Test that REQUIRED_SECTION has expected value."""
        assert REQUIRED_SECTION == "general_settings"

    def test_default_thread_name_is_string(self) -> None:
        """Test that DEFAULT_THREAD_NAME is a string."""
        assert isinstance(DEFAULT_THREAD_NAME, str)
        assert len(DEFAULT_THREAD_NAME) > 0

    def test_default_thread_name_value(self) -> None:
        """Test that DEFAULT_THREAD_NAME has expected value."""
        assert DEFAULT_THREAD_NAME == "s1"

    def test_thread_section_pattern_is_string(self) -> None:
        """Test that THREAD_SECTION_PATTERN is a string."""
        assert isinstance(THREAD_SECTION_PATTERN, str)
        assert len(THREAD_SECTION_PATTERN) > 0

    def test_thread_section_pattern_value(self) -> None:
        """Test that THREAD_SECTION_PATTERN has expected value."""
        assert THREAD_SECTION_PATTERN == r"^s\d"

    def test_thread_section_pattern_is_valid_regex(self) -> None:
        """Test that THREAD_SECTION_PATTERN is a valid regex."""
        try:
            compiled_pattern = re.compile(THREAD_SECTION_PATTERN)
            assert compiled_pattern is not None
        except re.error:
            pytest.fail("THREAD_SECTION_PATTERN is not a valid regex")

    def test_thread_section_pattern_matches_expected_strings(self) -> None:
        """Test that THREAD_SECTION_PATTERN matches expected thread section names."""
        pattern = re.compile(THREAD_SECTION_PATTERN)

        # Should match
        valid_sections = ["s1", "s2", "s10", "s999"]
        for section in valid_sections:
            assert pattern.match(section), f"Pattern should match '{section}'"

    def test_thread_section_pattern_rejects_invalid_strings(self) -> None:
        """Test that THREAD_SECTION_PATTERN rejects invalid thread section names."""
        pattern = re.compile(THREAD_SECTION_PATTERN)

        # Should not match (using fullmatch for exact matching)
        invalid_sections = ["s", "section1", "t1", "1s", "general_settings"]
        for section in invalid_sections:
            assert not pattern.match(section), f"Pattern should not match '{section}'"

        # Note: 's1a' would match with ^s\d pattern as it matches the start 's1'
        # This is the current expected behavior of the pattern

    def test_dict_param_options_is_list(self) -> None:
        """Test that DICT_PARAM_OPTIONS is a list."""
        assert isinstance(DICT_PARAM_OPTIONS, list)

    def test_dict_param_options_contains_expected_values(self) -> None:
        """Test that DICT_PARAM_OPTIONS contains expected parameter names."""
        expected_params = ["request_distribution", "requested_xt", "phi"]

        assert len(DICT_PARAM_OPTIONS) == len(expected_params)
        for param in expected_params:
            assert param in DICT_PARAM_OPTIONS

    def test_dict_param_options_all_strings(self) -> None:
        """Test that all items in DICT_PARAM_OPTIONS are strings."""
        for param in DICT_PARAM_OPTIONS:
            assert isinstance(param, str)
            assert len(param) > 0

    def test_dict_param_options_no_duplicates(self) -> None:
        """Test that DICT_PARAM_OPTIONS contains no duplicates."""
        assert len(DICT_PARAM_OPTIONS) == len(set(DICT_PARAM_OPTIONS))

    def test_constants_immutability(self) -> None:
        """Test that constants maintain their expected types."""
        # This test ensures constants haven't been accidentally modified
        assert isinstance(PROJECT_ROOT, str)
        assert isinstance(DEFAULT_CONFIG_PATH, str)
        assert isinstance(CONFIG_DIR_PATH, str)
        assert isinstance(REQUIRED_SECTION, str)
        assert isinstance(DEFAULT_THREAD_NAME, str)
        assert isinstance(THREAD_SECTION_PATTERN, str)
        assert isinstance(DICT_PARAM_OPTIONS, list)

    def test_path_relationships(self) -> None:
        """Test relationships between path constants."""
        # DEFAULT_CONFIG_PATH should be constructed from PROJECT_ROOT and
        # CONFIG_DIR_PATH
        expected_config_path = os.path.join(PROJECT_ROOT, CONFIG_DIR_PATH, "config.ini")
        assert DEFAULT_CONFIG_PATH == expected_config_path

    def test_project_root_structure(self) -> None:
        """Test that PROJECT_ROOT points to a reasonable directory structure."""
        # PROJECT_ROOT should be calculated relative to the configs module
        # Go up from tests to configs
        configs_dir = os.path.dirname(os.path.dirname(__file__))
        fusion_dir = os.path.dirname(configs_dir)  # Go up from configs to fusion
        expected_root = os.path.dirname(fusion_dir)  # Go up from fusion to project root

        assert os.path.normpath(PROJECT_ROOT) == os.path.normpath(expected_root)

    def test_constants_documentation(self) -> None:
        """Test that all constants are properly documented in the module."""
        import fusion.configs.constants as constants_module

        # Check that the module has the expected constants defined
        expected_constants = [
            "PROJECT_ROOT",
            "DEFAULT_CONFIG_PATH",
            "CONFIG_DIR_PATH",
            "REQUIRED_SECTION",
            "DEFAULT_THREAD_NAME",
            "THREAD_SECTION_PATTERN",
            "DICT_PARAM_OPTIONS",
        ]

        for constant in expected_constants:
            assert hasattr(constants_module, constant), f"Constant '{constant}' not found in module"
