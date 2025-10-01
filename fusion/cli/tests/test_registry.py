"""Unit tests for parameters registry module."""

import argparse
from unittest.mock import Mock

import pytest

from fusion.cli.parameters.registry import ArgumentRegistry, args_registry


class TestArgumentRegistry:
    """Tests for ArgumentRegistry class."""

    def test_argument_registry_initialization(self) -> None:
        """Test ArgumentRegistry initializes with empty registries."""
        registry = ArgumentRegistry()
        assert hasattr(registry, "_argument_groups")
        assert hasattr(registry, "_subcommand_parsers")

    def test_argument_registry_has_default_groups_after_init(self) -> None:
        """Test ArgumentRegistry has default groups after initialization."""
        registry = ArgumentRegistry()
        # Should have some default groups registered
        assert registry.get_group_count() > 0
        assert registry.has_group("config")
        assert registry.has_group("debug")

    def test_register_group_adds_group_successfully(self) -> None:
        """Test register_group adds new group to registry."""
        registry = ArgumentRegistry()
        mock_func = Mock()

        registry.register_group("test_group", mock_func)

        assert registry.has_group("test_group")
        assert registry._argument_groups["test_group"] == mock_func

    def test_register_group_with_duplicate_name_raises_error(self) -> None:
        """Test register_group raises ValueError for duplicate names."""
        registry = ArgumentRegistry()
        mock_func = Mock()
        registry.register_group("duplicate", mock_func)

        with pytest.raises(ValueError) as exc_info:
            registry.register_group("duplicate", mock_func)
        assert "already registered" in str(exc_info.value)

    def test_register_subcommand_adds_subcommand_successfully(self) -> None:
        """Test register_subcommand adds new subcommand to registry."""
        registry = ArgumentRegistry()
        mock_func = Mock()

        registry.register_subcommand("test_cmd", mock_func)

        assert registry.has_subcommand("test_cmd")
        assert registry._subcommand_parsers["test_cmd"] == mock_func

    def test_register_subcommand_with_duplicate_name_raises_error(self) -> None:
        """Test register_subcommand raises ValueError for duplicate names."""
        registry = ArgumentRegistry()
        mock_func = Mock()
        registry.register_subcommand("duplicate", mock_func)

        with pytest.raises(ValueError) as exc_info:
            registry.register_subcommand("duplicate", mock_func)
        assert "already registered" in str(exc_info.value)

    def test_add_groups_to_parser_with_valid_groups(self) -> None:
        """Test add_groups_to_parser adds all requested groups."""
        registry = ArgumentRegistry()
        mock_parser = Mock(spec=argparse.ArgumentParser)
        mock_func1 = Mock()
        mock_func2 = Mock()
        registry.register_group("group1", mock_func1)
        registry.register_group("group2", mock_func2)

        registry.add_groups_to_parser(mock_parser, ["group1", "group2"])

        mock_func1.assert_called_once_with(mock_parser)
        mock_func2.assert_called_once_with(mock_parser)

    def test_add_groups_to_parser_with_unknown_group_raises_error(self) -> None:
        """Test add_groups_to_parser raises ValueError for unknown group."""
        registry = ArgumentRegistry()
        mock_parser = Mock(spec=argparse.ArgumentParser)

        with pytest.raises(ValueError) as exc_info:
            registry.add_groups_to_parser(mock_parser, ["unknown_group"])
        assert "Unknown argument group" in str(exc_info.value)

    def test_add_groups_to_parser_error_includes_available_groups(self) -> None:
        """Test error message includes list of available groups."""
        registry = ArgumentRegistry()
        mock_parser = Mock(spec=argparse.ArgumentParser)
        registry.register_group("available1", Mock())
        registry.register_group("available2", Mock())

        with pytest.raises(ValueError) as exc_info:
            registry.add_groups_to_parser(mock_parser, ["unknown_group"])
        error_message = str(exc_info.value)
        assert "Available groups:" in error_message
        assert "available1" in error_message
        assert "available2" in error_message

    def test_create_parser_with_groups_returns_configured_parser(self) -> None:
        """Test create_parser_with_groups returns parser with groups."""
        registry = ArgumentRegistry()
        mock_func = Mock()
        registry.register_group("test_group", mock_func)

        parser = registry.create_parser_with_groups("Test parser", ["test_group"])

        assert isinstance(parser, argparse.ArgumentParser)
        mock_func.assert_called_once()

    def test_create_parser_with_groups_uses_description(self) -> None:
        """Test create_parser_with_groups uses provided description."""
        registry = ArgumentRegistry()
        registry.register_group("test_group", Mock())

        parser = registry.create_parser_with_groups(
            "Custom description", ["test_group"]
        )

        assert parser.description == "Custom description"

    def test_create_main_parser_returns_parser_with_subparsers(self) -> None:
        """Test create_main_parser returns parser with subcommands."""
        registry = ArgumentRegistry()
        mock_register = Mock()
        registry.register_subcommand("test_cmd", mock_register)

        parser = registry.create_main_parser()

        assert isinstance(parser, argparse.ArgumentParser)
        mock_register.assert_called_once()

    def test_create_main_parser_has_correct_description(self) -> None:
        """Test create_main_parser has correct description."""
        registry = ArgumentRegistry()

        parser = registry.create_main_parser()

        assert parser.description is not None
        assert "FUSION Simulator CLI" in parser.description

    def test_get_available_groups_returns_sorted_list(self) -> None:
        """Test get_available_groups returns sorted group names."""
        registry = ArgumentRegistry()
        registry.register_group("zebra", Mock())
        registry.register_group("alpha", Mock())

        groups = registry.get_available_groups()

        assert "alpha" in groups
        assert "zebra" in groups
        # Check if sorted (at least alpha comes before zebra)
        alpha_index = groups.index("alpha")
        zebra_index = groups.index("zebra")
        assert alpha_index < zebra_index

    def test_get_available_subcommands_returns_sorted_list(self) -> None:
        """Test get_available_subcommands returns sorted command names."""
        registry = ArgumentRegistry()
        registry.register_subcommand("zebra", Mock())
        registry.register_subcommand("alpha", Mock())

        commands = registry.get_available_subcommands()

        assert "alpha" in commands
        assert "zebra" in commands
        # Check if sorted
        alpha_index = commands.index("alpha")
        zebra_index = commands.index("zebra")
        assert alpha_index < zebra_index

    def test_get_group_count_returns_correct_count(self) -> None:
        """Test get_group_count returns correct number of groups."""
        registry = ArgumentRegistry()
        initial_count = registry.get_group_count()

        registry.register_group("group1", Mock())
        registry.register_group("group2", Mock())

        assert registry.get_group_count() == initial_count + 2

    def test_has_group_returns_correct_boolean(self) -> None:
        """Test has_group returns True for existing groups, False otherwise."""
        registry = ArgumentRegistry()
        registry.register_group("exists", Mock())

        assert registry.has_group("exists") is True
        assert registry.has_group("not_exists") is False

    def test_has_subcommand_returns_correct_boolean(self) -> None:
        """Test has_subcommand returns True for existing commands, False otherwise."""
        registry = ArgumentRegistry()
        registry.register_subcommand("exists", Mock())

        assert registry.has_subcommand("exists") is True
        assert registry.has_subcommand("not_exists") is False

    def test_registry_registers_core_groups(self) -> None:
        """Test that registry registers expected core groups."""
        registry = ArgumentRegistry()

        core_groups = ["config", "debug", "output"]
        for group in core_groups:
            assert registry.has_group(group), f"Core group '{group}' not registered"

    def test_registry_registers_compatibility_groups(self) -> None:
        """Test that registry registers expected compatibility groups."""
        registry = ArgumentRegistry()

        compatibility_groups = ["simulation", "network", "traffic"]
        for group in compatibility_groups:
            assert registry.has_group(group), (
                f"Compatibility group '{group}' not registered"
            )

    def test_registry_registers_training_groups(self) -> None:
        """Test that registry registers expected training groups."""
        registry = ArgumentRegistry()

        training_groups = ["training", "rl", "ml"]
        for group in training_groups:
            assert registry.has_group(group), f"Training group '{group}' not registered"

    def test_registry_registers_analysis_groups(self) -> None:
        """Test that registry registers expected analysis groups."""
        registry = ArgumentRegistry()

        analysis_groups = ["analysis", "statistics", "plotting"]
        for group in analysis_groups:
            assert registry.has_group(group), f"Analysis group '{group}' not registered"

    def test_registry_registers_interface_groups(self) -> None:
        """Test that registry registers expected interface groups."""
        registry = ArgumentRegistry()

        assert registry.has_group("gui"), "GUI group not registered"

    def test_registry_registers_subcommands(self) -> None:
        """Test that registry registers expected subcommands."""
        registry = ArgumentRegistry()

        assert registry.has_subcommand("run_sim"), "run_sim subcommand not registered"


class TestArgsRegistrySingleton:
    """Tests for the args_registry singleton instance."""

    def test_args_registry_is_singleton_instance(self) -> None:
        """Test that args_registry is a singleton ArgumentRegistry instance."""
        assert isinstance(args_registry, ArgumentRegistry)

    def test_args_registry_has_default_groups(self) -> None:
        """Test that args_registry has some default groups registered."""
        assert args_registry.get_group_count() > 0

    def test_args_registry_has_core_functionality(self) -> None:
        """Test that args_registry has core functionality working."""
        # Should have some basic groups
        assert args_registry.has_group("config")
        assert args_registry.has_group("debug")

        # Should be able to create parsers
        parser = args_registry.create_parser_with_groups("Test", ["config"])
        assert isinstance(parser, argparse.ArgumentParser)

    def test_args_registry_create_main_parser_works(self) -> None:
        """Test that args_registry can create main parser."""
        parser = args_registry.create_main_parser()
        assert isinstance(parser, argparse.ArgumentParser)
        assert parser.description is not None
        assert "FUSION" in parser.description
