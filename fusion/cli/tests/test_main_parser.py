"""Unit tests for main_parser module."""

from argparse import ArgumentParser, Namespace
from unittest.mock import Mock, patch

import pytest

from fusion.cli.main_parser import (
    AGENT_TYPE_CHOICES,
    TRAINING_GROUP_NAMES,
    GUINotSupportedError,
    build_main_argument_parser,
    build_parser,
    create_gui_argument_parser,
    create_training_argument_parser,
    get_gui_args,
    get_train_args,
)


class TestGroupConstants:
    """Tests for argument group constants."""

    def test_training_group_names_contains_expected_groups(self) -> None:
        """Test that TRAINING_GROUP_NAMES contains all expected groups."""
        expected_groups = [
            "config",
            "debug",
            "simulation",
            "network",
            "traffic",
            "training",
            "statistics",
        ]
        assert all(group in TRAINING_GROUP_NAMES for group in expected_groups)

    def test_training_group_names_is_list(self) -> None:
        """Test that TRAINING_GROUP_NAMES is a list."""
        assert isinstance(TRAINING_GROUP_NAMES, list)

    def test_agent_type_choices_contains_valid_options(self) -> None:
        """Test that AGENT_TYPE_CHOICES contains valid agent types."""
        assert "rl" in AGENT_TYPE_CHOICES
        assert "sl" in AGENT_TYPE_CHOICES

    def test_agent_type_choices_is_list(self) -> None:
        """Test that AGENT_TYPE_CHOICES is a list."""
        assert isinstance(AGENT_TYPE_CHOICES, list)


class TestBuildMainArgumentParser:
    """Tests for build_main_argument_parser function."""

    def test_build_main_argument_parser_returns_parser(self) -> None:
        """Test that build_main_argument_parser returns an ArgumentParser."""
        parser = build_main_argument_parser()

        assert isinstance(parser, ArgumentParser)

    @patch("fusion.cli.main_parser.args_registry.create_main_parser")
    def test_build_main_argument_parser_delegates_to_registry(self, mock_create_parser: Mock) -> None:
        """Test that build_main_argument_parser delegates to args_registry."""
        mock_parser = Mock(spec=ArgumentParser)
        mock_create_parser.return_value = mock_parser

        result = build_main_argument_parser()

        assert result == mock_parser
        mock_create_parser.assert_called_once()


class TestCreateTrainingArgumentParser:
    """Tests for create_training_argument_parser function."""

    @patch("fusion.cli.main_parser.args_registry.create_parser_with_groups")
    @patch("sys.argv", ["prog", "--agent_type", "rl"])
    def test_create_training_argument_parser_with_valid_args(self, mock_create_parser: Mock) -> None:
        """Test create_training_argument_parser with valid arguments."""
        mock_parser = Mock()
        mock_namespace = Namespace(agent_type="rl")
        mock_parser.parse_args.return_value = mock_namespace
        mock_create_parser.return_value = mock_parser

        result = create_training_argument_parser()

        assert result == mock_namespace
        mock_create_parser.assert_called_once_with(
            "Train an agent using reinforcement learning (RL) or supervised learning (SL)",
            TRAINING_GROUP_NAMES,
        )
        mock_parser.add_argument.assert_called_once()

    @patch("fusion.cli.main_parser.args_registry.create_parser_with_groups")
    @patch("sys.argv", ["prog", "--agent_type", "sl"])
    def test_create_training_argument_parser_with_sl_agent(self, mock_create_parser: Mock) -> None:
        """Test create_training_argument_parser with SL agent type."""
        mock_parser = Mock()
        mock_namespace = Namespace(agent_type="sl")
        mock_parser.parse_args.return_value = mock_namespace
        mock_create_parser.return_value = mock_parser

        result = create_training_argument_parser()

        assert result.agent_type == "sl"

    @patch("fusion.cli.main_parser.args_registry.create_parser_with_groups")
    def test_create_training_argument_parser_adds_agent_type_argument(self, mock_create_parser: Mock) -> None:
        """Test that create_training_argument_parser adds agent_type argument."""
        mock_parser = Mock()
        mock_parser.parse_args.return_value = Namespace(agent_type="rl")
        mock_create_parser.return_value = mock_parser

        with patch("sys.argv", ["prog", "--agent_type", "rl"]):
            create_training_argument_parser()

        # Verify add_argument was called with correct parameters
        mock_parser.add_argument.assert_called_once_with(
            "--agent_type",
            choices=AGENT_TYPE_CHOICES,
            required=True,
            help="Type of agent to train (rl=reinforcement learning, sl=supervised learning)",
        )

    @patch("fusion.cli.main_parser.args_registry.create_parser_with_groups")
    @patch("sys.argv", ["prog"])
    def test_create_training_argument_parser_exits_on_missing_required_arg(self, mock_create_parser: Mock) -> None:
        """Test that parser exits when required agent_type is missing."""
        mock_parser = Mock()
        mock_parser.parse_args.side_effect = SystemExit(2)
        mock_create_parser.return_value = mock_parser

        with pytest.raises(SystemExit):
            create_training_argument_parser()


class TestCreateGuiArgumentParser:
    """Tests for create_gui_argument_parser function - GUI not supported."""

    def test_create_gui_argument_parser_raises_gui_not_supported_error(self) -> None:
        """Test create_gui_argument_parser raises GUINotSupportedError."""
        with pytest.raises(GUINotSupportedError) as exc_info:
            create_gui_argument_parser()

        assert "not supported" in str(exc_info.value).lower()
        assert "6.1.0" in str(exc_info.value)

    def test_gui_not_supported_error_provides_helpful_message(self) -> None:
        """Test that GUINotSupportedError provides helpful guidance."""
        with pytest.raises(GUINotSupportedError) as exc_info:
            create_gui_argument_parser()

        error_message = str(exc_info.value)
        assert "CLI" in error_message or "cli" in error_message


class TestLegacyFunctions:
    """Tests for legacy wrapper functions."""

    def test_build_parser_delegates_to_build_main_argument_parser(self) -> None:
        """Test that build_parser is legacy wrapper for build_main_argument_parser."""
        with patch("fusion.cli.main_parser.build_main_argument_parser") as mock_build:
            mock_parser = Mock()
            mock_build.return_value = mock_parser

            result = build_parser()

            assert result == mock_parser
            mock_build.assert_called_once()

    def test_get_train_args_delegates_to_create_training_argument_parser(self) -> None:
        """Test that get_train_args is legacy wrapper."""
        with patch("fusion.cli.main_parser.create_training_argument_parser") as mock_create:
            mock_namespace = Mock()
            mock_create.return_value = mock_namespace

            result = get_train_args()

            assert result == mock_namespace
            mock_create.assert_called_once()

    def test_get_gui_args_raises_gui_not_supported_error(self) -> None:
        """Test that get_gui_args raises GUINotSupportedError."""
        with pytest.raises(GUINotSupportedError) as exc_info:
            get_gui_args()

        assert "not supported" in str(exc_info.value).lower()
