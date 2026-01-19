"""Unit tests for main_parser module."""

from argparse import ArgumentParser, Namespace
from unittest.mock import Mock, patch

import pytest

from fusion.cli.main_parser import (
    AGENT_TYPE_CHOICES,
    GUI_GROUP_NAMES,
    TRAINING_GROUP_NAMES,
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

    def test_gui_group_names_contains_expected_groups(self) -> None:
        """Test that GUI_GROUP_NAMES contains expected groups."""
        expected_groups = ["gui", "debug", "output"]
        assert all(group in GUI_GROUP_NAMES for group in expected_groups)

    def test_gui_group_names_is_list(self) -> None:
        """Test that GUI_GROUP_NAMES is a list."""
        assert isinstance(GUI_GROUP_NAMES, list)

    def test_agent_type_choices_contains_valid_options(self) -> None:
        """Test that AGENT_TYPE_CHOICES contains valid agent types."""
        assert "rl" in AGENT_TYPE_CHOICES
        assert "ml" in AGENT_TYPE_CHOICES

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
    def test_build_main_argument_parser_delegates_to_registry(
        self, mock_create_parser: Mock
    ) -> None:
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
    def test_create_training_argument_parser_with_valid_args(
        self, mock_create_parser: Mock
    ) -> None:
        """Test create_training_argument_parser with valid arguments."""
        mock_parser = Mock()
        mock_namespace = Namespace(agent_type="rl")
        mock_parser.parse_args.return_value = mock_namespace
        mock_create_parser.return_value = mock_parser

        result = create_training_argument_parser()

        assert result == mock_namespace
        mock_create_parser.assert_called_once_with(
            "Train an agent (RL or ML)", TRAINING_GROUP_NAMES
        )
        mock_parser.add_argument.assert_called_once()

    @patch("fusion.cli.main_parser.args_registry.create_parser_with_groups")
    @patch("sys.argv", ["prog", "--agent_type", "ml"])
    def test_create_training_argument_parser_with_ml_agent(
        self, mock_create_parser: Mock
    ) -> None:
        """Test create_training_argument_parser with ML agent type."""
        mock_parser = Mock()
        mock_namespace = Namespace(agent_type="ml")
        mock_parser.parse_args.return_value = mock_namespace
        mock_create_parser.return_value = mock_parser

        result = create_training_argument_parser()

        assert result.agent_type == "ml"

    @patch("fusion.cli.main_parser.args_registry.create_parser_with_groups")
    def test_create_training_argument_parser_adds_agent_type_argument(
        self, mock_create_parser: Mock
    ) -> None:
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
            help=(
                "Type of agent to train "
                "(rl=reinforcement learning, ml=machine learning)"
            ),
        )

    @patch("fusion.cli.main_parser.args_registry.create_parser_with_groups")
    @patch("sys.argv", ["prog"])
    def test_create_training_argument_parser_exits_on_missing_required_arg(
        self, mock_create_parser: Mock
    ) -> None:
        """Test that parser exits when required agent_type is missing."""
        mock_parser = Mock()
        mock_parser.parse_args.side_effect = SystemExit(2)
        mock_create_parser.return_value = mock_parser

        with pytest.raises(SystemExit):
            create_training_argument_parser()


class TestCreateGuiArgumentParser:
    """Tests for create_gui_argument_parser function."""

    @patch("fusion.cli.main_parser.args_registry.create_parser_with_groups")
    @patch("sys.argv", ["prog"])
    def test_create_gui_argument_parser_returns_namespace(
        self, mock_create_parser: Mock
    ) -> None:
        """Test create_gui_argument_parser returns parsed namespace."""
        mock_parser = Mock()
        mock_namespace = Namespace()
        mock_parser.parse_args.return_value = mock_namespace
        mock_create_parser.return_value = mock_parser

        result = create_gui_argument_parser()

        assert result == mock_namespace
        mock_create_parser.assert_called_once_with(
            "Launch GUI for FUSION", GUI_GROUP_NAMES
        )

    @patch("fusion.cli.main_parser.args_registry.create_parser_with_groups")
    def test_create_gui_argument_parser_uses_correct_description(
        self, mock_create_parser: Mock
    ) -> None:
        """Test that GUI parser uses correct description."""
        mock_parser = Mock()
        mock_parser.parse_args.return_value = Namespace()
        mock_create_parser.return_value = mock_parser

        with patch("sys.argv", ["prog"]):
            create_gui_argument_parser()

        mock_create_parser.assert_called_once_with(
            "Launch GUI for FUSION", GUI_GROUP_NAMES
        )

    @patch("fusion.cli.main_parser.args_registry.create_parser_with_groups")
    def test_create_gui_argument_parser_uses_gui_groups(
        self, mock_create_parser: Mock
    ) -> None:
        """Test that GUI parser uses GUI_GROUP_NAMES."""
        mock_parser = Mock()
        mock_parser.parse_args.return_value = Namespace()
        mock_create_parser.return_value = mock_parser

        with patch("sys.argv", ["prog"]):
            create_gui_argument_parser()

        args, kwargs = mock_create_parser.call_args
        assert args[1] == GUI_GROUP_NAMES


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
        with patch(
            "fusion.cli.main_parser.create_training_argument_parser"
        ) as mock_create:
            mock_namespace = Mock()
            mock_create.return_value = mock_namespace

            result = get_train_args()

            assert result == mock_namespace
            mock_create.assert_called_once()

    def test_get_gui_args_delegates_to_create_gui_argument_parser(self) -> None:
        """Test that get_gui_args is legacy wrapper."""
        with patch("fusion.cli.main_parser.create_gui_argument_parser") as mock_create:
            mock_namespace = Mock()
            mock_create.return_value = mock_namespace

            result = get_gui_args()

            assert result == mock_namespace
            mock_create.assert_called_once()

    def test_legacy_functions_return_same_types_as_new_functions(self) -> None:
        """Test that legacy functions return same types as new functions."""
        # Mock all the internal functions to prevent actual argument parsing
        with patch("fusion.cli.main_parser.build_main_argument_parser") as mock_build:
            with patch(
                "fusion.cli.main_parser.create_training_argument_parser"
            ) as mock_train:
                with patch(
                    "fusion.cli.main_parser.create_gui_argument_parser"
                ) as mock_gui:
                    # Set up return values
                    mock_parser = Mock(spec=ArgumentParser)
                    mock_namespace = Mock(spec=Namespace)

                    mock_build.return_value = mock_parser
                    mock_train.return_value = mock_namespace
                    mock_gui.return_value = mock_namespace

                    # Test that legacy functions delegate to new functions
                    result1 = build_parser()
                    result2 = get_train_args()
                    result3 = get_gui_args()

                    # Verify the calls were made
                    mock_build.assert_called()
                    mock_train.assert_called()
                    mock_gui.assert_called()

                    # Verify return types match
                    assert type(result1) is type(mock_parser)
                    assert type(result2) is type(mock_namespace)
                    assert type(result3) is type(mock_namespace)
