"""Unit tests for run_gui module."""

from typing import Any
from unittest.mock import Mock, patch

from fusion.cli.constants import ERROR_EXIT_CODE, INTERRUPT_EXIT_CODE, SUCCESS_EXIT_CODE
from fusion.cli.run_gui import main as gui_main


class TestGuiMain:
    """Tests for run_gui main function."""

    @patch("fusion.cli.run_gui.launch_gui_pipeline")
    @patch("fusion.cli.run_gui.create_gui_argument_parser")
    def test_gui_main_success_returns_zero(
        self, mock_create_parser: Any, mock_launch_gui: Any
    ) -> None:
        """Test GUI main returns SUCCESS_EXIT_CODE on successful execution."""
        mock_args = Mock()
        mock_create_parser.return_value = mock_args

        result = gui_main()

        assert result == SUCCESS_EXIT_CODE
        mock_create_parser.assert_called_once()
        mock_launch_gui.assert_called_once_with(mock_args)

    @patch("fusion.cli.run_gui.launch_gui_pipeline")
    @patch("fusion.cli.run_gui.create_gui_argument_parser")
    def test_gui_main_handles_keyboard_interrupt(
        self, mock_create_parser: Any, mock_launch_gui: Any
    ) -> None:
        """Test GUI main handles KeyboardInterrupt gracefully."""
        mock_create_parser.return_value = Mock()
        mock_launch_gui.side_effect = KeyboardInterrupt()

        result = gui_main()

        assert result == INTERRUPT_EXIT_CODE

    @patch("fusion.cli.run_gui.launch_gui_pipeline")
    @patch("fusion.cli.run_gui.create_gui_argument_parser")
    def test_gui_main_handles_import_error(
        self, mock_create_parser: Any, mock_launch_gui: Any
    ) -> None:
        """Test GUI main handles missing GUI dependencies."""
        mock_create_parser.return_value = Mock()
        mock_launch_gui.side_effect = ImportError("No module named 'tkinter'")

        result = gui_main()

        assert result == ERROR_EXIT_CODE

    @patch("fusion.cli.run_gui.launch_gui_pipeline")
    @patch("fusion.cli.run_gui.create_gui_argument_parser")
    def test_gui_main_handles_module_not_found_error(
        self, mock_create_parser: Any, mock_launch_gui: Any
    ) -> None:
        """Test GUI main handles ModuleNotFoundError for GUI dependencies."""
        mock_create_parser.return_value = Mock()
        mock_launch_gui.side_effect = ModuleNotFoundError(
            "No module named 'matplotlib'"
        )

        result = gui_main()

        assert result == ERROR_EXIT_CODE

    @patch("fusion.cli.run_gui.launch_gui_pipeline")
    @patch("fusion.cli.run_gui.create_gui_argument_parser")
    def test_gui_main_handles_runtime_error(
        self, mock_create_parser: Any, mock_launch_gui: Any
    ) -> None:
        """Test GUI main handles RuntimeError from display issues."""
        mock_create_parser.return_value = Mock()
        mock_launch_gui.side_effect = RuntimeError("No display found")

        result = gui_main()

        assert result == ERROR_EXIT_CODE

    @patch("fusion.cli.run_gui.launch_gui_pipeline")
    @patch("fusion.cli.run_gui.create_gui_argument_parser")
    def test_gui_main_handles_os_error(
        self, mock_create_parser: Any, mock_launch_gui: Any
    ) -> None:
        """Test GUI main handles OSError from file system issues."""
        mock_create_parser.return_value = Mock()
        mock_launch_gui.side_effect = OSError("Cannot create output directory")

        result = gui_main()

        assert result == ERROR_EXIT_CODE

    @patch("fusion.cli.run_gui.launch_gui_pipeline")
    @patch("fusion.cli.run_gui.create_gui_argument_parser")
    def test_gui_main_handles_value_error(
        self, mock_create_parser: Any, mock_launch_gui: Any
    ) -> None:
        """Test GUI main handles ValueError from configuration issues."""
        mock_create_parser.return_value = Mock()
        mock_launch_gui.side_effect = ValueError("Invalid window size")

        result = gui_main()

        assert result == ERROR_EXIT_CODE

    @patch("fusion.cli.run_gui.launch_gui_pipeline")
    @patch("fusion.cli.run_gui.create_gui_argument_parser")
    def test_gui_main_handles_type_error(
        self, mock_create_parser: Any, mock_launch_gui: Any
    ) -> None:
        """Test GUI main handles TypeError from configuration issues."""
        mock_create_parser.return_value = Mock()
        mock_launch_gui.side_effect = TypeError("Invalid type for GUI parameter")

        result = gui_main()

        assert result == ERROR_EXIT_CODE

    @patch("fusion.cli.run_gui.launch_gui_pipeline")
    @patch("fusion.cli.run_gui.create_gui_argument_parser")
    def test_gui_main_passes_arguments_to_launcher(
        self, mock_create_parser: Any, mock_launch_gui: Any
    ) -> None:
        """Test that GUI main passes parsed arguments to GUI launcher."""
        mock_args = Mock(width=800, height=600, debug=True)
        mock_create_parser.return_value = mock_args

        gui_main()

        mock_launch_gui.assert_called_once_with(mock_args)

    @patch("fusion.cli.run_gui.launch_gui_pipeline")
    @patch("fusion.cli.run_gui.create_gui_argument_parser")
    def test_gui_main_creates_parser_before_launching_gui(
        self, mock_create_parser: Any, mock_launch_gui: Any
    ) -> None:
        """Test that GUI main creates parser before launching GUI."""
        mock_args = Mock()
        mock_create_parser.return_value = mock_args

        gui_main()

        # Verify both functions were called
        assert mock_create_parser.called
        assert mock_launch_gui.called

    @patch("fusion.cli.run_gui.launch_gui_pipeline")
    @patch("fusion.cli.run_gui.create_gui_argument_parser")
    @patch("builtins.print")
    def test_gui_main_prints_user_friendly_messages_on_error(
        self, mock_print: Any, mock_create_parser: Any, mock_launch_gui: Any
    ) -> None:
        """Test that GUI main prints user-friendly error messages."""
        mock_create_parser.return_value = Mock()
        mock_launch_gui.side_effect = ImportError("Missing tkinter")

        gui_main()

        # Verify print was called with user-friendly messages
        assert mock_print.called
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        assert any("Missing GUI dependencies" in call for call in print_calls)
        assert any("pip install" in call for call in print_calls)

    @patch("fusion.cli.run_gui.launch_gui_pipeline")
    @patch("fusion.cli.run_gui.create_gui_argument_parser")
    @patch("builtins.print")
    def test_gui_main_prints_interrupt_message(
        self, mock_print: Any, mock_create_parser: Any, mock_launch_gui: Any
    ) -> None:
        """Test that GUI main prints appropriate message on KeyboardInterrupt."""
        mock_create_parser.return_value = Mock()
        mock_launch_gui.side_effect = KeyboardInterrupt()

        gui_main()

        print_calls = [call[0][0] for call in mock_print.call_args_list]
        assert any("interrupted" in call.lower() for call in print_calls)

    @patch("fusion.cli.run_gui.launch_gui_pipeline")
    @patch("fusion.cli.run_gui.create_gui_argument_parser")
    @patch("builtins.print")
    def test_gui_main_provides_helpful_suggestions_for_display_errors(
        self, mock_print: Any, mock_create_parser: Any, mock_launch_gui: Any
    ) -> None:
        """Test GUI main provides helpful suggestions for display-related errors."""
        mock_create_parser.return_value = Mock()
        mock_launch_gui.side_effect = RuntimeError("cannot connect to X server")

        gui_main()

        print_calls = [call[0][0] for call in mock_print.call_args_list]
        # Should contain helpful suggestions about display or framework
        assert any(
            "display" in call.lower() or "framework" in call.lower()
            for call in print_calls
        )

    @patch("fusion.cli.run_gui.launch_gui_pipeline")
    @patch("fusion.cli.run_gui.create_gui_argument_parser")
    @patch("builtins.print")
    def test_gui_main_provides_helpful_suggestions_for_import_errors(
        self, mock_print: Any, mock_create_parser: Any, mock_launch_gui: Any
    ) -> None:
        """Test GUI main provides helpful suggestions for import errors."""
        mock_create_parser.return_value = Mock()
        mock_launch_gui.side_effect = ImportError("No module named 'PyQt5'")

        gui_main()

        print_calls = [call[0][0] for call in mock_print.call_args_list]
        # Should contain installation suggestions
        assert any(
            "install" in call.lower() or "pip" in call.lower() for call in print_calls
        )
