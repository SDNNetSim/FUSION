"""Unit tests for run_train module."""

from unittest.mock import Mock, patch

from fusion.cli.constants import ERROR_EXIT_CODE, INTERRUPT_EXIT_CODE, SUCCESS_EXIT_CODE
from fusion.cli.run_train import main as train_main


class TestTrainMain:
    """Tests for run_train main function."""

    @patch("fusion.cli.run_train.run_training_pipeline")
    @patch("fusion.cli.run_train.create_training_argument_parser")
    def test_train_main_success_returns_zero(
        self, mock_create_parser: Mock, mock_run_pipeline: Mock
    ) -> None:
        """Test train main returns SUCCESS_EXIT_CODE on successful execution."""
        mock_args = Mock()
        mock_create_parser.return_value = mock_args

        result = train_main()

        assert result == SUCCESS_EXIT_CODE
        mock_create_parser.assert_called_once()
        mock_run_pipeline.assert_called_once_with(mock_args)

    @patch("fusion.cli.run_train.run_training_pipeline")
    @patch("fusion.cli.run_train.create_training_argument_parser")
    def test_train_main_handles_keyboard_interrupt(
        self, mock_create_parser: Mock, mock_run_pipeline: Mock
    ) -> None:
        """Test train main handles KeyboardInterrupt gracefully."""
        mock_create_parser.return_value = Mock()
        mock_run_pipeline.side_effect = KeyboardInterrupt()

        result = train_main()

        assert result == INTERRUPT_EXIT_CODE

    @patch("fusion.cli.run_train.run_training_pipeline")
    @patch("fusion.cli.run_train.create_training_argument_parser")
    def test_train_main_handles_import_error(
        self, mock_create_parser: Mock, mock_run_pipeline: Mock
    ) -> None:
        """Test train main handles ImportError with helpful message."""
        mock_create_parser.return_value = Mock()
        mock_run_pipeline.side_effect = ImportError("Missing module")

        result = train_main()

        assert result == ERROR_EXIT_CODE

    @patch("fusion.cli.run_train.run_training_pipeline")
    @patch("fusion.cli.run_train.create_training_argument_parser")
    def test_train_main_handles_module_not_found_error(
        self, mock_create_parser: Mock, mock_run_pipeline: Mock
    ) -> None:
        """Test train main handles ModuleNotFoundError."""
        mock_create_parser.return_value = Mock()
        mock_run_pipeline.side_effect = ModuleNotFoundError("No module named 'torch'")

        result = train_main()

        assert result == ERROR_EXIT_CODE

    @patch("fusion.cli.run_train.run_training_pipeline")
    @patch("fusion.cli.run_train.create_training_argument_parser")
    def test_train_main_handles_os_error(
        self, mock_create_parser: Mock, mock_run_pipeline: Mock
    ) -> None:
        """Test train main handles OSError with helpful message."""
        mock_create_parser.return_value = Mock()
        mock_run_pipeline.side_effect = OSError("Permission denied")

        result = train_main()

        assert result == ERROR_EXIT_CODE

    @patch("fusion.cli.run_train.run_training_pipeline")
    @patch("fusion.cli.run_train.create_training_argument_parser")
    def test_train_main_handles_value_error(
        self, mock_create_parser: Mock, mock_run_pipeline: Mock
    ) -> None:
        """Test train main handles ValueError from configuration issues."""
        mock_create_parser.return_value = Mock()
        mock_run_pipeline.side_effect = ValueError("Invalid config")

        result = train_main()

        assert result == ERROR_EXIT_CODE

    @patch("fusion.cli.run_train.run_training_pipeline")
    @patch("fusion.cli.run_train.create_training_argument_parser")
    def test_train_main_handles_type_error(
        self, mock_create_parser: Mock, mock_run_pipeline: Mock
    ) -> None:
        """Test train main handles TypeError from configuration issues."""
        mock_create_parser.return_value = Mock()
        mock_run_pipeline.side_effect = TypeError("Invalid type")

        result = train_main()

        assert result == ERROR_EXIT_CODE

    @patch("fusion.cli.run_train.run_training_pipeline")
    @patch("fusion.cli.run_train.create_training_argument_parser")
    def test_train_main_handles_runtime_error(
        self, mock_create_parser: Mock, mock_run_pipeline: Mock
    ) -> None:
        """Test train main handles RuntimeError with resource suggestions."""
        mock_create_parser.return_value = Mock()
        mock_run_pipeline.side_effect = RuntimeError("CUDA out of memory")

        result = train_main()

        assert result == ERROR_EXIT_CODE

    @patch("fusion.cli.run_train.run_training_pipeline")
    @patch("fusion.cli.run_train.create_training_argument_parser")
    def test_train_main_handles_memory_error(
        self, mock_create_parser: Mock, mock_run_pipeline: Mock
    ) -> None:
        """Test train main handles MemoryError with resource suggestions."""
        mock_create_parser.return_value = Mock()
        mock_run_pipeline.side_effect = MemoryError("Out of memory")

        result = train_main()

        assert result == ERROR_EXIT_CODE

    @patch("fusion.cli.run_train.run_training_pipeline")
    @patch("fusion.cli.run_train.create_training_argument_parser")
    def test_train_main_passes_arguments_to_pipeline(
        self, mock_create_parser: Mock, mock_run_pipeline: Mock
    ) -> None:
        """Test that train main passes parsed arguments to training pipeline."""
        mock_args = Mock(agent_type="rl", config_path="test.ini")
        mock_create_parser.return_value = mock_args

        train_main()

        mock_run_pipeline.assert_called_once_with(mock_args)

    @patch("fusion.cli.run_train.run_training_pipeline")
    @patch("fusion.cli.run_train.create_training_argument_parser")
    def test_train_main_creates_parser_before_running_pipeline(
        self, mock_create_parser: Mock, mock_run_pipeline: Mock
    ) -> None:
        """Test that train main creates parser before running pipeline."""
        mock_args = Mock()
        mock_create_parser.return_value = mock_args

        train_main()

        # Verify order of calls
        assert mock_create_parser.called
        assert mock_run_pipeline.called
        # Parser should be called first
        call_order = [
            call[0]
            for call in [mock_create_parser.call_args, mock_run_pipeline.call_args]
        ]
        assert call_order[0] is not None  # Parser was called

    @patch("fusion.cli.run_train.run_training_pipeline")
    @patch("fusion.cli.run_train.create_training_argument_parser")
    @patch("builtins.print")
    def test_train_main_prints_user_friendly_messages_on_error(
        self, mock_print: Mock, mock_create_parser: Mock, mock_run_pipeline: Mock
    ) -> None:
        """Test that train main prints user-friendly error messages."""
        mock_create_parser.return_value = Mock()
        mock_run_pipeline.side_effect = ImportError("Missing torch")

        train_main()

        # Verify print was called with user-friendly messages
        assert mock_print.called
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        assert any("❌" in call for call in print_calls)
        assert any("💡" in call for call in print_calls)

    @patch("fusion.cli.run_train.run_training_pipeline")
    @patch("fusion.cli.run_train.create_training_argument_parser")
    @patch("builtins.print")
    def test_train_main_prints_interrupt_message(
        self, mock_print: Mock, mock_create_parser: Mock, mock_run_pipeline: Mock
    ) -> None:
        """Test that train main prints appropriate message on KeyboardInterrupt."""
        mock_create_parser.return_value = Mock()
        mock_run_pipeline.side_effect = KeyboardInterrupt()

        train_main()

        print_calls = [call[0][0] for call in mock_print.call_args_list]
        assert any("🛑" in call for call in print_calls)
        assert any("💾" in call for call in print_calls)
