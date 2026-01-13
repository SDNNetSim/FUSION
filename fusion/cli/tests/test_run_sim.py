"""Unit tests for run_sim module."""

from typing import Any
from unittest.mock import Mock, patch

from fusion.cli.run_sim import main as sim_main


class TestSimMain:
    """Tests for run_sim main function."""

    @patch("fusion.cli.run_sim.run_simulation_pipeline")
    @patch("fusion.cli.run_sim.build_parser")
    def test_sim_main_success_returns_zero(
        self, mock_build_parser: Any, mock_run_pipeline: Any
    ) -> None:
        """Test sim main returns 0 on successful execution."""
        mock_parser = Mock()
        mock_args = Mock()
        mock_parser.parse_args.return_value = mock_args
        mock_build_parser.return_value = mock_parser

        result = sim_main()

        assert result == 0
        mock_build_parser.assert_called_once()
        mock_run_pipeline.assert_called_once()

    @patch("fusion.cli.run_sim.run_simulation_pipeline")
    @patch("fusion.cli.run_sim.build_parser")
    def test_sim_main_handles_keyboard_interrupt(
        self, mock_build_parser: Any, mock_run_pipeline: Any
    ) -> None:
        """Test sim main handles KeyboardInterrupt gracefully."""
        mock_parser = Mock()
        mock_args = Mock()
        mock_parser.parse_args.return_value = mock_args
        mock_build_parser.return_value = mock_parser
        mock_run_pipeline.side_effect = KeyboardInterrupt()

        result = sim_main()

        assert result == 1

    @patch("fusion.cli.run_sim.run_simulation_pipeline")
    @patch("fusion.cli.run_sim.build_parser")
    def test_sim_main_handles_general_exception(
        self, mock_build_parser: Any, mock_run_pipeline: Any
    ) -> None:
        """Test sim main handles general exceptions."""
        mock_parser = Mock()
        mock_args = Mock()
        mock_parser.parse_args.return_value = mock_args
        mock_build_parser.return_value = mock_parser
        mock_run_pipeline.side_effect = ValueError("Test error")

        result = sim_main()

        assert result == 1

    @patch("fusion.cli.run_sim.run_simulation_pipeline")
    @patch("fusion.cli.run_sim.build_parser")
    def test_sim_main_passes_args_to_pipeline(
        self, mock_build_parser: Any, mock_run_pipeline: Any
    ) -> None:
        """Test sim main passes parsed arguments to simulation pipeline."""
        mock_parser = Mock()
        mock_args = Mock(config_path="test.ini")
        mock_parser.parse_args.return_value = mock_args
        mock_build_parser.return_value = mock_parser

        sim_main()

        # Verify args are passed to pipeline (along with stop_flag)
        assert mock_run_pipeline.called
        call_args = mock_run_pipeline.call_args[0]
        assert call_args[0] == mock_args

    @patch("fusion.cli.run_sim.run_simulation_pipeline")
    @patch("fusion.cli.run_sim.build_parser")
    def test_sim_main_creates_parser_before_running_pipeline(
        self, mock_build_parser: Any, mock_run_pipeline: Any
    ) -> None:
        """Test sim main creates parser before running pipeline."""
        mock_parser = Mock()
        mock_args = Mock()
        mock_parser.parse_args.return_value = mock_args
        mock_build_parser.return_value = mock_parser

        sim_main()

        # Verify both functions were called
        assert mock_build_parser.called
        assert mock_run_pipeline.called

    @patch("fusion.cli.run_sim.run_simulation_pipeline")
    @patch("fusion.cli.run_sim.build_parser")
    @patch("fusion.cli.run_sim.logger")
    def test_sim_main_logs_interrupt_message(
        self, mock_logger: Any, mock_build_parser: Any, mock_run_pipeline: Any
    ) -> None:
        """Test sim main logs appropriate message on KeyboardInterrupt."""
        mock_parser = Mock()
        mock_args = Mock()
        mock_parser.parse_args.return_value = mock_args
        mock_build_parser.return_value = mock_parser
        mock_run_pipeline.side_effect = KeyboardInterrupt()

        sim_main()

        mock_logger.info.assert_called()
        log_message = mock_logger.info.call_args[0][0].lower()
        assert "interrupted" in log_message

    @patch("fusion.cli.run_sim.run_simulation_pipeline")
    @patch("fusion.cli.run_sim.build_parser")
    @patch("fusion.cli.run_sim.logger")
    def test_sim_main_logs_error_message_on_exception(
        self, mock_logger: Any, mock_build_parser: Any, mock_run_pipeline: Any
    ) -> None:
        """Test sim main logs error message on general exception."""
        mock_parser = Mock()
        mock_args = Mock()
        mock_parser.parse_args.return_value = mock_args
        mock_build_parser.return_value = mock_parser
        mock_run_pipeline.side_effect = RuntimeError("Test error")

        sim_main()

        mock_logger.error.assert_called()

    @patch("fusion.cli.run_sim.multiprocessing.Event")
    @patch("fusion.cli.run_sim.run_simulation_pipeline")
    @patch("fusion.cli.run_sim.build_parser")
    def test_sim_main_creates_stop_flag_when_none_provided(
        self, mock_build_parser: Any, mock_run_pipeline: Any, mock_event: Any
    ) -> None:
        """Test sim main creates multiprocessing Event when no stop_flag provided."""
        mock_parser = Mock()
        mock_args = Mock()
        mock_parser.parse_args.return_value = mock_args
        mock_build_parser.return_value = mock_parser
        mock_stop_flag = Mock()
        mock_event.return_value = mock_stop_flag

        sim_main()

        mock_event.assert_called_once()
        mock_run_pipeline.assert_called_once_with(mock_args, mock_stop_flag)

    @patch("fusion.cli.run_sim.run_simulation_pipeline")
    @patch("fusion.cli.run_sim.build_parser")
    def test_sim_main_uses_provided_stop_flag(
        self, mock_build_parser: Any, mock_run_pipeline: Any
    ) -> None:
        """Test sim main uses provided stop_flag parameter."""
        mock_parser = Mock()
        mock_args = Mock()
        mock_parser.parse_args.return_value = mock_args
        mock_build_parser.return_value = mock_parser
        mock_stop_flag = Mock()

        sim_main(stop_flag=mock_stop_flag)

        mock_run_pipeline.assert_called_once_with(mock_args, mock_stop_flag)

    @patch("fusion.cli.run_sim.run_simulation_pipeline")
    @patch("fusion.cli.run_sim.build_parser")
    @patch("fusion.cli.run_sim.logger")
    def test_sim_main_logs_exception_with_traceback(
        self, mock_logger: Any, mock_build_parser: Any, mock_run_pipeline: Any
    ) -> None:
        """Test sim main logs exception with traceback info."""
        mock_parser = Mock()
        mock_args = Mock()
        mock_parser.parse_args.return_value = mock_args
        mock_build_parser.return_value = mock_parser
        mock_run_pipeline.side_effect = ValueError("Test error")

        sim_main()

        # Should log error with exc_info=True for traceback
        mock_logger.error.assert_called()
        # Check that exc_info was passed for traceback
        call_kwargs = mock_logger.error.call_args[1]
        assert call_kwargs.get("exc_info") is True
