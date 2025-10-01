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
    @patch("builtins.print")
    def test_sim_main_prints_interrupt_message(
        self, mock_print: Any, mock_build_parser: Any, mock_run_pipeline: Any
    ) -> None:
        """Test sim main prints appropriate message on KeyboardInterrupt."""
        mock_parser = Mock()
        mock_args = Mock()
        mock_parser.parse_args.return_value = mock_args
        mock_build_parser.return_value = mock_parser
        mock_run_pipeline.side_effect = KeyboardInterrupt()

        sim_main()

        print_calls = [call[0][0] for call in mock_print.call_args_list]
        assert any("interrupted" in call.lower() for call in print_calls)

    @patch("fusion.cli.run_sim.run_simulation_pipeline")
    @patch("fusion.cli.run_sim.build_parser")
    @patch("builtins.print")
    def test_sim_main_prints_error_message_on_exception(
        self, mock_print: Any, mock_build_parser: Any, mock_run_pipeline: Any
    ) -> None:
        """Test sim main prints error message on general exception."""
        mock_parser = Mock()
        mock_args = Mock()
        mock_parser.parse_args.return_value = mock_args
        mock_build_parser.return_value = mock_parser
        mock_run_pipeline.side_effect = RuntimeError("Test error")

        sim_main()

        print_calls = [call[0][0] for call in mock_print.call_args_list]
        assert any("error" in call.lower() for call in print_calls)

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
    @patch("builtins.print")
    def test_sim_main_shows_exception_traceback_info(
        self, mock_print: Any, mock_build_parser: Any, mock_run_pipeline: Any
    ) -> None:
        """Test sim main shows exception type and traceback info."""
        mock_parser = Mock()
        mock_args = Mock()
        mock_parser.parse_args.return_value = mock_args
        mock_build_parser.return_value = mock_parser
        mock_run_pipeline.side_effect = ValueError("Test error")

        sim_main()

        print_calls = [call[0][0] for call in mock_print.call_args_list]
        # Should show exception type
        assert any("Exception type:" in call for call in print_calls)
        # Should show traceback info
        assert any("Last few calls:" in call for call in print_calls)

