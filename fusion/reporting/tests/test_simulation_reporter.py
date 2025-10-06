"""
Unit tests for fusion.reporting.simulation_reporter module.

This module provides comprehensive testing for the SimulationReporter class which
handles formatting and outputting simulation statistics, including iteration stats,
blocking statistics, and summary reports.
"""

import logging
import unittest
from typing import Any
from unittest.mock import Mock

from fusion.reporting.simulation_reporter import SimulationReporter


class TestSimulationReporterInit(unittest.TestCase):
    """Unit tests for SimulationReporter initialization."""

    def test_init_with_no_logger_creates_default_logger(self) -> None:
        """Test initialization without logger creates default logger."""
        # Act
        reporter = SimulationReporter()

        # Assert
        self.assertIsNotNone(reporter.logger)
        self.assertIsInstance(reporter.logger, logging.Logger)
        self.assertTrue(reporter.verbose)

    def test_init_with_custom_logger_uses_provided_logger(self) -> None:
        """Test initialization with custom logger uses the provided logger."""
        # Arrange
        mock_logger = Mock(spec=logging.Logger)

        # Act
        reporter = SimulationReporter(logger=mock_logger)

        # Assert
        self.assertEqual(reporter.logger, mock_logger)

    def test_init_with_verbose_false_sets_verbose_flag(self) -> None:
        """Test initialization with verbose=False sets the flag correctly."""
        # Act
        reporter = SimulationReporter(verbose=False)

        # Assert
        self.assertFalse(reporter.verbose)

    def test_init_with_verbose_true_sets_verbose_flag(self) -> None:
        """Test initialization with verbose=True sets the flag correctly."""
        # Act
        reporter = SimulationReporter(verbose=True)

        # Assert
        self.assertTrue(reporter.verbose)


class TestReportIterationStats(unittest.TestCase):
    """Unit tests for report_iteration_stats method."""

    mock_logger: Mock
    reporter: SimulationReporter

    def setUp(self) -> None:
        """Set up test fixtures with mock logger."""
        self.mock_logger = Mock(spec=logging.Logger)
        self.reporter = SimulationReporter(logger=self.mock_logger, verbose=True)

    def test_report_iteration_stats_with_print_false_does_nothing(self) -> None:
        """Test that print_flag=False prevents any output."""
        # Arrange
        blocking_list = [0.1, 0.2, 0.3]

        # Act
        self.reporter.report_iteration_stats(
            iteration=0,
            max_iterations=10,
            erlang=100.0,
            blocking_list=blocking_list,
            print_flag=False,
        )

        # Assert
        self.mock_logger.info.assert_not_called()

    def test_report_iteration_stats_with_valid_data_logs_correctly(self) -> None:
        """Test iteration stats logging with valid blocking data."""
        # Arrange
        blocking_list = [0.1, 0.2, 0.3]
        expected_mean = 0.2

        # Act
        self.reporter.report_iteration_stats(
            iteration=0,
            max_iterations=10,
            erlang=100.0,
            blocking_list=blocking_list,
            print_flag=True,
        )

        # Assert
        self.assertEqual(self.mock_logger.info.call_count, 2)
        progress_call = self.mock_logger.info.call_args_list[0][0][0]
        stats_call = self.mock_logger.info.call_args_list[1][0][0]
        self.assertIn("Iteration 1 out of 10", progress_call)
        self.assertIn("Erlang: 100.0", progress_call)
        self.assertIn(f"Mean of blocking: {expected_mean}", stats_call)

    def test_report_iteration_stats_with_empty_blocking_list_uses_zero(self) -> None:
        """Test iteration stats with empty blocking list defaults to 0.0."""
        # Act
        self.reporter.report_iteration_stats(
            iteration=5,
            max_iterations=10,
            erlang=150.0,
            blocking_list=[],
            print_flag=True,
        )

        # Assert
        stats_call = self.mock_logger.info.call_args_list[1][0][0]
        self.assertIn("Mean of blocking: 0.0", stats_call)

    def test_report_iteration_stats_with_verbose_false_logs_every_tenth(self) -> None:
        """Test non-verbose mode logs only every 10th iteration."""
        # Arrange
        self.reporter.verbose = False

        # Act - iteration 5 (not 10th)
        self.reporter.report_iteration_stats(
            iteration=4,
            max_iterations=100,
            erlang=100.0,
            blocking_list=[0.1],
            print_flag=True,
        )

        # Assert - should not log
        self.mock_logger.info.assert_not_called()

    def test_report_iteration_stats_verbose_false_logs_tenth_iteration(self) -> None:
        """Test non-verbose mode logs on 10th iteration."""
        # Arrange
        self.reporter.verbose = False

        # Act - iteration 10 (10th iteration)
        self.reporter.report_iteration_stats(
            iteration=9,
            max_iterations=100,
            erlang=100.0,
            blocking_list=[0.1],
            print_flag=True,
        )

        # Assert - should log once
        self.assertEqual(self.mock_logger.info.call_count, 1)
        call_args = self.mock_logger.info.call_args[0]
        self.assertIn("Progress", call_args[0])
        self.assertEqual(call_args[1], 10)  # iteration_num
        self.assertEqual(call_args[2], 100)  # max_iterations

    def test_report_iteration_stats_verbose_false_logs_last_iteration(self) -> None:
        """Test non-verbose mode logs on final iteration."""
        # Arrange
        self.reporter.verbose = False

        # Act - last iteration (not a multiple of 10)
        self.reporter.report_iteration_stats(
            iteration=99,
            max_iterations=100,
            erlang=100.0,
            blocking_list=[0.1],
            print_flag=True,
        )

        # Assert - should log
        self.assertEqual(self.mock_logger.info.call_count, 1)

    def test_report_iteration_stats_converts_to_one_based_iteration(self) -> None:
        """Test iteration number is converted from 0-based to 1-based for display."""
        # Act
        self.reporter.report_iteration_stats(
            iteration=0,
            max_iterations=10,
            erlang=100.0,
            blocking_list=[0.1],
            print_flag=True,
        )

        # Assert
        progress_call = self.mock_logger.info.call_args_list[0][0][0]
        self.assertIn("Iteration 1 out of 10", progress_call)


class TestReportSimulationStart(unittest.TestCase):
    """Unit tests for report_simulation_start method."""

    mock_logger: Mock
    reporter: SimulationReporter

    def setUp(self) -> None:
        """Set up test fixtures with mock logger."""
        self.mock_logger = Mock(spec=logging.Logger)
        self.reporter = SimulationReporter(logger=self.mock_logger)

    def test_report_simulation_start_logs_header_and_info(self) -> None:
        """Test simulation start report logs header and simulation info."""
        # Arrange
        sim_info = {"topology": "NSFNET", "erlang": 100, "algorithm": "FirstFit"}

        # Act
        self.reporter.report_simulation_start(sim_info)

        # Assert
        self.assertTrue(self.mock_logger.info.call_count >= 5)
        calls = [call[0][0] for call in self.mock_logger.info.call_args_list]
        self.assertIn("=" * 60, calls)
        self.assertIn("SIMULATION STARTING", calls)

    def test_report_simulation_start_logs_all_dict_entries(self) -> None:
        """Test all dictionary entries are logged correctly."""
        # Arrange
        sim_info = {"param1": "value1", "param2": 42}

        # Act
        self.reporter.report_simulation_start(sim_info)

        # Assert
        # Check that logger.info was called with each key-value pair
        call_args_list = self.mock_logger.info.call_args_list
        logged_messages = [str(call) for call in call_args_list]
        combined_logs = " ".join(logged_messages)
        self.assertIn("param1", combined_logs)
        self.assertIn("param2", combined_logs)

    def test_report_simulation_start_with_empty_dict_logs_headers_only(self) -> None:
        """Test simulation start with empty dict only logs headers."""
        # Act
        self.reporter.report_simulation_start({})

        # Assert
        # Should have: top separator, title, bottom separator, final separator
        self.assertEqual(self.mock_logger.info.call_count, 4)


class TestReportSimulationComplete(unittest.TestCase):
    """Unit tests for report_simulation_complete method."""

    mock_logger: Mock
    reporter: SimulationReporter

    def setUp(self) -> None:
        """Set up test fixtures with mock logger."""
        self.mock_logger = Mock(spec=logging.Logger)
        self.reporter = SimulationReporter(logger=self.mock_logger)

    def test_report_simulation_complete_with_confidence_interval_logs_ci(self) -> None:
        """Test completion report includes confidence interval when provided."""
        # Act
        self.reporter.report_simulation_complete(
            erlang=100.0, iterations_completed=50, confidence_interval=95.5
        )

        # Assert
        self.mock_logger.info.assert_called_once()
        call_message = self.mock_logger.info.call_args[0][0]
        self.assertIn("Confidence interval of 95.5%", call_message)
        self.assertIn("50 iterations", call_message)
        self.assertIn("Erlang: 100.0", call_message)

    def test_report_simulation_complete_without_ci_logs_max_iterations(self) -> None:
        """Test completion report without CI logs maximum iterations message."""
        # Act
        self.reporter.report_simulation_complete(
            erlang=100.0, iterations_completed=100, confidence_interval=None
        )

        # Assert
        self.mock_logger.info.assert_called_once()
        call_message = self.mock_logger.info.call_args[0][0]
        self.assertIn("Maximum iterations (100)", call_message)
        self.assertIn("Erlang: 100.0", call_message)
        self.assertNotIn("Confidence interval", call_message)


class TestReportBlockingStatistics(unittest.TestCase):
    """Unit tests for report_blocking_statistics method."""

    mock_logger: Mock
    reporter: SimulationReporter

    def setUp(self) -> None:
        """Set up test fixtures with mock logger."""
        self.mock_logger = Mock(spec=logging.Logger)
        self.reporter = SimulationReporter(logger=self.mock_logger)

    def test_report_blocking_stats_with_zero_requests_does_nothing(self) -> None:
        """Test blocking stats with zero total requests does nothing."""
        # Act
        self.reporter.report_blocking_statistics(
            blocked_requests=0,
            total_requests=0,
            bit_rate_blocked=0.0,
            bit_rate_total=0.0,
            blocking_reasons_dict={},
        )

        # Assert
        self.mock_logger.info.assert_not_called()

    def test_report_blocking_stats_with_valid_data_logs_probabilities(self) -> None:
        """Test blocking statistics are calculated and logged correctly."""
        # Arrange
        blocking_reasons = {"congestion": 0.6, "distance": 0.4}

        # Act
        self.reporter.report_blocking_statistics(
            blocked_requests=50,
            total_requests=100,
            bit_rate_blocked=250.0,
            bit_rate_total=500.0,
            blocking_reasons_dict=blocking_reasons,
        )

        # Assert
        self.assertTrue(self.mock_logger.info.call_count >= 5)
        # Check that blocking probability (0.5) was logged
        call_args_list = self.mock_logger.info.call_args_list
        # Find the call with blocking probability
        found_blocking_prob = False
        for call_args in call_args_list:
            args = call_args[0]
            if len(args) >= 2 and "Request blocking probability" in args[0]:
                self.assertAlmostEqual(args[1], 0.5, places=4)
                found_blocking_prob = True
                break
        self.assertTrue(found_blocking_prob, "Blocking probability not logged")

    def test_report_blocking_stats_with_zero_bit_rate_total_uses_zero(self) -> None:
        """Test bit rate blocking probability is 0 when total bit rate is 0."""
        # Act
        self.reporter.report_blocking_statistics(
            blocked_requests=10,
            total_requests=100,
            bit_rate_blocked=100.0,
            bit_rate_total=0.0,
            blocking_reasons_dict={},
        )

        # Assert
        # Check that bit rate blocking probability was logged as 0.0
        call_args_list = self.mock_logger.info.call_args_list
        found_bit_rate_prob = False
        for call_args in call_args_list:
            args = call_args[0]
            if len(args) >= 2 and "Bit-rate blocking probability" in args[0]:
                self.assertAlmostEqual(args[1], 0.0, places=4)
                found_bit_rate_prob = True
                break
        self.assertTrue(found_bit_rate_prob, "Bit-rate blocking probability not logged")

    def test_report_blocking_stats_logs_blocking_reasons(self) -> None:
        """Test blocking reasons are logged when non-zero."""
        # Arrange
        blocking_reasons = {"congestion": 0.7, "distance": 0.3, "xt_threshold": 0.0}

        # Act
        self.reporter.report_blocking_statistics(
            blocked_requests=50,
            total_requests=100,
            bit_rate_blocked=250.0,
            bit_rate_total=500.0,
            blocking_reasons_dict=blocking_reasons,
        )

        # Assert
        call_args_list = self.mock_logger.info.call_args_list
        # Find calls with blocking reasons
        logged_reasons = {}
        for call_args in call_args_list:
            args = call_args[0]
            if len(args) >= 3 and "%s: %.2f%%" in args[0]:
                reason = args[1]
                percentage = args[2]
                logged_reasons[reason] = percentage

        # Check that congestion and distance were logged with correct percentages
        self.assertIn("congestion", logged_reasons)
        self.assertAlmostEqual(logged_reasons["congestion"], 70.0, places=2)
        self.assertIn("distance", logged_reasons)
        self.assertAlmostEqual(logged_reasons["distance"], 30.0, places=2)
        # xt_threshold should not be logged since it's 0.0
        self.assertNotIn("xt_threshold", logged_reasons)

    def test_report_blocking_stats_skips_zero_blocking_reasons(self) -> None:
        """Test blocking reasons with 0% are not logged."""
        # Arrange
        blocking_reasons = {"congestion": 0.5, "distance": 0.0}

        # Act
        self.reporter.report_blocking_statistics(
            blocked_requests=50,
            total_requests=100,
            bit_rate_blocked=250.0,
            bit_rate_total=500.0,
            blocking_reasons_dict=blocking_reasons,
        )

        # Assert
        calls = [str(call) for call in self.mock_logger.info.call_args_list]
        combined = " ".join(calls)
        self.assertIn("congestion", combined)
        # distance with 0% should not be logged, but the word might appear
        # in other context, so we check the percentage is not there
        if "distance" in combined:
            self.assertNotIn("0.00%", combined)

    def test_report_blocking_stats_with_empty_reasons_dict_skips_reasons(self) -> None:
        """Test empty blocking reasons dict does not log reasons section."""
        # Act
        self.reporter.report_blocking_statistics(
            blocked_requests=50,
            total_requests=100,
            bit_rate_blocked=250.0,
            bit_rate_total=500.0,
            blocking_reasons_dict={},
        )

        # Assert
        calls = [str(call) for call in self.mock_logger.info.call_args_list]
        combined = " ".join(calls)
        self.assertNotIn("Blocking reasons:", combined)


class TestReportSaveLocation(unittest.TestCase):
    """Unit tests for report_save_location method."""

    mock_logger: Mock
    reporter: SimulationReporter

    def setUp(self) -> None:
        """Set up test fixtures with mock logger."""
        self.mock_logger = Mock(spec=logging.Logger)
        self.reporter = SimulationReporter(logger=self.mock_logger)

    def test_report_save_location_logs_path_correctly(self) -> None:
        """Test save location is logged with correct path."""
        # Arrange
        save_path = "/path/to/results/simulation_001.csv"

        # Act
        self.reporter.report_save_location(save_path)

        # Assert
        self.mock_logger.info.assert_called_once()
        call_message = self.mock_logger.info.call_args[0][0]
        self.assertIn("Saving results to:", call_message)
        self.assertIn(save_path, call_message)


class TestReportError(unittest.TestCase):
    """Unit tests for report_error method."""

    mock_logger: Mock
    reporter: SimulationReporter

    def setUp(self) -> None:
        """Set up test fixtures with mock logger."""
        self.mock_logger = Mock(spec=logging.Logger)
        self.reporter = SimulationReporter(logger=self.mock_logger)

    def test_report_error_with_exception_logs_full_error(self) -> None:
        """Test error report with exception includes exception details."""
        # Arrange
        error_msg = "Failed to process request"
        exception = ValueError("Invalid input value")

        # Act
        self.reporter.report_error(error_msg, exception)

        # Assert
        self.mock_logger.error.assert_called_once()
        call_message = self.mock_logger.error.call_args[0][0]
        self.assertIn(error_msg, call_message)
        self.assertIn("ValueError", call_message)
        self.assertIn("Invalid input value", call_message)

    def test_report_error_without_exception_logs_message_only(self) -> None:
        """Test error report without exception logs only the message."""
        # Arrange
        error_msg = "Failed to save results"

        # Act
        self.reporter.report_error(error_msg, exception=None)

        # Assert
        self.mock_logger.error.assert_called_once()
        call_message = self.mock_logger.error.call_args[0][0]
        self.assertEqual(call_message, error_msg)

    def test_report_error_with_different_exception_types(self) -> None:
        """Test error reporting works with various exception types."""
        # Arrange
        exceptions = [
            RuntimeError("Runtime issue"),
            FileNotFoundError("File missing"),
            KeyError("key"),
        ]

        # Act & Assert
        for exc in exceptions:
            self.reporter.report_error("Test error", exc)
            call_message = self.mock_logger.error.call_args[0][0]
            self.assertIn(type(exc).__name__, call_message)
            self.mock_logger.reset_mock()


class TestReportWarning(unittest.TestCase):
    """Unit tests for report_warning method."""

    mock_logger: Mock
    reporter: SimulationReporter

    def setUp(self) -> None:
        """Set up test fixtures with mock logger."""
        self.mock_logger = Mock(spec=logging.Logger)
        self.reporter = SimulationReporter(logger=self.mock_logger)

    def test_report_warning_logs_message_correctly(self) -> None:
        """Test warning message is logged correctly."""
        # Arrange
        warning_msg = "Low memory warning"

        # Act
        self.reporter.report_warning(warning_msg)

        # Assert
        self.mock_logger.warning.assert_called_once_with(warning_msg)


class TestCreateSummaryReport(unittest.TestCase):
    """Unit tests for create_summary_report method."""

    reporter: SimulationReporter

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.reporter = SimulationReporter()

    def test_create_summary_report_with_complete_stats_returns_formatted(self) -> None:
        """Test summary report with complete statistics returns formatted string."""
        # Arrange
        stats = {
            "blocking_mean": 0.15,
            "blocking_variance": 0.002,
            "ci_percent_block": 95.0,
            "bit_rate_blocking_mean": 0.12,
            "bit_rate_blocking_variance": 0.001,
            "ci_percent_bit_rate_block": 94.5,
        }

        # Act
        result = self.reporter.create_summary_report(stats)

        # Assert
        self.assertIsInstance(result, str)
        self.assertIn("SIMULATION SUMMARY REPORT", result)
        self.assertIn("Blocking Mean: 0.15", result)
        self.assertIn("Blocking Variance: 0.002", result)
        self.assertIn("Confidence Interval: 95.0%", result)
        self.assertIn("Bit-rate Blocking Mean: 0.12", result)
        self.assertIn("Bit-rate Blocking Variance: 0.001", result)
        self.assertIn("Bit-rate Confidence Interval: 94.5%", result)

    def test_create_summary_report_with_missing_stats_uses_na(self) -> None:
        """Test summary report with missing statistics uses N/A."""
        # Arrange
        stats: dict[str, Any] = {}

        # Act
        result = self.reporter.create_summary_report(stats)

        # Assert
        self.assertIn("N/A", result)
        # Should have N/A for each missing statistic
        self.assertEqual(result.count("N/A"), 6)

    def test_create_summary_report_with_partial_stats_mixes_values_and_na(
        self,
    ) -> None:
        """Test summary report with partial statistics shows both values and N/A."""
        # Arrange
        stats = {"blocking_mean": 0.15, "ci_percent_block": 95.0}

        # Act
        result = self.reporter.create_summary_report(stats)

        # Assert
        self.assertIn("Blocking Mean: 0.15", result)
        self.assertIn("Confidence Interval: 95.0%", result)
        self.assertIn("N/A", result)
        # Should have N/A for 4 missing statistics
        self.assertEqual(result.count("N/A"), 4)

    def test_create_summary_report_has_proper_formatting(self) -> None:
        """Test summary report has proper separator lines and structure."""
        # Arrange
        stats = {"blocking_mean": 0.15}

        # Act
        result = self.reporter.create_summary_report(stats)

        # Assert
        lines = result.split("\n")
        self.assertEqual(lines[0], "=" * 60)  # Top separator
        self.assertEqual(lines[1], "SIMULATION SUMMARY REPORT")  # Title
        self.assertEqual(lines[2], "=" * 60)  # Bottom separator
        self.assertEqual(lines[-1], "=" * 60)  # Final separator


if __name__ == "__main__":
    unittest.main()
