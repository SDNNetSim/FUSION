"""
Simulation reporter for formatting and outputting simulation statistics.

This module handles all reporting and output formatting for simulation runs,
separating presentation logic from data collection.
"""

import logging
from statistics import mean
from typing import Optional, Dict, Any

from fusion.utils.logging_config import get_logger


class SimulationReporter:
    """
    Handles reporting and output formatting for simulation statistics.

    This class is responsible for presenting simulation results in various
    formats, including console output, log files, and structured reports.
    """

    def __init__(self, logger: Optional[logging.Logger] = None, verbose: bool = True):
        """
        Initialize the simulation reporter.

        :param logger: Logger instance to use (creates one if not provided)
        :param verbose: Whether to output detailed information
        """
        self.logger = logger or get_logger(__name__)
        self.verbose = verbose

    def report_iteration_stats(
            self,
            iteration: int,
            max_iterations: int,
            erlang: float,
            blocking_list: list,
            print_flag: bool = True
    ) -> None:
        """
        Report statistics for a completed iteration.

        This method replaces the print_iter_stats method from SimStats,
        providing proper logging and formatting.

        :param iteration: Current iteration number (0-based)
        :param max_iterations: Total number of iterations
        :param erlang: Erlang value for this simulation
        :param blocking_list: List of blocking probabilities from all iterations
        :param print_flag: Whether to output the statistics
        """
        if not print_flag:
            return

        # Calculate current blocking mean
        if blocking_list:
            blocking_mean = round(mean(blocking_list), 4)
        else:
            blocking_mean = 0.0

        # Format iteration progress
        iteration_num = iteration + 1  # Convert to 1-based for display
        progress_msg = (
            f"Iteration {iteration_num} out of {max_iterations} completed "
            f"for Erlang: {erlang}"
        )

        # Format blocking statistics
        stats_msg = f"Mean of blocking: {blocking_mean}"

        # Log at appropriate level
        if self.verbose:
            self.logger.info(progress_msg)
            self.logger.info(stats_msg)
        else:
            # For non-verbose, only log every 10th iteration
            if iteration_num % 10 == 0 or iteration_num == max_iterations:
                self.logger.info("Progress: %d/%d - %s", iteration_num, max_iterations, stats_msg)

    def report_simulation_start(self, sim_info: Dict[str, Any]) -> None:
        """
        Report the start of a simulation run.

        :param sim_info: Dictionary containing simulation parameters
        """
        self.logger.info("=" * 60)
        self.logger.info("SIMULATION STARTING")
        self.logger.info("=" * 60)

        for key, value in sim_info.items():
            self.logger.info("%s: %s", key, value)

        self.logger.info("=" * 60)

    def report_simulation_complete(
            self,
            erlang: float,
            iterations_completed: int,
            confidence_interval: Optional[float] = None
    ) -> None:
        """
        Report the completion of a simulation run.

        :param erlang: Erlang value that was simulated
        :param iterations_completed: Number of iterations completed
        :param confidence_interval: Optional confidence interval achieved
        """
        if confidence_interval is not None:
            self.logger.info(
                f"Confidence interval of {round(confidence_interval, 2)}% reached. "
                f"{iterations_completed} iterations completed for Erlang: {erlang}"
            )
        else:
            self.logger.info(
                f"Maximum iterations ({iterations_completed}) completed for Erlang: {erlang}"
            )

    def report_blocking_statistics(
            self,
            blocked_requests: int,
            total_requests: int,
            bit_rate_blocked: float,
            bit_rate_total: float,
            block_reasons: Dict[str, float]
    ) -> None:
        """
        Report detailed blocking statistics.

        :param blocked_requests: Number of blocked requests
        :param total_requests: Total number of requests
        :param bit_rate_blocked: Blocked bit rate
        :param bit_rate_total: Total bit rate requested
        :param block_reasons: Dictionary of blocking reasons with percentages
        """
        if total_requests == 0:
            return

        blocking_prob = blocked_requests / total_requests
        bit_rate_blocking_prob = bit_rate_blocked / bit_rate_total if bit_rate_total > 0 else 0

        self.logger.info("-" * 40)
        self.logger.info("BLOCKING STATISTICS")
        self.logger.info("-" * 40)
        self.logger.info("Request blocking probability: %.4f", blocking_prob)
        self.logger.info("Bit-rate blocking probability: %.4f", bit_rate_blocking_prob)

        if block_reasons and any(v > 0 for v in block_reasons.values()):
            self.logger.info("\nBlocking reasons:")
            for reason, percentage in block_reasons.items():
                if percentage > 0:
                    self.logger.info("  %s: %.2f%%", reason, percentage * 100)

        self.logger.info("-" * 40)

    def report_save_location(self, save_path: str) -> None:
        """
        Report where simulation results are being saved.

        :param save_path: Path where results are saved
        """
        self.logger.info(f"Saving results to: {save_path}")

    def report_error(self, error_msg: str, exception: Optional[Exception] = None) -> None:
        """
        Report an error during simulation.

        :param error_msg: Error message to log
        :param exception: Optional exception object
        """
        if exception:
            self.logger.error(f"{error_msg}: {type(exception).__name__}: {str(exception)}")
        else:
            self.logger.error(error_msg)

    def report_warning(self, warning_msg: str) -> None:
        """
        Report a warning during simulation.

        :param warning_msg: Warning message to log
        """
        self.logger.warning(warning_msg)

    def create_summary_report(self, stats_dict: Dict[str, Any]) -> str:
        """
        Create a formatted summary report from simulation statistics.

        :param stats_dict: Dictionary containing all simulation statistics
        :return: Formatted report as a string
        """
        report_lines = [
            "=" * 60,
            "SIMULATION SUMMARY REPORT",
            "=" * 60,
            "",
            f"Blocking Mean: {stats_dict.get('blocking_mean', 'N/A')}",
            f"Blocking Variance: {stats_dict.get('blocking_variance', 'N/A')}",
            f"Confidence Interval: {stats_dict.get('ci_percent_block', 'N/A')}%",
            "",
            f"Bit-rate Blocking Mean: {stats_dict.get('bit_rate_blocking_mean', 'N/A')}",
            f"Bit-rate Blocking Variance: {stats_dict.get('bit_rate_blocking_variance', 'N/A')}",
            f"Bit-rate Confidence Interval: {stats_dict.get('ci_percent_bit_rate_block', 'N/A')}%",
            "",
            "=" * 60
        ]

        return "\n".join(report_lines)
