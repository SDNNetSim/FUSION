"""
Simulation execution entry points for FUSION.

This module provides legacy and compatibility entry points for running
FUSION simulations, maintaining backward compatibility while redirecting
to the modern batch runner implementation.
"""

from typing import Any

from fusion.sim.batch_runner import run_batch_simulation


def run_simulation(config_dict: dict[str, Any]) -> dict | None:
    """
    Legacy entry point maintained for backward compatibility.

    New code should use run_batch_simulation directly from batch_runner module.

    :param config_dict: Configuration dictionary for simulation
    :type config_dict: Dict[str, Any]
    :return: First simulation result for compatibility, None if no results
    :rtype: Optional[Dict]
    """
    # Use the new batch runner with single-threaded execution
    results = run_batch_simulation(config_dict, parallel=False)

    # Return first result for compatibility
    return results[0] if results else None


def run_simulation_pipeline(args: Any, stop_flag: Any = None) -> list[dict]:  # pylint: disable=unused-argument
    """
    Pipeline function for running simulations from CLI.

    Now uses the new batch_runner orchestrator for improved performance
    and reliability.

    :param args: Parsed command line arguments
    :type args: Any
    :param stop_flag: Optional threading stop flag for cancellation
    :type stop_flag: Any
    :return: List of simulation results
    :rtype: List[Dict]
    """
    from fusion.cli.config_setup import (  # pylint: disable=import-outside-toplevel
        load_and_validate_config,
    )

    # Convert args to config dictionary
    config_dict = load_and_validate_config(args)

    # Determine if parallel execution is requested
    parallel = args.parallel if hasattr(args, "parallel") else False
    num_processes = args.num_processes if hasattr(args, "num_processes") else None

    # Run using the new batch runner
    results = run_batch_simulation(
        config_dict, parallel=parallel, num_processes=num_processes
    )

    return results
