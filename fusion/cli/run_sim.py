"""
CLI entry point for running simulations.

Follows architecture best practice: entry points should have no logic.
"""

import multiprocessing
import traceback
from typing import Any

from fusion.cli.main_parser import build_parser
from fusion.sim.run_simulation import run_simulation_pipeline
from fusion.utils.logging_config import get_logger

logger = get_logger(__name__)


def main(stop_flag: Any = None) -> int:
    """
    Entry point for running simulations from the command line.

    Delegates all logic to appropriate modules following the clean architecture
    pattern where entry points contain minimal logic and delegate to pipelines.

    :param stop_flag: Optional multiprocessing event for stopping simulation
    :type stop_flag: Any
    :return: Exit code (0 for success, 1 for error or interruption)
    :rtype: int
    :raises KeyboardInterrupt: When user interrupts the simulation
    """
    try:
        if stop_flag is None:
            stop_flag = multiprocessing.Event()

        parser = build_parser()
        args = parser.parse_args()

        # Delegate to simulation pipeline - no business logic here
        run_simulation_pipeline(args, stop_flag)

    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
        print("\nSimulation interrupted by user")  # User-facing message
        return 1
    except Exception as e:  # pylint: disable=broad-exception-caught
        # Custom error handling system pending - see cli/TODO.md for implementation plan
        # Log detailed error information for debugging
        logger.error(f"Simulation error: {e}", exc_info=True)

        # User-facing error message
        print(f"Error running simulation: {e}")

        # If it's a runtime error with more context, show the full chain
        if hasattr(e, "__cause__") and e.__cause__:
            print(f"  ↳ Caused by: {e.__cause__}")
            cause = e.__cause__
            if hasattr(cause, "__cause__") and cause.__cause__:
                print(f"    ↳ Root cause: {cause.__cause__}")

        # Show exception type for better debugging
        print(f"  Exception type: {type(e).__name__}")

        # For debugging: show a few lines of traceback
        print("  Last few calls:")
        tb_lines = traceback.format_tb(e.__traceback__)
        for line in tb_lines[-3:]:  # Show last 3 calls
            print(f"    {line.strip()}")

        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
