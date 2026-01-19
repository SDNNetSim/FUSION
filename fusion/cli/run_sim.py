"""
CLI entry point for running simulations.

Follows architecture best practice: entry points should have no logic.
"""

import multiprocessing
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
        return 1
    except Exception as e:  # noqa: BLE001
        # TODO (v6.1.0): Replace broad exception with specific error types - see cli/TODO.md
        logger.error(f"Simulation error: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
