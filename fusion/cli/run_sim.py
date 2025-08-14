# fusion/cli/run_sim.py

"""
CLI entry point for running simulations.
Follows architecture best practice: entry points should have no logic.
"""

import multiprocessing

from fusion.cli.main_parser import build_parser
from fusion.sim.run_simulation import run_simulation_pipeline


def main(stop_flag=None):
    """
    Entry point for running simulations from the command line.
    Delegates all logic to appropriate modules.

    Args:
        stop_flag: Optional multiprocessing event for stopping simulation
    """
    try:
        if stop_flag is None:
            stop_flag = multiprocessing.Event()

        parser = build_parser()
        args = parser.parse_args()

        # Delegate to simulation pipeline - no business logic here
        run_simulation_pipeline(args, stop_flag)

    except KeyboardInterrupt:
        print("\n🛑 Simulation interrupted by user")
        return 1
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"❌ Error running simulation: {e}")
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
