"""CLI entry point for running FUSION network simulations.

This module provides the main command-line interface for executing network
simulations. It handles argument parsing, multiprocessing coordination,
and provides comprehensive error reporting with actionable user feedback.
"""

import multiprocessing
import sys
import traceback
from typing import Optional

from fusion.cli.constants import SUCCESS_EXIT_CODE, ERROR_EXIT_CODE, INTERRUPT_EXIT_CODE, DEFAULT_MAX_TRACEBACK_LINES
from fusion.cli.main_parser import build_main_argument_parser
from fusion.sim.run_simulation import run_simulation_pipeline


def main(stop_flag: Optional[multiprocessing.Event] = None) -> int:
    """
    Main entry point for running FUSION network simulations.
    
    Parses command line arguments and delegates simulation execution
    to the appropriate simulation pipeline module. Handles multiprocessing
    coordination, user interruptions, and provides detailed error reporting
    for debugging purposes.

    :param stop_flag: Optional multiprocessing event for coordinated simulation stopping
    :type stop_flag: Optional[multiprocessing.Event]
    :return: Exit code (0 for success, 1 for error or interruption)
    :rtype: int
    :raises SystemExit: On argument parsing errors (handled by argparse)
    """
    try:
        if stop_flag is None:
            stop_flag = multiprocessing.Event()

        main_parser = build_main_argument_parser()
        simulation_arguments = main_parser.parse_args()

        run_simulation_pipeline(simulation_arguments, stop_flag)

    except KeyboardInterrupt:
        print("\n🛑 Simulation interrupted by user")
        if stop_flag is not None:
            stop_flag.set()  # Signal other processes to stop
        return INTERRUPT_EXIT_CODE
    except (ImportError, ModuleNotFoundError) as e:
        print(f"❌ Missing simulation dependencies: {e}")
        print("💡 Try installing dependencies with: pip install -e .")
        return ERROR_EXIT_CODE
    except (OSError, IOError) as e:
        print(f"❌ File system error: {e}")
        print("💡 Check file permissions and available disk space")
        return ERROR_EXIT_CODE
    except (ValueError, TypeError) as e:
        print(f"❌ Configuration error: {e}")
        print("💡 Check your configuration file and command line arguments")
        _display_detailed_error_info(e)
        return ERROR_EXIT_CODE
    except (RuntimeError, MemoryError) as e:
        print(f"❌ Runtime error: {e}")
        print("💡 Check system resources and simulation parameters")
        _display_detailed_error_info(e)
        return ERROR_EXIT_CODE

    return SUCCESS_EXIT_CODE


def _display_detailed_error_info(exception: Exception) -> None:
    """
    Display detailed error information for debugging purposes.
    
    Shows exception chain, type information, and relevant traceback lines
    to help users and developers diagnose simulation issues effectively.

    :param exception: The exception to analyze and display
    :type exception: Exception
    """
    # Show exception chain if available
    if hasattr(exception, '__cause__') and exception.__cause__:
        print(f"  ↳ Caused by: {exception.__cause__}")
        cause = exception.__cause__
        if hasattr(cause, '__cause__') and cause.__cause__:
            print(f"    ↳ Root cause: {cause.__cause__}")

    # Show exception type for better debugging
    print(f"  Exception type: {type(exception).__name__}")

    # Show relevant traceback lines
    print("  Last few calls:")
    traceback_lines = traceback.format_tb(exception.__traceback__)
    for line in traceback_lines[-DEFAULT_MAX_TRACEBACK_LINES:]:
        print(f"    {line.strip()}")


# Backward compatibility function alias
def run_simulation_main(stop_flag: Optional[multiprocessing.Event] = None) -> int:
    """
    Legacy function name for main simulation entry point.
    
    :param stop_flag: Optional multiprocessing event for stopping simulation
    :type stop_flag: Optional[multiprocessing.Event]
    :return: Exit code from main function
    :rtype: int
    :deprecated: Use main() directly instead
    """
    return main(stop_flag)


def run_sim_main() -> None:
    """
    Execute the simulation main function and exit with appropriate code.
    
    Convenience function that handles the sys.exit call for the main
    simulation entry point. Provides clean separation between main logic
    and process exit handling.

    :raises SystemExit: Always exits with code from main() function
    """
    sys.exit(main())


if __name__ == "__main__":
    run_sim_main()
