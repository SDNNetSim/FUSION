"""CLI entry point for launching the FUSION GUI interface.

This module provides the command-line interface for launching the graphical
user interface. It handles GUI dependency validation, display configuration,
and provides helpful error messages for common setup issues.
"""

from fusion.cli.constants import ERROR_EXIT_CODE, INTERRUPT_EXIT_CODE, SUCCESS_EXIT_CODE
from fusion.cli.main_parser import create_gui_argument_parser
from fusion.cli.utils import create_entry_point_wrapper
from fusion.gui.runner import launch_gui_pipeline
from fusion.utils.logging_config import get_logger

logger = get_logger(__name__)


def main() -> int:
    """Launch the FUSION GUI interface.

    Parses command line arguments and delegates GUI launch operations
    to the appropriate GUI pipeline module. Handles user interruptions
    and errors gracefully with appropriate exit codes and user feedback.

    :return: Exit code (0 for success, 1 for error or interruption)
    :rtype: int
    :raises SystemExit: On argument parsing errors (handled by argparse)
    """
    try:
        gui_arguments = create_gui_argument_parser()

        launch_gui_pipeline(gui_arguments)

    except KeyboardInterrupt:
        logger.info("GUI launch interrupted by user")
        print("\n🛑 GUI launch interrupted by user")  # User-facing message
        return INTERRUPT_EXIT_CODE
    except (ImportError, ModuleNotFoundError) as e:
        logger.error(f"Missing GUI dependencies: {e}")
        print(f"❌ Missing GUI dependencies: {e}")  # User-facing message
        print("💡 Try installing GUI dependencies with: pip install -e .[gui]")
        return ERROR_EXIT_CODE
    except (OSError, RuntimeError) as e:
        logger.error(f"GUI framework error: {e}")
        print(f"❌ GUI framework error: {e}")  # User-facing message
        print("💡 Check your display settings and GUI framework installation")
        return ERROR_EXIT_CODE
    except (ValueError, TypeError) as e:
        logger.error(f"Configuration error launching GUI: {e}")
        print(f"❌ Configuration error launching GUI: {e}")  # User-facing message
        return ERROR_EXIT_CODE

    return SUCCESS_EXIT_CODE


# Create entry point functions using shared utilities
launch_gui_main, run_gui_main = create_entry_point_wrapper(
    main,
    "GUI",
    "Convenience function that handles the sys.exit call for the main GUI entry point."
    " Provides a clean separation between the main logic and process exit handling.",
)


if __name__ == "__main__":
    run_gui_main()
