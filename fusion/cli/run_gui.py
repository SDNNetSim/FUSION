"""
CLI entry point for launching the FUSION GUI interface.

This module provides the command-line interface for launching the graphical
user interface. It handles GUI dependency validation, display configuration,
and provides helpful error messages for common setup issues.
"""

import sys

from fusion.cli.constants import SUCCESS_EXIT_CODE, ERROR_EXIT_CODE, INTERRUPT_EXIT_CODE
from fusion.cli.main_parser import create_gui_argument_parser
from fusion.gui.runner import launch_gui_pipeline


def main() -> int:
    """
    Main entry point for launching the FUSION GUI interface.
    
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
        print("\n🛑 GUI launch interrupted by user")
        return INTERRUPT_EXIT_CODE
    except (ImportError, ModuleNotFoundError) as e:
        print(f"❌ Missing GUI dependencies: {e}")
        print("💡 Try installing GUI dependencies with: pip install -e .[gui]")
        return ERROR_EXIT_CODE
    except (OSError, RuntimeError) as e:
        print(f"❌ GUI framework error: {e}")
        print("💡 Check your display settings and GUI framework installation")
        return ERROR_EXIT_CODE
    except (ValueError, TypeError) as e:
        print(f"❌ Configuration error launching GUI: {e}")
        return ERROR_EXIT_CODE

    return SUCCESS_EXIT_CODE


# Backward compatibility function alias
def launch_gui_main() -> int:
    """
    Legacy function name for main GUI entry point.
    
    :return: Exit code from main function
    :rtype: int
    :deprecated: Use main() directly instead
    """
    return main()


def run_gui_main() -> None:
    """
    Execute the GUI main function and exit with appropriate code.
    
    Convenience function that handles the sys.exit call for the main
    GUI entry point. Provides a clean separation between the main logic
    and process exit handling.

    :raises SystemExit: Always exits with code from main() function
    """
    sys.exit(main())


if __name__ == "__main__":
    run_gui_main()
