# fusion/cli/run_gui.py

"""
CLI entry point for launching the GUI.
Follows architecture best practice: entry points should have no logic.
"""

from fusion.cli.main_parser import get_gui_args
from fusion.gui.runner import launch_gui_pipeline


def main():
    """
    Entry point for launching the GUI from the command line.
    Delegates all logic to appropriate modules.
    """
    try:
        args = get_gui_args()

        # Delegate to GUI pipeline - no business logic here
        launch_gui_pipeline(args)

    except KeyboardInterrupt:
        print("\n🛑 GUI launch interrupted by user")
        return 1
    except Exception as e:  # pylint: disable=broad-exception-caught
        # TODO: Replace with custom error module and specific exception types
        # Consider: ImportError, ModuleNotFoundError, GUI framework errors
        print(f"❌ Error launching GUI: {e}")
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
