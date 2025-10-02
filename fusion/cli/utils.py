"""Shared utilities for CLI entry points.

Provides common patterns and boilerplate code used across
CLI modules to reduce duplication and maintain consistency.
"""

import sys
from collections.abc import Callable


def create_entry_point_wrapper(
    main_func: Callable[[], int],
    _legacy_name: str,
    _entry_point_description: str,
) -> tuple[Callable[[], int], Callable[[], None]]:
    """Create standardized entry point wrapper functions.

    Generates both a legacy compatibility function and a main entry point
    function that handles sys.exit for any CLI main function.

    :param main_func: The main function that returns an exit code
    :type main_func: Callable[[], int]
    :param _legacy_name: Name for the legacy compatibility function (unused)
    :type _legacy_name: str
    :param _entry_point_description: Description of what the entry point does (unused)
    :type _entry_point_description: str
    :return: Tuple of (legacy_function, main_entry_function)
    :rtype: tuple[Callable[[], int], Callable[[], None]]
    """

    def legacy_main() -> int:
        """Legacy function name for main entry point.

        :return: Exit code from main function
        :rtype: int
        :deprecated: Use main() directly instead
        """
        return main_func()

    def main_entry() -> None:
        """Execute the main function and exit with appropriate code.

        Convenience function that handles the sys.exit call for the main
        entry point. Provides clean separation between main logic
        and process exit handling.

        :raises SystemExit: Always exits with code from main() function
        """
        sys.exit(main_func())

    return legacy_main, main_entry


def create_main_wrapper(main_func: Callable[[], int]) -> Callable[[], None]:
    """Create a simple main wrapper that calls sys.exit.

    :param main_func: The main function that returns an exit code
    :type main_func: Callable[[], int]
    :return: Wrapper function that calls sys.exit
    :rtype: Callable[[], None]
    """

    def wrapper() -> None:
        sys.exit(main_func())

    return wrapper
