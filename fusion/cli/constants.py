"""CLI-specific constants for FUSION command-line interfaces.

Provides shared constants used across CLI entry points including exit codes,
argument configurations, and common CLI settings to ensure consistency
and maintainability across the command-line interface modules.
"""

# Standard CLI exit codes following Unix conventions
SUCCESS_EXIT_CODE: int = 0
"""Exit code indicating successful program completion."""

ERROR_EXIT_CODE: int = 1
"""Exit code indicating program failure or error condition."""

INTERRUPT_EXIT_CODE: int = 1
"""Exit code indicating program interruption by user (Ctrl+C)."""

# Debugging and display settings
DEFAULT_MAX_TRACEBACK_LINES: int = 3
"""Default number of traceback lines to display in error messages."""
