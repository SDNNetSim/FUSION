"""
CLI entry point for launching the FUSION GUI interface.

NOTE: The GUI is currently under active development and not supported.
This module will be updated in version 6.1.0.
"""


class GUINotSupportedError(Exception):
    """Raised when attempting to use the unsupported GUI module."""

    pass


def main() -> int:
    """
    Launch the FUSION GUI interface.

    :raises GUINotSupportedError: Always raised - GUI is not currently supported
    """
    raise GUINotSupportedError(
        "The FUSION GUI is currently under active development and not supported. "
        "This feature will be available in version 6.1.0. "
        "Please use the CLI interface instead: python -m fusion.cli.run_sim run_sim --config_path <path>"
    )


if __name__ == "__main__":
    main()
