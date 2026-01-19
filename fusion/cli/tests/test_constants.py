"""Unit tests for constants module."""

from fusion.cli.constants import (
    DEFAULT_MAX_TRACEBACK_LINES,
    ERROR_EXIT_CODE,
    INTERRUPT_EXIT_CODE,
    SUCCESS_EXIT_CODE,
)


class TestExitCodes:
    """Tests for CLI exit code constants."""

    def test_exit_codes_are_integers(self) -> None:
        """Test that all exit codes are integers."""
        assert isinstance(SUCCESS_EXIT_CODE, int)
        assert isinstance(ERROR_EXIT_CODE, int)
        assert isinstance(INTERRUPT_EXIT_CODE, int)

    def test_exit_codes_follow_unix_conventions(self) -> None:
        """Test that exit codes follow Unix conventions."""
        assert SUCCESS_EXIT_CODE == 0
        assert ERROR_EXIT_CODE == 1
        assert INTERRUPT_EXIT_CODE == 1

    def test_success_exit_code_is_zero(self) -> None:
        """Test that success exit code is zero."""
        assert SUCCESS_EXIT_CODE == 0

    def test_error_exit_code_is_nonzero(self) -> None:
        """Test that error exit code is non-zero."""
        assert ERROR_EXIT_CODE != 0

    def test_interrupt_exit_code_is_nonzero(self) -> None:
        """Test that interrupt exit code is non-zero."""
        assert INTERRUPT_EXIT_CODE != 0


class TestTracebackSettings:
    """Tests for traceback configuration constants."""

    def test_default_traceback_lines_is_positive(self) -> None:
        """Test that default traceback lines is a positive integer."""
        assert isinstance(DEFAULT_MAX_TRACEBACK_LINES, int)
        assert DEFAULT_MAX_TRACEBACK_LINES > 0

    def test_default_traceback_lines_is_reasonable(self) -> None:
        """Test that default traceback lines is a reasonable value."""
        assert 1 <= DEFAULT_MAX_TRACEBACK_LINES <= 20
