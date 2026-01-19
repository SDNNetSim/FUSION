"""Unit tests for run_gui module."""

import pytest

from fusion.cli.run_gui import (
    GUINotSupportedError,
)
from fusion.cli.run_gui import (
    main as gui_main,
)


class TestGuiMain:
    """Tests for run_gui main function - GUI not supported."""

    def test_gui_main_raises_gui_not_supported_error(self) -> None:
        """Test GUI main raises GUINotSupportedError since GUI is not supported."""
        with pytest.raises(GUINotSupportedError) as exc_info:
            gui_main()

        assert "not supported" in str(exc_info.value).lower()

    def test_gui_not_supported_error_mentions_version(self) -> None:
        """Test that error message mentions when GUI will be available."""
        with pytest.raises(GUINotSupportedError) as exc_info:
            gui_main()

        assert "6.1.0" in str(exc_info.value)

    def test_gui_not_supported_error_provides_cli_alternative(self) -> None:
        """Test that error message provides CLI alternative."""
        with pytest.raises(GUINotSupportedError) as exc_info:
            gui_main()

        error_message = str(exc_info.value)
        assert "CLI" in error_message or "cli" in error_message


class TestGUINotSupportedError:
    """Tests for GUINotSupportedError exception class."""

    def test_gui_not_supported_error_is_exception(self) -> None:
        """Test that GUINotSupportedError is an Exception."""
        assert issubclass(GUINotSupportedError, Exception)

    def test_gui_not_supported_error_can_be_raised_with_message(self) -> None:
        """Test that GUINotSupportedError can be raised with custom message."""
        custom_message = "Custom error message"
        with pytest.raises(GUINotSupportedError) as exc_info:
            raise GUINotSupportedError(custom_message)

        assert custom_message in str(exc_info.value)
