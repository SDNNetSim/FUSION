"""Unit tests for utils module."""

from unittest.mock import Mock

import pytest

from fusion.cli.utils import create_entry_point_wrapper, create_main_wrapper


class TestCreateEntryPointWrapper:
    """Tests for create_entry_point_wrapper function."""

    def test_create_entry_point_wrapper_returns_correct_functions(self) -> None:
        """Test that entry point wrapper returns two callable functions."""

        def dummy_main() -> int:
            return 0

        legacy_func, main_func = create_entry_point_wrapper(
            dummy_main, "test", "Test description"
        )

        assert callable(legacy_func)
        assert callable(main_func)

    def test_create_entry_point_wrapper_legacy_function_calls_main(self) -> None:
        """Test that legacy function calls the main function correctly."""
        mock_main = Mock(return_value=42)

        legacy_func, _ = create_entry_point_wrapper(
            mock_main, "test", "Test description"
        )
        result = legacy_func()

        assert result == 42
        mock_main.assert_called_once()

    def test_create_entry_point_wrapper_main_function_exits_with_code(self) -> None:
        """Test that main entry function calls sys.exit with correct code."""
        mock_main = Mock(return_value=42)

        _, main_func = create_entry_point_wrapper(mock_main, "test", "Test description")

        with pytest.raises(SystemExit) as exc_info:
            main_func()
        assert exc_info.value.code == 42

    def test_create_entry_point_wrapper_handles_zero_exit_code(self) -> None:
        """Test that wrapper handles zero exit code correctly."""
        mock_main = Mock(return_value=0)

        _, main_func = create_entry_point_wrapper(mock_main, "test", "Test description")

        with pytest.raises(SystemExit) as exc_info:
            main_func()
        assert exc_info.value.code == 0

    def test_create_entry_point_wrapper_handles_nonzero_exit_code(self) -> None:
        """Test that wrapper handles non-zero exit code correctly."""
        mock_main = Mock(return_value=1)

        legacy_func, main_func = create_entry_point_wrapper(
            mock_main, "test", "Test description"
        )

        # Test legacy function
        assert legacy_func() == 1

        # Test main function
        with pytest.raises(SystemExit) as exc_info:
            main_func()
        assert exc_info.value.code == 1

    def test_create_entry_point_wrapper_with_different_exit_codes(self) -> None:
        """Test wrapper with various exit codes."""
        test_codes = [0, 1, 2, 42, 127, 255]

        for code in test_codes:
            mock_main = Mock(return_value=code)
            legacy_func, main_func = create_entry_point_wrapper(
                mock_main, "test", "Test description"
            )

            # Test legacy function returns the code
            assert legacy_func() == code

            # Test main function exits with the code
            with pytest.raises(SystemExit) as exc_info:
                main_func()
            assert exc_info.value.code == code


class TestCreateMainWrapper:
    """Tests for create_main_wrapper function."""

    def test_create_main_wrapper_calls_sys_exit(self) -> None:
        """Test that create_main_wrapper creates function that calls sys.exit."""
        mock_main = Mock(return_value=13)

        wrapper = create_main_wrapper(mock_main)

        with pytest.raises(SystemExit) as exc_info:
            wrapper()
        assert exc_info.value.code == 13
        mock_main.assert_called_once()

    def test_create_main_wrapper_returns_callable(self) -> None:
        """Test that create_main_wrapper returns a callable function."""
        mock_main = Mock(return_value=0)

        wrapper = create_main_wrapper(mock_main)

        assert callable(wrapper)

    def test_create_main_wrapper_with_zero_exit_code(self) -> None:
        """Test create_main_wrapper with zero exit code."""
        mock_main = Mock(return_value=0)

        wrapper = create_main_wrapper(mock_main)

        with pytest.raises(SystemExit) as exc_info:
            wrapper()
        assert exc_info.value.code == 0

    def test_create_main_wrapper_with_various_exit_codes(self) -> None:
        """Test create_main_wrapper with different exit codes."""
        test_codes = [0, 1, 42, 127]

        for code in test_codes:
            mock_main = Mock(return_value=code)
            wrapper = create_main_wrapper(mock_main)

            with pytest.raises(SystemExit) as exc_info:
                wrapper()
            assert exc_info.value.code == code
            mock_main.assert_called_once()

    def test_create_main_wrapper_preserves_main_function_behavior(self) -> None:
        """Test that wrapper preserves the behavior of the main function."""
        call_count = 0

        def counting_main() -> int:
            nonlocal call_count
            call_count += 1
            return call_count

        wrapper = create_main_wrapper(counting_main)

        # First call
        with pytest.raises(SystemExit) as exc_info:
            wrapper()
        assert exc_info.value.code == 1
        assert call_count == 1

        # Second call
        with pytest.raises(SystemExit) as exc_info:
            wrapper()
        assert exc_info.value.code == 2
        assert call_count == 2
