"""Unit tests for fusion.sim.utils.formatting module."""


from typing import Any

from ..formatting import int_to_string, list_to_title, snake_to_title


class TestSnakeToTitle:
    """Tests for snake_case to Title Case conversion."""

    def test_snake_to_title_with_simple_string_converts_correctly(self) -> None:
        """Test basic snake_case to Title Case conversion."""
        # Arrange
        snake_str = "hello_world"
        expected = "Hello World"

        # Act
        result = snake_to_title(snake_str)

        # Assert
        assert result == expected

    def test_snake_to_title_with_single_word_capitalizes(self) -> None:
        """Test single word conversion."""
        # Arrange
        snake_str = "hello"
        expected = "Hello"

        # Act
        result = snake_to_title(snake_str)

        # Assert
        assert result == expected

    def test_snake_to_title_with_multiple_underscores_converts_correctly(self) -> None:
        """Test conversion with multiple underscores."""
        # Arrange
        snake_str = "this_is_a_test"
        expected = "This Is A Test"

        # Act
        result = snake_to_title(snake_str)

        # Assert
        assert result == expected

    def test_snake_to_title_with_empty_string_returns_empty(self) -> None:
        """Test conversion of empty string."""
        # Arrange
        snake_str = ""
        expected = ""

        # Act
        result = snake_to_title(snake_str)

        # Assert
        assert result == expected

    def test_snake_to_title_with_leading_underscore_handles_correctly(self) -> None:
        """Test conversion with leading underscore."""
        # Arrange
        snake_str = "_private_method"
        expected = " Private Method"

        # Act
        result = snake_to_title(snake_str)

        # Assert
        assert result == expected

    def test_snake_to_title_with_trailing_underscore_handles_correctly(self) -> None:
        """Test conversion with trailing underscore."""
        # Arrange
        snake_str = "method_name_"
        expected = "Method Name "

        # Act
        result = snake_to_title(snake_str)

        # Assert
        assert result == expected

    def test_snake_to_title_with_numbers_preserves_numbers(self) -> None:
        """Test conversion with numbers in string."""
        # Arrange
        snake_str = "test_123_value"
        expected = "Test 123 Value"

        # Act
        result = snake_to_title(snake_str)

        # Assert
        assert result == expected


class TestIntToString:
    """Tests for integer to formatted string conversion."""

    def test_int_to_string_with_large_number_adds_commas(self) -> None:
        """Test formatting large numbers with thousands separator."""
        # Arrange
        number = 1234567
        expected = "1,234,567"

        # Act
        result = int_to_string(number)

        # Assert
        assert result == expected

    def test_int_to_string_with_small_number_no_commas(self) -> None:
        """Test formatting numbers below 1000."""
        # Arrange
        number = 999
        expected = "999"

        # Act
        result = int_to_string(number)

        # Assert
        assert result == expected

    def test_int_to_string_with_zero_returns_zero(self) -> None:
        """Test formatting zero."""
        # Arrange
        number = 0
        expected = "0"

        # Act
        result = int_to_string(number)

        # Assert
        assert result == expected

    def test_int_to_string_with_negative_number_formats_correctly(self) -> None:
        """Test formatting negative numbers."""
        # Arrange
        number = -1234567
        expected = "-1,234,567"

        # Act
        result = int_to_string(number)

        # Assert
        assert result == expected

    def test_int_to_string_with_thousand_adds_comma(self) -> None:
        """Test formatting exactly 1000."""
        # Arrange
        number = 1000
        expected = "1,000"

        # Act
        result = int_to_string(number)

        # Assert
        assert result == expected

    def test_int_to_string_with_million_formats_correctly(self) -> None:
        """Test formatting one million."""
        # Arrange
        number = 1000000
        expected = "1,000,000"

        # Act
        result = int_to_string(number)

        # Assert
        assert result == expected


class TestListToTitle:
    """Tests for list to title string conversion."""

    def test_list_to_title_with_multiple_items_joins_with_ampersand(self) -> None:
        """Test converting list with multiple items."""
        # Arrange
        input_list: list[tuple[str, Any]] = [
            ("Alice", None), ("Bob", None), ("Charlie", None)
        ]
        expected = "Alice, Bob & Charlie"

        # Act
        result = list_to_title(input_list)

        # Assert
        assert result == expected

    def test_list_to_title_with_single_item_returns_item(self) -> None:
        """Test converting list with single item."""
        # Arrange
        input_list: list[tuple[str, Any]] = [("Alice", None)]
        expected = "Alice"

        # Act
        result = list_to_title(input_list)

        # Assert
        assert result == expected

    def test_list_to_title_with_two_items_joins_with_ampersand(self) -> None:
        """Test converting list with two items."""
        # Arrange
        input_list: list[tuple[str, Any]] = [("Alice", None), ("Bob", None)]
        expected = "Alice & Bob"

        # Act
        result = list_to_title(input_list)

        # Assert
        assert result == expected

    def test_list_to_title_with_empty_list_returns_empty(self) -> None:
        """Test converting empty list."""
        # Arrange
        input_list: list = []
        expected = ""

        # Act
        result = list_to_title(input_list)

        # Assert
        assert result == expected

    def test_list_to_title_with_duplicate_items_removes_duplicates(self) -> None:
        """Test that duplicate items are removed."""
        # Arrange
        input_list: list[tuple[str, Any]] = [
            ("Alice", None), ("Bob", None), ("Alice", None), ("Charlie", None)
        ]
        expected = "Alice, Bob & Charlie"

        # Act
        result = list_to_title(input_list)

        # Assert
        assert result == expected

    def test_list_to_title_with_nested_data_uses_first_element(self) -> None:
        """Test that only first element of tuples is used."""
        # Arrange
        input_list = [("Alice", 1), ("Bob", 2), ("Charlie", 3)]
        expected = "Alice, Bob & Charlie"

        # Act
        result = list_to_title(input_list)

        # Assert
        assert result == expected

    def test_list_to_title_preserves_order(self) -> None:
        """Test that order of first occurrences is preserved."""
        # Arrange
        input_list: list[tuple[str, Any]] = [
            ("Charlie", None), ("Alice", None), ("Bob", None)
        ]
        expected = "Charlie, Alice & Bob"

        # Act
        result = list_to_title(input_list)

        # Assert
        assert result == expected
