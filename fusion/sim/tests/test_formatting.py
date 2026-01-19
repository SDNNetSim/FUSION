"""Unit tests for fusion/sim/utils/formatting.py module."""

from fusion.sim.utils.formatting import int_to_string, list_to_title, snake_to_title


class TestSnakeToTitle:
    """Tests for snake_to_title function."""

    def test_snake_to_title_with_single_word_returns_capitalized(self) -> None:
        """Test that single word is capitalized correctly."""
        # Arrange
        snake_str = "hello"

        # Act
        result = snake_to_title(snake_str)

        # Assert
        assert result == "Hello"

    def test_snake_to_title_with_multiple_words_returns_title_case(self) -> None:
        """Test that snake_case string is converted to Title Case."""
        # Arrange
        snake_str = "hello_world_test"

        # Act
        result = snake_to_title(snake_str)

        # Assert
        assert result == "Hello World Test"

    def test_snake_to_title_with_empty_string_returns_empty(self) -> None:
        """Test that empty string returns empty string."""
        # Arrange
        snake_str = ""

        # Act
        result = snake_to_title(snake_str)

        # Assert
        assert result == ""

    def test_snake_to_title_with_already_capitalized_preserves_case(self) -> None:
        """Test that already capitalized words are handled correctly."""
        # Arrange
        snake_str = "HELLO_WORLD"

        # Act
        result = snake_to_title(snake_str)

        # Assert
        assert result == "Hello World"


class TestIntToString:
    """Tests for int_to_string function."""

    def test_int_to_string_with_small_number_returns_string(self) -> None:
        """Test that small numbers are converted correctly."""
        # Arrange
        number = 123

        # Act
        result = int_to_string(number)

        # Assert
        assert result == "123"

    def test_int_to_string_with_thousands_adds_comma(self) -> None:
        """Test that thousands separator is added correctly."""
        # Arrange
        number = 1234567

        # Act
        result = int_to_string(number)

        # Assert
        assert result == "1,234,567"

    def test_int_to_string_with_zero_returns_zero_string(self) -> None:
        """Test that zero is converted correctly."""
        # Arrange
        number = 0

        # Act
        result = int_to_string(number)

        # Assert
        assert result == "0"

    def test_int_to_string_with_millions_adds_multiple_commas(self) -> None:
        """Test that millions are formatted with multiple commas."""
        # Arrange
        number = 1000000

        # Act
        result = int_to_string(number)

        # Assert
        assert result == "1,000,000"


class TestListToTitle:
    """Tests for list_to_title function."""

    def test_list_to_title_with_single_item_returns_item(self) -> None:
        """Test that single item list returns just the item."""
        # Arrange
        input_list = [("Alice", None)]

        # Act
        result = list_to_title(input_list)

        # Assert
        assert result == "Alice"

    def test_list_to_title_with_multiple_items_joins_with_ampersand(self) -> None:
        """Test that multiple items are joined with commas and ampersand."""
        # Arrange
        input_list = [("Alice", None), ("Bob", None), ("Charlie", None)]

        # Act
        result = list_to_title(input_list)

        # Assert
        assert result == "Alice, Bob & Charlie"

    def test_list_to_title_with_empty_list_returns_empty_string(self) -> None:
        """Test that empty list returns empty string."""
        # Arrange
        input_list: list[tuple[str, str]] = []

        # Act
        result = list_to_title(input_list)

        # Assert
        assert result == ""

    def test_list_to_title_with_two_items_uses_ampersand_only(self) -> None:
        """Test that two items are joined with ampersand only."""
        # Arrange
        input_list = [("Alice", None), ("Bob", None)]

        # Act
        result = list_to_title(input_list)

        # Assert
        assert result == "Alice & Bob"

    def test_list_to_title_with_duplicate_items_removes_duplicates(self) -> None:
        """Test that duplicate items are removed."""
        # Arrange
        input_list = [
            ("Alice", None),
            ("Bob", None),
            ("Alice", None),
            ("Charlie", None),
        ]

        # Act
        result = list_to_title(input_list)

        # Assert
        assert result == "Alice, Bob & Charlie"
