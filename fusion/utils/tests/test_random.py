"""Unit tests for fusion.utils.random module."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from fusion.utils.random import (
    generate_exponential_random_variable,
    generate_uniform_random_variable,
    set_random_seed,
)


class TestSetRandomSeed:
    """Tests for set_random_seed function."""

    def test_set_random_seed_with_valid_seed_produces_deterministic_output(
        self,
    ) -> None:
        """Test that same seed produces same random sequence."""
        # Arrange & Act
        set_random_seed(42)
        result1 = generate_uniform_random_variable()
        set_random_seed(42)
        result2 = generate_uniform_random_variable()

        # Assert
        assert result1 == result2

    def test_set_random_seed_with_different_seeds_produces_different_output(
        self,
    ) -> None:
        """Test that different seeds produce different sequences."""
        # Arrange & Act
        set_random_seed(42)
        result1 = generate_uniform_random_variable()
        set_random_seed(123)
        result2 = generate_uniform_random_variable()

        # Assert
        assert result1 != result2

    def test_set_random_seed_with_zero_seed_succeeds(self) -> None:
        """Test setting seed to zero is valid."""
        # Act & Assert - should not raise
        set_random_seed(0)
        result = generate_uniform_random_variable()
        assert isinstance(result, (float, int))

    def test_set_random_seed_with_negative_seed_raises_value_error(self) -> None:
        """Test that negative seed raises ValueError."""
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            set_random_seed(-1)

        assert str(exc_info.value) == "Seed value must be non-negative"

    def test_set_random_seed_with_large_seed_succeeds(self) -> None:
        """Test setting large seed value."""
        # Arrange
        large_seed = 2**31 - 1

        # Act & Assert - should not raise
        set_random_seed(large_seed)
        result = generate_uniform_random_variable()
        assert isinstance(result, (float, int))

    @patch("fusion.utils.random.np.random.seed")
    def test_set_random_seed_calls_numpy_seed(self, mock_seed: Mock) -> None:
        """Test that set_random_seed calls numpy's seed function."""
        # Act
        set_random_seed(42)

        # Assert
        mock_seed.assert_called_once_with(42)


class TestGenerateUniformRandomVariable:
    """Tests for generate_uniform_random_variable function."""

    def test_generate_uniform_without_scale_returns_float_in_range(self) -> None:
        """Test generating uniform random variable without scale parameter."""
        # Arrange
        set_random_seed(42)

        # Act
        result = generate_uniform_random_variable()

        # Assert
        assert isinstance(result, (float, np.floating))
        assert 0 <= result <= 1

    def test_generate_uniform_with_scale_returns_int_in_range(self) -> None:
        """Test generating uniform random variable with scale parameter."""
        # Arrange
        set_random_seed(42)
        scale_param = 10.0

        # Act
        result = generate_uniform_random_variable(scale_param)

        # Assert
        assert isinstance(result, (int, np.integer))
        assert 0 <= result < scale_param

    def test_generate_uniform_with_scale_produces_varied_results(self) -> None:
        """Test that scaled uniform produces different values."""
        # Arrange
        set_random_seed(42)
        results = []

        # Act
        for _ in range(10):
            results.append(generate_uniform_random_variable(100.0))

        # Assert - should have some variation
        assert len(set(results)) > 1

    def test_generate_uniform_with_zero_scale_raises_value_error(self) -> None:
        """Test that zero scale parameter raises ValueError."""
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            generate_uniform_random_variable(0.0)

        assert str(exc_info.value) == "Scale parameter must be positive"

    def test_generate_uniform_with_negative_scale_raises_value_error(self) -> None:
        """Test that negative scale parameter raises ValueError."""
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            generate_uniform_random_variable(-5.0)

        assert str(exc_info.value) == "Scale parameter must be positive"

    def test_generate_uniform_without_scale_multiple_calls_vary(self) -> None:
        """Test that multiple calls without scale produce varied results."""
        # Arrange
        set_random_seed(42)
        results = []

        # Act
        for _ in range(10):
            results.append(generate_uniform_random_variable())

        # Assert
        assert len(set(results)) > 1

    def test_generate_uniform_with_small_scale_returns_limited_values(self) -> None:
        """Test that small scale parameter limits possible values."""
        # Arrange
        set_random_seed(42)
        scale_param = 2.0

        # Act
        results = [generate_uniform_random_variable(scale_param) for _ in range(10)]

        # Assert - with scale=2, can only get 0 or 1
        assert all(r in {0, 1} for r in results)

    @patch("fusion.utils.random.np.random.uniform")
    def test_generate_uniform_calls_numpy_uniform(self, mock_uniform: Mock) -> None:
        """Test that function calls numpy's uniform function."""
        # Arrange
        mock_uniform.return_value = 0.5

        # Act
        generate_uniform_random_variable()

        # Assert
        mock_uniform.assert_called_once_with(0, 1)


class TestGenerateExponentialRandomVariable:
    """Tests for generate_exponential_random_variable function."""

    def test_generate_exponential_with_valid_scale_returns_positive(self) -> None:
        """Test generating exponential random variable returns positive value."""
        # Arrange
        set_random_seed(42)
        scale_param = 2.5

        # Act
        result = generate_exponential_random_variable(scale_param)

        # Assert
        assert isinstance(result, float)
        assert result >= 0

    def test_generate_exponential_with_scale_produces_varied_results(self) -> None:
        """Test that exponential produces different values."""
        # Arrange
        set_random_seed(42)
        scale_param = 5.0
        results = []

        # Act
        for _ in range(10):
            results.append(generate_exponential_random_variable(scale_param))

        # Assert
        assert len(set(results)) > 1

    def test_generate_exponential_with_zero_scale_raises_value_error(self) -> None:
        """Test that zero scale parameter raises ValueError."""
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            generate_exponential_random_variable(0.0)

        assert str(exc_info.value) == "Scale parameter must be positive"

    def test_generate_exponential_with_negative_scale_raises_value_error(self) -> None:
        """Test that negative scale parameter raises ValueError."""
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            generate_exponential_random_variable(-1.0)

        assert str(exc_info.value) == "Scale parameter must be positive"

    def test_generate_exponential_mean_approximates_scale(self) -> None:
        """Test that exponential mean approximates 1/scale parameter."""
        # Arrange
        set_random_seed(42)
        scale_param = 2.0
        num_samples = 1000

        # Act
        samples = [generate_exponential_random_variable(scale_param) for _ in range(num_samples)]
        mean = np.mean(samples)

        # Assert - mean should be approximately 1/scale_param
        expected_mean = 1.0 / scale_param
        # Allow 20% tolerance due to randomness
        assert abs(mean - expected_mean) / expected_mean < 0.2

    def test_generate_exponential_with_small_scale_produces_larger_values(
        self,
    ) -> None:
        """Test that smaller scale parameter produces larger values on average."""
        # Arrange
        set_random_seed(42)
        small_scale = 0.5
        large_scale = 5.0
        num_samples = 100

        # Act
        small_scale_samples = [generate_exponential_random_variable(small_scale) for _ in range(num_samples)]
        set_random_seed(42)
        large_scale_samples = [generate_exponential_random_variable(large_scale) for _ in range(num_samples)]

        # Assert
        assert np.mean(small_scale_samples) > np.mean(large_scale_samples)

    @patch("fusion.utils.random.generate_uniform_random_variable")
    def test_generate_exponential_uses_inverse_transform(self, mock_uniform: Mock) -> None:
        """Test that exponential uses inverse transform method."""
        # Arrange
        mock_uniform.return_value = 0.5
        scale_param = 2.0

        # Act
        result = generate_exponential_random_variable(scale_param)

        # Assert
        mock_uniform.assert_called_once()
        # Verify inverse transform formula: -1/λ * ln(U)
        expected = (-1.0 / scale_param) * np.log(0.5)
        assert abs(result - expected) < 1e-10

    def test_generate_exponential_deterministic_with_seed(self) -> None:
        """Test that exponential is deterministic with same seed."""
        # Arrange & Act
        set_random_seed(123)
        result1 = generate_exponential_random_variable(3.0)
        set_random_seed(123)
        result2 = generate_exponential_random_variable(3.0)

        # Assert
        assert result1 == result2
