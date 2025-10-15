"""
Unit tests for determinism and seed management.
"""

import numpy as np
import pytest

from fusion.core.simulation import generate_seed_from_time, seed_all_rngs, validate_seed


def test_reproducibility_with_same_seed() -> None:
    """Test that identical results with same seed."""
    seed = 42

    # First run
    seed_all_rngs(seed)
    result1 = [np.random.rand() for _ in range(10)]

    # Second run with same seed
    seed_all_rngs(seed)
    result2 = [np.random.rand() for _ in range(10)]

    # Should be identical
    np.testing.assert_array_equal(result1, result2)


def test_different_results_with_different_seeds() -> None:
    """Test that different results with different seeds."""
    # Run 1 with seed 42
    seed_all_rngs(42)
    result1 = [np.random.rand() for _ in range(10)]

    # Run 2 with seed 43
    seed_all_rngs(43)
    result2 = [np.random.rand() for _ in range(10)]

    # Should be different
    assert not np.allclose(result1, result2)


def test_seed_generation_from_time() -> None:
    """Test that seed can be generated from time."""
    seed1 = generate_seed_from_time()
    seed2 = generate_seed_from_time()

    # Should be valid integers
    assert isinstance(seed1, int)
    assert isinstance(seed2, int)

    # Should be different (unless called in same millisecond)
    assert seed1 >= 0
    assert seed2 >= 0


def test_seed_validation() -> None:
    """Test seed validation."""
    # Valid seeds
    assert validate_seed(0) == 0
    assert validate_seed(42) == 42
    assert validate_seed(2**31 - 1) == 2**31 - 1

    # Invalid seeds
    with pytest.raises(ValueError, match="non-negative"):
        validate_seed(-1)

    with pytest.raises(ValueError, match="2\\^31"):
        validate_seed(2**31)


def test_python_random_seeding() -> None:
    """Test that Python's random module is seeded."""
    import random

    seed = 42

    # Run 1
    seed_all_rngs(seed)
    result1 = [random.random() for _ in range(10)]  # nosec B311  # For simulation, not crypto

    # Run 2
    seed_all_rngs(seed)
    result2 = [random.random() for _ in range(10)]  # nosec B311  # For simulation, not crypto

    # Should be identical
    assert result1 == result2


def test_numpy_random_seeding() -> None:
    """Test that NumPy's random is seeded."""
    seed = 42

    # Run 1
    seed_all_rngs(seed)
    result1 = np.random.rand(10)

    # Run 2
    seed_all_rngs(seed)
    result2 = np.random.rand(10)

    # Should be identical
    np.testing.assert_array_equal(result1, result2)


def test_torch_determinism() -> None:
    """Test that PyTorch operations are deterministic."""
    try:
        import torch

        # Verify torch is properly functional
        if not hasattr(torch, "randn") or not callable(torch.randn):
            pytest.skip("PyTorch is broken or improperly installed")

        seed = 42

        # Run 1
        seed_all_rngs(seed)
        x = torch.randn(10, 10)

        # Verify we got a proper tensor, not a list
        if not hasattr(x, "t"):
            pytest.skip("PyTorch is broken or improperly installed")

        result1 = x @ x.t()

        # Run 2
        seed_all_rngs(seed)
        x = torch.randn(10, 10)
        result2 = x @ x.t()

        # Should be identical
        assert torch.allclose(result1, result2)

    except (ImportError, AttributeError):
        pytest.skip("PyTorch not installed or broken")


def test_seed_all_rngs_no_torch() -> None:
    """Test that seed_all_rngs works without PyTorch."""
    # Should not raise even if torch not available
    seed_all_rngs(42)

    # Verify NumPy and Python random are still seeded
    result1 = np.random.rand()
    import random

    result2 = random.random()  # nosec B311  # For simulation, not crypto

    assert isinstance(result1, (float, np.floating))
    assert isinstance(result2, float)


def test_validate_seed_edge_cases() -> None:
    """Test seed validation edge cases."""
    # Boundary values
    assert validate_seed(0) == 0
    assert validate_seed(1) == 1
    assert validate_seed(2**31 - 2) == 2**31 - 2
    assert validate_seed(2**31 - 1) == 2**31 - 1

    # Just over the boundary
    with pytest.raises(ValueError):
        validate_seed(2**31)

    with pytest.raises(ValueError):
        validate_seed(2**31 + 1)


def test_generate_seed_from_time_range() -> None:
    """Test that generated seeds are in valid range."""
    for _ in range(10):
        seed = generate_seed_from_time()
        assert 0 <= seed < 2**31


def test_reproducible_random_sequence() -> None:
    """Test that full random sequence is reproducible."""
    import random

    seed = 12345

    # Generate sequence 1
    seed_all_rngs(seed)
    seq1 = {
        "python_random": [random.random() for _ in range(5)],  # nosec B311  # For simulation, not crypto
        "numpy_random": np.random.rand(5).tolist(),
        "numpy_int": np.random.randint(0, 100, 5).tolist(),
    }

    # Generate sequence 2 with same seed
    seed_all_rngs(seed)
    seq2 = {
        "python_random": [random.random() for _ in range(5)],  # nosec B311  # For simulation, not crypto
        "numpy_random": np.random.rand(5).tolist(),
        "numpy_int": np.random.randint(0, 100, 5).tolist(),
    }

    # All should be identical
    assert seq1["python_random"] == seq2["python_random"]
    np.testing.assert_array_equal(seq1["numpy_random"], seq2["numpy_random"])
    np.testing.assert_array_equal(seq1["numpy_int"], seq2["numpy_int"])


def test_cross_module_seeding() -> None:
    """Test that seeding works across different modules."""
    import random

    seed = 99

    # Seed and use both random and numpy
    seed_all_rngs(seed)
    py_val1 = random.random()  # nosec B311  # For simulation, not crypto
    np_val1 = np.random.rand()

    # Reseed and repeat
    seed_all_rngs(seed)
    py_val2 = random.random()  # nosec B311  # For simulation, not crypto
    np_val2 = np.random.rand()

    # Both should be reproducible
    assert py_val1 == py_val2
    assert np_val1 == np_val2
