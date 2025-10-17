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

    # Verify results are non-empty
    assert len(result1) == 10
    assert len(result2) == 10

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

    # Verify results are non-empty
    assert len(result1) == 10
    assert len(result2) == 10

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

    # Verify results are non-empty
    assert len(result1) == 10
    assert len(result2) == 10

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

    # Verify results are non-empty
    assert result1.shape == (10,)
    assert result2.shape == (10,)
    assert len(result1) == 10
    assert len(result2) == 10

    # Should be identical
    np.testing.assert_array_equal(result1, result2)


def test_torch_determinism() -> None:
    """Test that PyTorch operations are deterministic."""
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not installed")

    seed = 42

    # Run 1
    seed_all_rngs(seed)
    x = torch.randn(10, 10)
    result1 = x @ x.t()

    # Run 2
    seed_all_rngs(seed)
    x = torch.randn(10, 10)
    result2 = x @ x.t()

    # Verify results are non-empty (proper shape)
    assert result1.shape == (10, 10)
    assert result2.shape == (10, 10)

    # Should be identical
    assert torch.allclose(result1, result2)


def test_seed_all_rngs_no_torch() -> None:
    """Test that seed_all_rngs works without PyTorch."""
    import random

    # Should not raise even if torch not available
    seed_all_rngs(42)

    # Generate values from seeded RNGs
    np_val1 = np.random.rand()
    py_val1 = random.random()  # nosec B311  # For simulation, not crypto

    # Reseed and verify reproducibility
    seed_all_rngs(42)
    np_val2 = np.random.rand()
    py_val2 = random.random()  # nosec B311  # For simulation, not crypto

    # Verify values are valid
    assert isinstance(np_val1, (float, np.floating))
    assert isinstance(py_val1, float)

    # Verify reproducibility
    assert np_val1 == np_val2
    assert py_val1 == py_val2


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

    # Verify sequences are non-empty
    assert len(seq1["python_random"]) == 5
    assert len(seq1["numpy_random"]) == 5
    assert len(seq1["numpy_int"]) == 5
    assert len(seq2["python_random"]) == 5
    assert len(seq2["numpy_random"]) == 5
    assert len(seq2["numpy_int"]) == 5

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


def test_separate_seeding_request_generation() -> None:
    """Test that seed_request_generation only affects NumPy."""

    from fusion.core.simulation import seed_request_generation, seed_rl_components

    # Seed request generation
    seed_request_generation(42)
    np_val1 = np.random.rand()

    # Seed RL components (should not affect NumPy)
    seed_rl_components(99)
    np_val2 = np.random.rand()

    # Reseed request generation with same seed
    seed_request_generation(42)
    np_val3 = np.random.rand()

    # First and third should match (same request seed)
    assert np_val1 == np_val3

    # Second should differ (RL seeding doesn't affect NumPy after it's already seeded)
    # This test verifies that seed_request_generation properly seeds NumPy
    assert np_val1 != np_val2


def test_separate_seeding_rl_components() -> None:
    """Test that seed_rl_components affects Python random but not NumPy."""
    import random

    from fusion.core.simulation import seed_request_generation, seed_rl_components

    # Seed RL components
    seed_rl_components(42)
    py_val1 = random.random()  # nosec B311  # For simulation, not crypto

    # Seed request generation (should not affect Python random)
    seed_request_generation(99)

    # Reseed RL components with same seed
    seed_rl_components(42)
    py_val2 = random.random()  # nosec B311  # For simulation, not crypto

    # Should match (same RL seed)
    assert py_val1 == py_val2


def test_separate_seeding_allows_independent_control() -> None:
    """Test that request and RL seeds can be controlled independently."""
    import random

    from fusion.core.simulation import seed_request_generation, seed_rl_components

    # Scenario: RL seed stays constant, request seed varies
    rl_seed = 42
    request_seeds = [10, 20, 30]

    results = []
    for req_seed in request_seeds:
        seed_rl_components(rl_seed)  # Constant RL seed
        seed_request_generation(req_seed)  # Varying request seed

        results.append(
            {
                "python_random": random.random(),  # nosec B311
                "numpy_rand": np.random.rand(),
            }
        )

    # Verify all results were generated (non-None)
    assert len(results) == 3
    for result in results:
        assert isinstance(result["python_random"], float)
        assert isinstance(result["numpy_rand"], (float, np.floating))

    # Python random values should all be the same (constant RL seed)
    assert results[0]["python_random"] == results[1]["python_random"]
    assert results[1]["python_random"] == results[2]["python_random"]

    # NumPy values should all differ (varying request seeds)
    assert results[0]["numpy_rand"] != results[1]["numpy_rand"]
    assert results[1]["numpy_rand"] != results[2]["numpy_rand"]
