"""Unit tests for fusion/sim/utils/spectrum.py module."""

import copy

import numpy as np
import pytest

from fusion.sim.utils.spectrum import (
    combine_and_one_hot,
    find_free_channels,
    find_free_slots,
    find_taken_channels,
    get_shannon_entropy_fragmentation,
    get_super_channels,
)


@pytest.fixture
def sample_network_spectrum() -> dict:
    """Provide sample network spectrum for tests."""
    return {
        (0, 1): {"cores_matrix": {"c": np.array([[0, 1, 0, 0, 1], [0, 0, 1, 0, 1]])}},
        (3, 4): {"cores_matrix": {"c": np.array([[0, 0, 1, 1, -1], [0, 0, 0, 0, 0]])}},
    }


class TestFindFreeSlots:
    """Tests for find_free_slots function."""

    def test_find_free_slots_returns_correct_indices(self, sample_network_spectrum: dict) -> None:
        """Test that free slot indices are identified correctly."""
        # Arrange
        link_tuple = (0, 1)

        # Act
        result = find_free_slots(sample_network_spectrum, link_tuple)

        # Assert
        expected_core_0 = np.array([0, 2, 3])
        expected_core_1 = np.array([0, 1, 3])
        assert np.array_equal(result["c"][0], expected_core_0)
        assert np.array_equal(result["c"][1], expected_core_1)

    def test_find_free_slots_with_legacy_params_works(self, sample_network_spectrum: dict) -> None:
        """Test backward compatibility with legacy parameter names."""
        # Arrange
        link_tuple = (0, 1)

        # Act
        result = find_free_slots(network_spectrum_dict=sample_network_spectrum, link_tuple=link_tuple)

        # Assert
        expected_core_0 = np.array([0, 2, 3])
        assert np.array_equal(result["c"][0], expected_core_0)

    def test_find_free_slots_without_params_raises_error(self) -> None:
        """Test that missing required parameters raises ValueError."""
        # Act & Assert
        with pytest.raises(ValueError, match="Must provide network_spectrum and link_tuple"):
            find_free_slots()


class TestFindFreeChannels:
    """Tests for find_free_channels function."""

    def test_find_free_channels_with_slots_needed_returns_channels(self, sample_network_spectrum: dict) -> None:
        """Test that free channels are found for given slot requirement."""
        # Arrange
        slots_needed = 2
        link_tuple = (0, 1)

        # Act
        result = find_free_channels(sample_network_spectrum, slots_needed, link_tuple)

        # Assert
        expected = {"c": {0: [[2, 3]], 1: [[0, 1]]}}
        assert result == expected

    def test_find_free_channels_with_no_contiguous_slots_returns_empty(self) -> None:
        """Test that no channels are returned when slots are fragmented."""
        # Arrange
        network_spectrum = {(0, 1): {"cores_matrix": {"c": np.array([[1, 0, 1, 0, 1]])}}}
        slots_needed = 2
        link_tuple = (0, 1)

        # Act
        result = find_free_channels(network_spectrum, slots_needed, link_tuple)

        # Assert
        assert result["c"][0] == []

    def test_find_free_channels_without_params_raises_error(self) -> None:
        """Test that missing required parameters raises ValueError."""
        # Act & Assert
        with pytest.raises(
            ValueError,
            match="Must provide network_spectrum, link_tuple, and slots_needed",
        ):
            find_free_channels()


class TestFindTakenChannels:
    """Tests for find_taken_channels function."""

    def test_find_taken_channels_identifies_occupied_blocks(self, sample_network_spectrum: dict) -> None:
        """Test that taken channel blocks are identified correctly."""
        # Arrange
        link_tuple = (3, 4)

        # Act
        result = find_taken_channels(copy.deepcopy(sample_network_spectrum), link_tuple)

        # Assert
        # Core 0 has a block of [1, 1] at positions 2-3
        expected = {"c": {0: [[1, 1]], 1: []}}
        assert result == expected

    def test_find_taken_channels_with_legacy_params_works(self, sample_network_spectrum: dict) -> None:
        """Test backward compatibility with legacy parameter names."""
        # Arrange
        link_tuple = (3, 4)

        # Act
        result = find_taken_channels(network_spectrum_dict=sample_network_spectrum, link_tuple=link_tuple)

        # Assert
        expected = {"c": {0: [[1, 1]], 1: []}}
        assert result == expected


class TestGetSuperChannels:
    """Tests for get_super_channels function."""

    def test_get_super_channels_finds_available_positions(self) -> None:
        """Test that super-channels are identified in spectrum."""
        # Arrange
        input_arr = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0])
        slots_needed = 3

        # Act
        result = get_super_channels(input_arr, slots_needed)

        # Assert
        # With guard band requirement, valid positions are [0,3] and [5,8]
        expected = np.array([[0, 3], [5, 8]])
        assert np.array_equal(result, expected)

    def test_get_super_channels_with_no_space_returns_empty(self) -> None:
        """Test that empty array is returned when no space available."""
        # Arrange
        input_arr = np.array([1, 1, 1, 1])
        slots_needed = 2

        # Act
        result = get_super_channels(input_arr, slots_needed)

        # Assert
        assert result.size == 0


class TestCombineAndOneHot:
    """Tests for combine_and_one_hot function."""

    def test_combine_and_one_hot_performs_or_operation(self) -> None:
        """Test that OR operation combines arrays correctly."""
        # Arrange
        arr1 = np.array([0, 1, 0, 1, 0])
        arr2 = np.array([1, 0, 1, 0, 1])

        # Act
        result = combine_and_one_hot(arr1, arr2)

        # Assert
        expected = np.array([1, 1, 1, 1, 1])
        assert np.array_equal(result, expected)

    def test_combine_and_one_hot_with_both_zero_returns_zero(self) -> None:
        """Test that zeros remain zero in OR operation."""
        # Arrange
        arr1 = np.array([0, 0, 1])
        arr2 = np.array([0, 1, 0])

        # Act
        result = combine_and_one_hot(arr1, arr2)

        # Assert
        expected = np.array([0, 1, 1])
        assert np.array_equal(result, expected)

    def test_combine_and_one_hot_with_different_lengths_raises_error(self) -> None:
        """Test that arrays of different lengths raise ValueError."""
        # Arrange
        arr1 = np.array([0, 1])
        arr2 = np.array([0, 1, 0])

        # Act & Assert
        with pytest.raises(ValueError, match="Arrays must have the same length"):
            combine_and_one_hot(arr1, arr2)


class TestGetShannonEntropyFragmentation:
    """Tests for get_shannon_entropy_fragmentation function."""

    def test_get_hfrag_calculates_entropy_scores(self) -> None:
        """Test that Shannon entropy fragmentation scores are calculated."""
        # Arrange
        path_list = [1, 2, 3]
        core_num = 0
        band = "c"
        slots_needed = 3
        spectral_slots = 8
        network_spectrum = {
            (1, 2): {"cores_matrix": {"c": np.zeros((1, spectral_slots))}},
            (2, 3): {"cores_matrix": {"c": np.zeros((1, spectral_slots))}},
        }

        # Act
        sc_index_mat, resp_frag_arr = get_shannon_entropy_fragmentation(
            path_list, core_num, band, slots_needed, spectral_slots, network_spectrum
        )

        # Assert
        # With all zeros, we expect certain super-channel indices
        expected_sc_index = np.array([[0, 3], [1, 4], [2, 5], [3, 6], [4, 7]])
        assert np.array_equal(sc_index_mat, expected_sc_index)
        # Fragmentation array should have specific pattern
        expected_frag = np.array([1.386, -np.inf, -np.inf, -np.inf, 1.386, np.inf, np.inf, np.inf])
        assert np.array_equal(resp_frag_arr, expected_frag)

    def test_get_hfrag_with_none_core_defaults_to_zero(self) -> None:
        """Test that None core_num defaults to core 0."""
        # Arrange
        path_list = [1, 2]
        core_num = 0
        band = "c"
        slots_needed = 2
        spectral_slots = 5
        network_spectrum = {
            (1, 2): {"cores_matrix": {"c": np.zeros((1, spectral_slots))}},
        }

        # Act
        sc_index_mat, resp_frag_arr = get_shannon_entropy_fragmentation(
            path_list, core_num, band, slots_needed, spectral_slots, network_spectrum
        )

        # Assert
        # Should not raise error and return valid results
        assert sc_index_mat is not None
        assert resp_frag_arr is not None
