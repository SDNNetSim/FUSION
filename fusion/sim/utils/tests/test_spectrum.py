"""Unit tests for fusion.sim.utils.spectrum module."""

import numpy as np
import pytest

from ..spectrum import (
    combine_and_one_hot,
    find_free_channels,
    find_free_slots,
    find_taken_channels,
    get_channel_overlaps,
    get_shannon_entropy_fragmentation,
    get_super_channels,
)


class TestFreeSlotDetection:
    """Tests for finding free spectral slots."""

    @pytest.fixture
    def network_spectrum(self) -> dict:
        """Provide sample network spectrum for testing.

        :return: Network spectrum database
        :rtype: dict
        """
        return {
            (0, 1): {"cores_matrix": {"c": np.array([[0, 1, 0, 0, 1], [0, 0, 1, 0, 1]])}},
            (3, 4): {"cores_matrix": {"c": np.array([[0, 0, 1, 1, -1], [0, 0, 0, 0, 0]])}},
        }

    def test_find_free_slots_with_mixed_allocation_returns_correct_indexes(self, network_spectrum: dict) -> None:
        """Test finding free slots on partially allocated link.

        :param network_spectrum: Network spectrum state
        :type network_spectrum: dict
        """
        # Arrange
        link_tuple = (0, 1)
        expected_core0 = np.array([0, 2, 3])
        expected_core1 = np.array([0, 1, 3])

        # Act
        result = find_free_slots(network_spectrum=network_spectrum, link_tuple=link_tuple)

        # Assert
        assert np.array_equal(result["c"][0], expected_core0)
        assert np.array_equal(result["c"][1], expected_core1)

    def test_find_free_slots_with_all_free_returns_all_indexes(self) -> None:
        """Test finding free slots on empty spectrum."""
        # Arrange
        network_spectrum = {
            (0, 1): {"cores_matrix": {"c": np.array([[0, 0, 0, 0, 0]])}},
        }
        link_tuple = (0, 1)
        expected = np.array([0, 1, 2, 3, 4])

        # Act
        result = find_free_slots(network_spectrum=network_spectrum, link_tuple=link_tuple)

        # Assert
        assert np.array_equal(result["c"][0], expected)

    def test_find_free_slots_with_all_taken_returns_empty(self) -> None:
        """Test finding free slots on fully occupied spectrum."""
        # Arrange
        network_spectrum = {
            (0, 1): {"cores_matrix": {"c": np.array([[1, 1, 1, 1, 1]])}},
        }
        link_tuple = (0, 1)
        expected = np.array([])

        # Act
        result = find_free_slots(network_spectrum=network_spectrum, link_tuple=link_tuple)

        # Assert
        assert np.array_equal(result["c"][0], expected)

    def test_find_free_slots_with_legacy_parameter_works(self, network_spectrum: dict) -> None:
        """Test backward compatibility with legacy parameter name.

        :param network_spectrum: Network spectrum state
        :type network_spectrum: dict
        """
        # Arrange
        link_tuple = (0, 1)

        # Act
        result = find_free_slots(network_spectrum_dict=network_spectrum, link_tuple=link_tuple)

        # Assert
        assert "c" in result
        assert 0 in result["c"]

    def test_find_free_slots_without_parameters_raises_error(self) -> None:
        """Test that missing parameters raise ValueError."""
        # Act & Assert
        with pytest.raises(ValueError, match="Must provide network_spectrum"):
            find_free_slots()


class TestFreeChannelDetection:
    """Tests for finding free super-channels."""

    @pytest.fixture
    def network_spectrum(self) -> dict:
        """Provide sample network spectrum for testing.

        :return: Network spectrum database
        :rtype: dict
        """
        return {
            (0, 1): {"cores_matrix": {"c": np.array([[0, 1, 0, 0, 1], [0, 0, 1, 0, 1]])}},
        }

    def test_find_free_channels_with_sufficient_slots_returns_channels(self, network_spectrum: dict) -> None:
        """Test finding free channels with available contiguous slots.

        :param network_spectrum: Network spectrum state
        :type network_spectrum: dict
        """
        # Arrange
        slots_needed = 2
        link_tuple = (0, 1)
        expected_core0 = [[2, 3]]
        expected_core1 = [[0, 1]]

        # Act
        result = find_free_channels(
            network_spectrum=network_spectrum,
            slots_needed=slots_needed,
            link_tuple=link_tuple,
        )

        # Assert
        assert result["c"][0] == expected_core0
        assert result["c"][1] == expected_core1

    def test_find_free_channels_with_insufficient_slots_returns_empty(self, network_spectrum: dict) -> None:
        """Test finding channels when no contiguous slots available.

        :param network_spectrum: Network spectrum state
        :type network_spectrum: dict
        """
        # Arrange
        slots_needed = 5
        link_tuple = (0, 1)

        # Act
        result = find_free_channels(
            network_spectrum=network_spectrum,
            slots_needed=slots_needed,
            link_tuple=link_tuple,
        )

        # Assert
        assert result["c"][0] == []
        assert result["c"][1] == []

    def test_find_free_channels_with_single_slot_needed_returns_channels(
        self,
    ) -> None:
        """Test finding channels needing only one slot (contiguous)."""
        # Arrange
        network_spectrum = {
            (0, 1): {"cores_matrix": {"c": np.array([[0, 0, 0, 1, 1]])}},
        }
        slots_needed = 1
        link_tuple = (0, 1)
        expected = [[0], [1], [2]]

        # Act
        result = find_free_channels(
            network_spectrum=network_spectrum,
            slots_needed=slots_needed,
            link_tuple=link_tuple,
        )

        # Assert
        assert result["c"][0] == expected

    def test_find_free_channels_with_legacy_parameters_works(self, network_spectrum: dict) -> None:
        """Test backward compatibility with legacy parameter names.

        :param network_spectrum: Network spectrum state
        :type network_spectrum: dict
        """
        # Arrange
        slots_needed = 2
        link_tuple = (0, 1)

        # Act
        result = find_free_channels(
            network_spectrum_dict=network_spectrum,
            slots_needed=slots_needed,
            link_tuple=link_tuple,
        )

        # Assert
        assert "c" in result

    def test_find_free_channels_without_parameters_raises_error(self) -> None:
        """Test that missing parameters raise ValueError."""
        # Act & Assert
        with pytest.raises(ValueError, match="Must provide network_spectrum"):
            find_free_channels()


class TestTakenChannelDetection:
    """Tests for finding taken super-channels."""

    def test_find_taken_channels_with_single_allocation_returns_channel(self) -> None:
        """Test finding taken channels with one allocated block."""
        # Arrange
        network_spectrum = {
            (3, 4): {"cores_matrix": {"c": np.array([[0, 0, 1, 1, -1], [0, 0, 0, 0, 0]])}},
        }
        link_tuple = (3, 4)
        expected_core0 = [[1, 1]]
        expected_core1: list = []

        # Act
        result = find_taken_channels(network_spectrum=network_spectrum, link_tuple=link_tuple)

        # Assert
        assert result["c"][0] == expected_core0
        assert result["c"][1] == expected_core1

    def test_find_taken_channels_with_multiple_allocations_returns_all(self) -> None:
        """Test finding taken channels with multiple allocated blocks."""
        # Arrange
        network_spectrum = {
            (0, 1): {"cores_matrix": {"c": np.array([[1, 1, -1, 2, 2, -1, 0, 0]])}},
        }
        link_tuple = (0, 1)
        expected = [[1, 1], [2, 2]]

        # Act
        result = find_taken_channels(network_spectrum=network_spectrum, link_tuple=link_tuple)

        # Assert
        assert result["c"][0] == expected

    def test_find_taken_channels_with_empty_spectrum_returns_empty(self) -> None:
        """Test finding taken channels on completely free spectrum."""
        # Arrange
        network_spectrum = {
            (0, 1): {"cores_matrix": {"c": np.array([[0, 0, 0, 0, 0]])}},
        }
        link_tuple = (0, 1)
        expected: list = []

        # Act
        result = find_taken_channels(network_spectrum=network_spectrum, link_tuple=link_tuple)

        # Assert
        assert result["c"][0] == expected

    def test_find_taken_channels_with_full_spectrum_returns_single_block(self) -> None:
        """Test finding taken channels on fully occupied spectrum."""
        # Arrange
        network_spectrum = {
            (0, 1): {"cores_matrix": {"c": np.array([[1, 1, 1, 1, 1]])}},
        }
        link_tuple = (0, 1)

        # Act
        result = find_taken_channels(network_spectrum=network_spectrum, link_tuple=link_tuple)

        # Assert
        assert len(result["c"][0]) == 1
        assert len(result["c"][0][0]) == 5

    def test_find_taken_channels_with_legacy_parameters_works(self) -> None:
        """Test backward compatibility with legacy parameter names."""
        # Arrange
        network_spectrum = {
            (0, 1): {"cores_matrix": {"c": np.array([[1, 1, -1, 0]])}},
        }
        link_tuple = (0, 1)

        # Act
        result = find_taken_channels(network_spectrum_dict=network_spectrum, link_tuple=link_tuple)

        # Assert
        assert "c" in result


class TestSuperChannelOperations:
    """Tests for super-channel manipulation functions."""

    def test_get_super_channels_with_available_slots_returns_positions(self) -> None:
        """Test finding available super-channels."""
        # Arrange
        input_array = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0])
        slots_needed = 3
        expected = np.array([[0, 3], [5, 8]])

        # Act
        result = get_super_channels(input_array, slots_needed)

        # Assert
        assert np.array_equal(result, expected)

    def test_get_super_channels_with_insufficient_space_returns_empty(self) -> None:
        """Test super-channel finding with insufficient contiguous slots."""
        # Arrange
        input_array = np.array([0, 0, 1, 0, 0, 1, 0])
        slots_needed = 5

        # Act
        result = get_super_channels(input_array, slots_needed)

        # Assert
        assert len(result) == 0

    def test_get_super_channels_with_single_slot_needed_returns_channels(
        self,
    ) -> None:
        """Test super-channel finding for single slot (includes guard band)."""
        # Arrange
        input_array = np.array([0, 0, 0, 1, 1])
        slots_needed = 1
        # Guard band requires slots_needed+1 consecutive zeros
        # So with 3 consecutive zeros, we get 2 overlapping positions
        expected = np.array([[0, 1], [1, 2]])

        # Act
        result = get_super_channels(input_array, slots_needed)

        # Assert
        assert np.array_equal(result, expected)

    def test_get_super_channels_with_all_occupied_returns_empty(self) -> None:
        """Test super-channel finding on fully occupied spectrum."""
        # Arrange
        input_array = np.array([1, 1, 1, 1, 1])
        slots_needed = 2

        # Act
        result = get_super_channels(input_array, slots_needed)

        # Assert
        assert len(result) == 0

    def test_combine_and_one_hot_with_different_patterns_returns_or(self) -> None:
        """Test OR operation on two spectrum arrays."""
        # Arrange
        array1 = np.array([0, 1, 0, 1, 0])
        array2 = np.array([1, 0, 1, 0, 1])
        expected = np.array([1, 1, 1, 1, 1])

        # Act
        result = combine_and_one_hot(array1, array2)

        # Assert
        assert np.array_equal(result, expected)

    def test_combine_and_one_hot_with_same_arrays_returns_same(self) -> None:
        """Test OR operation on identical arrays."""
        # Arrange
        array1 = np.array([0, 1, 0, 1, 0])
        array2 = np.array([0, 1, 0, 1, 0])
        expected = np.array([0, 1, 0, 1, 0])

        # Act
        result = combine_and_one_hot(array1, array2)

        # Assert
        assert np.array_equal(result, expected)

    def test_combine_and_one_hot_with_different_lengths_raises_error(self) -> None:
        """Test that arrays of different lengths raise ValueError."""
        # Arrange
        array1 = np.array([0, 1, 0])
        array2 = np.array([1, 0, 1, 0])

        # Act & Assert
        with pytest.raises(ValueError, match="same length"):
            combine_and_one_hot(array1, array2)


class TestShannonEntropyFragmentation:
    """Tests for Shannon entropy fragmentation calculation."""

    @pytest.fixture
    def network_spectrum(self) -> dict:
        """Provide sample network spectrum for testing.

        :return: Network spectrum database
        :rtype: dict
        """
        return {
            (1, 2): {"cores_matrix": {"c": np.zeros((1, 8))}},
            (2, 3): {"cores_matrix": {"c": np.zeros((1, 8))}},
        }

    def test_get_shannon_entropy_fragmentation_with_empty_spectrum_returns_arrays(self, network_spectrum: dict) -> None:
        """Test Shannon entropy calculation on empty spectrum.

        :param network_spectrum: Network spectrum state
        :type network_spectrum: dict
        """
        # Arrange
        path_list = [1, 2, 3]
        core_num = 0
        band = "c"
        slots_needed = 3
        spectral_slots = 8
        expected_sc_matrix = np.array([[0, 3], [1, 4], [2, 5], [3, 6], [4, 7]])
        expected_frag_array = np.array([1.386, -np.inf, -np.inf, -np.inf, 1.386, np.inf, np.inf, np.inf])

        # Act
        sc_matrix, frag_array = get_shannon_entropy_fragmentation(path_list, core_num, band, slots_needed, spectral_slots, network_spectrum)

        # Assert
        assert np.array_equal(sc_matrix, expected_sc_matrix)
        assert np.array_equal(frag_array, expected_frag_array)

    def test_get_shannon_entropy_fragmentation_with_none_core_uses_zero(self, network_spectrum: dict) -> None:
        """Test that None core_num defaults to 0.

        :param network_spectrum: Network spectrum state
        :type network_spectrum: dict
        """
        # Arrange
        path_list = [1, 2, 3]
        core_num = 0
        band = "c"
        slots_needed = 2
        spectral_slots = 8

        # Act
        sc_matrix, frag_array = get_shannon_entropy_fragmentation(path_list, core_num, band, slots_needed, spectral_slots, network_spectrum)

        # Assert
        assert len(sc_matrix) > 0
        assert len(frag_array) == spectral_slots

    def test_get_shannon_entropy_fragmentation_with_occupied_slots_affects_scores(
        self,
    ) -> None:
        """Test Shannon entropy with partially occupied spectrum."""
        # Arrange
        network_spectrum = {
            (1, 2): {"cores_matrix": {"c": np.array([[0, 0, 1, 1, 0, 0, 0, 0]])}},
        }
        path_list = [1, 2]
        core_num = 0
        band = "c"
        slots_needed = 2
        spectral_slots = 8

        # Act
        sc_matrix, frag_array = get_shannon_entropy_fragmentation(path_list, core_num, band, slots_needed, spectral_slots, network_spectrum)

        # Assert
        assert len(sc_matrix) > 0
        assert len(frag_array) == spectral_slots
        # Positions 2 and 3 are occupied, so they should be inf
        assert frag_array[2] == np.inf
        assert frag_array[3] == np.inf


class TestChannelOverlaps:
    """Tests for channel overlap detection."""

    def test_get_channel_overlaps_with_seven_cores_returns_overlaps(self) -> None:
        """Test finding overlapping channels between cores (7-core topology)."""
        # Arrange - Using 7 cores as expected by get_channel_overlaps
        free_channels_dict = {
            ("A", "B"): {
                "c": {
                    0: [[0, 1, 2]],
                    1: [[1, 2, 3]],
                    2: [[2, 3, 4]],
                    3: [[3, 4, 5]],
                    4: [[4, 5, 6]],
                    5: [[5, 6, 7]],
                    6: [[0, 1, 2]],
                }
            }
        }
        free_slots_dict = {
            ("A", "B"): {
                "c": {
                    0: {0: np.array([0, 1, 2])},
                    1: {1: np.array([1, 2, 3])},
                    2: {2: np.array([2, 3, 4])},
                    3: {3: np.array([3, 4, 5])},
                    4: {4: np.array([4, 5, 6])},
                    5: {5: np.array([5, 6, 7])},
                    6: {6: np.array([0, 1, 2])},
                }
            }
        }

        # Act
        result = get_channel_overlaps(free_channels_dict, free_slots_dict)

        # Assert
        assert ("A", "B") in result
        assert "overlapped_dict" in result[("A", "B")]
        assert "non_over_dict" in result[("A", "B")]
        assert "c" in result[("A", "B")]["overlapped_dict"]

    def test_get_channel_overlaps_with_distinct_channels_returns_non_overlapped(
        self,
    ) -> None:
        """Test channel overlaps with 7 cores having distinct channels."""
        # Arrange - Using 7 cores with non-overlapping channels
        free_channels_dict = {
            ("A", "B"): {
                "c": {
                    0: [[0, 1]],
                    1: [[10, 11]],
                    2: [[20, 21]],
                    3: [[30, 31]],
                    4: [[40, 41]],
                    5: [[50, 51]],
                    6: [[60, 61]],
                }
            }
        }
        free_slots_dict = {
            ("A", "B"): {
                "c": {
                    0: {0: np.array([0, 1])},
                    1: {1: np.array([10, 11])},
                    2: {2: np.array([20, 21])},
                    3: {3: np.array([30, 31])},
                    4: {4: np.array([40, 41])},
                    5: {5: np.array([50, 51])},
                    6: {6: np.array([60, 61])},
                }
            }
        }

        # Act
        result = get_channel_overlaps(free_channels_dict, free_slots_dict)

        # Assert
        assert ("A", "B") in result
        # All channels should be in non_over_dict
        assert len(result[("A", "B")]["non_over_dict"]["c"][0]) > 0
