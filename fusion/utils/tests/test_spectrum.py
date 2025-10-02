"""Unit tests for fusion.utils.spectrum module."""

from typing import Any

import numpy as np
import pytest

from fusion.utils.spectrum import (
    find_free_channels,
    find_free_slots,
    find_taken_channels,
    get_channel_overlaps,
)


class TestFindFreeSlots:
    """Tests for find_free_slots function."""

    def test_find_free_slots_with_empty_spectrum_returns_all_slots(self) -> None:
        """Test finding free slots on empty spectrum."""
        # Arrange
        network_spectrum = {
            (1, 2): {
                "cores_matrix": {
                    "c": np.array([[0, 0, 0, 0], [0, 0, 0, 0]]),
                }
            }
        }
        link_tuple = (1, 2)

        # Act
        result = find_free_slots(
            network_spectrum=network_spectrum, link_tuple=link_tuple
        )

        # Assert
        assert "c" in result
        assert 0 in result["c"]
        assert 1 in result["c"]
        np.testing.assert_array_equal(result["c"][0], np.array([0, 1, 2, 3]))
        np.testing.assert_array_equal(result["c"][1], np.array([0, 1, 2, 3]))

    def test_find_free_slots_with_occupied_slots_returns_free_only(self) -> None:
        """Test finding free slots excludes occupied slots."""
        # Arrange
        network_spectrum = {
            (1, 2): {
                "cores_matrix": {
                    "c": np.array([[1, 0, 0, 1], [0, 1, 0, 0]]),
                }
            }
        }
        link_tuple = (1, 2)

        # Act
        result = find_free_slots(
            network_spectrum=network_spectrum, link_tuple=link_tuple
        )

        # Assert
        np.testing.assert_array_equal(result["c"][0], np.array([1, 2]))
        np.testing.assert_array_equal(result["c"][1], np.array([0, 2, 3]))

    def test_find_free_slots_with_multiple_bands_returns_all_bands(self) -> None:
        """Test finding free slots across multiple bands."""
        # Arrange
        network_spectrum = {
            (1, 2): {
                "cores_matrix": {
                    "c": np.array([[0, 0], [0, 0]]),
                    "l": np.array([[1, 0], [0, 1]]),
                }
            }
        }
        link_tuple = (1, 2)

        # Act
        result = find_free_slots(
            network_spectrum=network_spectrum, link_tuple=link_tuple
        )

        # Assert
        assert "c" in result
        assert "l" in result
        np.testing.assert_array_equal(result["c"][0], np.array([0, 1]))
        np.testing.assert_array_equal(result["l"][0], np.array([1]))
        np.testing.assert_array_equal(result["l"][1], np.array([0]))

    def test_find_free_slots_with_none_network_spectrum_raises_error(self) -> None:
        """Test that None network_spectrum raises ValueError."""
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            find_free_slots(network_spectrum=None, link_tuple=(1, 2))

        assert "Must provide network_spectrum and link_tuple" in str(exc_info.value)

    def test_find_free_slots_with_none_link_tuple_raises_error(self) -> None:
        """Test that None link_tuple raises ValueError."""
        # Arrange
        network_spectrum = {(1, 2): {"cores_matrix": {"c": np.array([[0]])}}}

        # Act & Assert
        with pytest.raises(ValueError):
            find_free_slots(network_spectrum=network_spectrum, link_tuple=None)

    def test_find_free_slots_with_legacy_parameter_name_works(self) -> None:
        """Test backward compatibility with network_spectrum_dict parameter."""
        # Arrange
        network_spectrum = {(1, 2): {"cores_matrix": {"c": np.array([[0, 1], [1, 0]])}}}

        # Act
        result = find_free_slots(
            network_spectrum_dict=network_spectrum, link_tuple=(1, 2)
        )

        # Assert
        assert "c" in result
        np.testing.assert_array_equal(result["c"][0], np.array([0]))


class TestFindFreeChannels:
    """Tests for find_free_channels function."""

    def test_find_free_channels_with_contiguous_slots_finds_channels(self) -> None:
        """Test finding free channels with contiguous free slots."""
        # Arrange
        network_spectrum = {
            (1, 2): {
                "cores_matrix": {
                    "c": np.array([[0, 0, 0, 1, 1]]),
                }
            }
        }
        slots_needed = 2
        link_tuple = (1, 2)

        # Act
        result = find_free_channels(
            network_spectrum=network_spectrum,
            slots_needed=slots_needed,
            link_tuple=link_tuple,
        )

        # Assert
        assert "c" in result
        assert 0 in result["c"]
        assert [0, 1] in result["c"][0]
        assert [1, 2] in result["c"][0]

    def test_find_free_channels_with_insufficient_space_returns_empty(self) -> None:
        """Test finding channels when insufficient contiguous space."""
        # Arrange
        network_spectrum = {
            (1, 2): {
                "cores_matrix": {
                    "c": np.array([[0, 1, 0, 1, 0]]),
                }
            }
        }
        slots_needed = 3
        link_tuple = (1, 2)

        # Act
        result = find_free_channels(
            network_spectrum=network_spectrum,
            slots_needed=slots_needed,
            link_tuple=link_tuple,
        )

        # Assert
        assert result["c"][0] == []

    def test_find_free_channels_with_exact_fit_finds_channel(self) -> None:
        """Test finding channel that exactly fits available space."""
        # Arrange
        network_spectrum = {
            (1, 2): {
                "cores_matrix": {
                    "c": np.array([[0, 0, 0]]),
                }
            }
        }
        slots_needed = 3
        link_tuple = (1, 2)

        # Act
        result = find_free_channels(
            network_spectrum=network_spectrum,
            slots_needed=slots_needed,
            link_tuple=link_tuple,
        )

        # Assert
        assert [0, 1, 2] in result["c"][0]

    def test_find_free_channels_with_multiple_cores_processes_all(self) -> None:
        """Test finding channels across multiple cores."""
        # Arrange
        network_spectrum = {
            (1, 2): {
                "cores_matrix": {
                    "c": np.array([[0, 0, 1], [1, 0, 0]]),
                }
            }
        }
        slots_needed = 2
        link_tuple = (1, 2)

        # Act
        result = find_free_channels(
            network_spectrum=network_spectrum,
            slots_needed=slots_needed,
            link_tuple=link_tuple,
        )

        # Assert
        assert [0, 1] in result["c"][0]
        assert [1, 2] in result["c"][1]

    def test_find_free_channels_with_none_parameters_raises_error(self) -> None:
        """Test that None parameters raise ValueError."""
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            find_free_channels(network_spectrum=None, slots_needed=2, link_tuple=(1, 2))

        assert "Must provide network_spectrum, link_tuple, and slots_needed" in str(
            exc_info.value
        )

    def test_find_free_channels_with_legacy_parameters_works(self) -> None:
        """Test backward compatibility with legacy parameter names."""
        # Arrange
        network_spectrum = {(1, 2): {"cores_matrix": {"c": np.array([[0, 0, 0]])}}}

        # Act
        result = find_free_channels(
            network_spectrum_dict=network_spectrum, slots_needed=2, link_tuple=(1, 2)
        )

        # Assert
        assert [0, 1] in result["c"][0]

    def test_find_free_channels_sliding_window_behavior(self) -> None:
        """Test sliding window finds all possible channels."""
        # Arrange
        network_spectrum = {
            (1, 2): {
                "cores_matrix": {
                    "c": np.array([[0, 0, 0, 0]]),
                }
            }
        }
        slots_needed = 2

        # Act
        result = find_free_channels(
            network_spectrum=network_spectrum,
            slots_needed=slots_needed,
            link_tuple=(1, 2),
        )

        # Assert - should find 3 possible 2-slot channels
        assert len(result["c"][0]) == 3
        assert [0, 1] in result["c"][0]
        assert [1, 2] in result["c"][0]
        assert [2, 3] in result["c"][0]


class TestFindTakenChannels:
    """Tests for find_taken_channels function."""

    def test_find_taken_channels_with_empty_spectrum_returns_empty(self) -> None:
        """Test finding taken channels on empty spectrum."""
        # Arrange
        network_spectrum = {(1, 2): {"cores_matrix": {"c": np.array([[0, 0, 0]])}}}

        # Act
        result = find_taken_channels(
            network_spectrum=network_spectrum, link_tuple=(1, 2)
        )

        # Assert
        assert result["c"][0] == []

    def test_find_taken_channels_with_occupied_slots_finds_channels(self) -> None:
        """Test finding taken channels with occupied slots."""
        # Arrange
        network_spectrum = {
            (1, 2): {
                "cores_matrix": {
                    "c": np.array([[1, 1, -1, 0, 2, 2, -1]]),
                }
            }
        }

        # Act
        result = find_taken_channels(
            network_spectrum=network_spectrum, link_tuple=(1, 2)
        )

        # Assert
        assert [1, 1] in result["c"][0]
        assert [2, 2] in result["c"][0]

    def test_find_taken_channels_with_trailing_occupied_includes_trailing(self) -> None:
        """Test that trailing occupied slots are included."""
        # Arrange
        network_spectrum = {
            (1, 2): {
                "cores_matrix": {
                    "c": np.array([[0, 1, 1, 1]]),
                }
            }
        }

        # Act
        result = find_taken_channels(
            network_spectrum=network_spectrum, link_tuple=(1, 2)
        )

        # Assert
        assert len(result["c"][0]) == 1
        assert [1, 1, 1] in result["c"][0]

    def test_find_taken_channels_with_multiple_bands_processes_all(self) -> None:
        """Test finding taken channels across multiple bands."""
        # Arrange
        network_spectrum = {
            (1, 2): {
                "cores_matrix": {
                    "c": np.array([[1, -1, 0]]),
                    "l": np.array([[0, 2, -1]]),
                }
            }
        }

        # Act
        result = find_taken_channels(
            network_spectrum=network_spectrum, link_tuple=(1, 2)
        )

        # Assert
        assert [1] in result["c"][0]
        assert [2] in result["l"][0]

    def test_find_taken_channels_with_none_parameters_raises_error(self) -> None:
        """Test that None parameters raise ValueError."""
        # Act & Assert
        with pytest.raises(ValueError):
            find_taken_channels(network_spectrum=None, link_tuple=(1, 2))

    def test_find_taken_channels_with_legacy_parameter_works(self) -> None:
        """Test backward compatibility with legacy parameter name."""
        # Arrange
        network_spectrum = {(1, 2): {"cores_matrix": {"c": np.array([[1, 1, -1]])}}}

        # Act
        result = find_taken_channels(
            network_spectrum_dict=network_spectrum, link_tuple=(1, 2)
        )

        # Assert
        assert [1, 1] in result["c"][0]


class TestGetChannelOverlaps:
    """Tests for get_channel_overlaps function."""

    def test_get_channel_overlaps_basic_structure(self) -> None:
        """Test channel overlaps returns correct structure."""
        # Arrange - need 7 cores (0-6), nested structure for slots
        cores_slots = {i: np.array([0, 1, 2]) for i in range(7)}
        free_channels_dict = {(1, 2): {"c": {i: [[0, 1, 2]] for i in range(7)}}}
        free_slots_dict: dict[Any, dict[str, dict[int, dict[int, Any]]]] = {
            (1, 2): {"c": {i: cores_slots.copy() for i in range(7)}}
        }

        # Act
        result = get_channel_overlaps(free_channels_dict, free_slots_dict)

        # Assert
        assert (1, 2) in result
        assert "overlapped_dict" in result[(1, 2)]
        assert "non_over_dict" in result[(1, 2)]
        assert "c" in result[(1, 2)]["overlapped_dict"]

    def test_get_channel_overlaps_with_empty_channels_returns_empty_lists(self) -> None:
        """Test channel overlaps with no free channels."""
        # Arrange - need 7 cores minimum, nested structure
        cores_slots = {i: np.array([]) for i in range(7)}
        free_channels_dict: dict[tuple[int, int], dict[str, dict[int, list[Any]]]] = {
            (1, 2): {"c": {i: [] for i in range(7)}}
        }
        free_slots_dict: dict[Any, dict[str, dict[int, dict[int, Any]]]] = {
            (1, 2): {"c": {i: cores_slots.copy() for i in range(7)}}
        }

        # Act
        result = get_channel_overlaps(free_channels_dict, free_slots_dict)

        # Assert
        assert result[(1, 2)]["overlapped_dict"]["c"][0] == []
        assert result[(1, 2)]["non_over_dict"]["c"][0] == []

    def test_get_channel_overlaps_processes_multiple_links(self) -> None:
        """Test channel overlaps processes multiple links."""
        # Arrange - need 7 cores, nested structure
        cores_dict: dict[int, list[Any]] = {i: [] for i in range(7)}
        cores_slots = {i: np.array([]) for i in range(7)}
        slots_nested = {i: cores_slots.copy() for i in range(7)}
        free_channels_dict = {
            (1, 2): {"c": cores_dict.copy()},
            (2, 3): {"c": cores_dict.copy()},
        }
        free_slots_dict: dict[Any, dict[str, dict[int, dict[int, Any]]]] = {
            (1, 2): {"c": slots_nested.copy()},
            (2, 3): {"c": slots_nested.copy()},
        }

        # Act
        result = get_channel_overlaps(free_channels_dict, free_slots_dict)

        # Assert
        assert (1, 2) in result
        assert (2, 3) in result

    def test_get_channel_overlaps_handles_multiple_bands(self) -> None:
        """Test channel overlaps handles multiple bands."""
        # Arrange - need 7 cores, nested structure
        cores_dict: dict[int, list[Any]] = {i: [] for i in range(7)}
        cores_slots = {i: np.array([]) for i in range(7)}
        slots_nested = {i: cores_slots.copy() for i in range(7)}
        free_channels_dict = {
            (1, 2): {
                "c": cores_dict.copy(),
                "l": cores_dict.copy(),
            }
        }
        free_slots_dict: dict[Any, dict[str, dict[int, dict[int, Any]]]] = {
            (1, 2): {
                "c": slots_nested.copy(),
                "l": slots_nested.copy(),
            }
        }

        # Act
        result = get_channel_overlaps(free_channels_dict, free_slots_dict)

        # Assert
        assert "c" in result[(1, 2)]["overlapped_dict"]
        assert "l" in result[(1, 2)]["overlapped_dict"]
