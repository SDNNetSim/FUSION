"""Unit tests for fusion.utils.network module."""

import networkx as nx
import numpy as np
import pytest

from fusion.utils.network import (
    find_core_congestion,
    find_path_congestion,
    find_path_fragmentation,
    find_path_length,
    get_path_modulation,
)


class TestFindPathLength:
    """Tests for find_path_length function."""

    def test_find_path_length_with_single_hop_returns_length(self) -> None:
        """Test finding path length for single hop path."""
        # Arrange
        topology = nx.Graph()
        topology.add_edge(1, 2, length=100.0)
        path_list = [1, 2]

        # Act
        result = find_path_length(path_list, topology)

        # Assert
        assert result == 100.0

    def test_find_path_length_with_multi_hop_sums_lengths(self) -> None:
        """Test finding path length sums all hop lengths."""
        # Arrange
        topology = nx.Graph()
        topology.add_edge(1, 2, length=100.0)
        topology.add_edge(2, 3, length=150.0)
        topology.add_edge(3, 4, length=75.0)
        path_list = [1, 2, 3, 4]

        # Act
        result = find_path_length(path_list, topology)

        # Assert
        assert result == 325.0

    def test_find_path_length_with_single_node_returns_zero(self) -> None:
        """Test path length for single node is zero."""
        # Arrange
        topology = nx.Graph()
        topology.add_node(1)
        path_list = [1]

        # Act
        result = find_path_length(path_list, topology)

        # Assert
        assert result == 0

    def test_find_path_length_with_varying_lengths_calculates_correctly(self) -> None:
        """Test path length calculation with varying hop lengths."""
        # Arrange
        topology = nx.Graph()
        topology.add_edge(1, 2, length=50.5)
        topology.add_edge(2, 3, length=25.3)
        path_list = [1, 2, 3]

        # Act
        result = find_path_length(path_list, topology)

        # Assert
        assert abs(result - 75.8) < 1e-10


class TestFindCoreCongestion:
    """Tests for find_core_congestion function."""

    def test_find_core_congestion_with_empty_core_returns_zero(self) -> None:
        """Test core congestion on empty core."""
        # Arrange
        network_spectrum = {
            (1, 2): {"cores_matrix": {"c": np.array([[0, 0, 0, 0], [0, 0, 0, 0]])}},
        }
        core_index = 0
        path_list = [1, 2]

        # Act
        result = find_core_congestion(core_index, network_spectrum, path_list)

        # Assert
        assert result == 0.0

    def test_find_core_congestion_with_full_core_returns_one(self) -> None:
        """Test core congestion on fully occupied core."""
        # Arrange
        network_spectrum = {
            (1, 2): {"cores_matrix": {"c": np.array([[1, 1, 1, 1], [0, 0, 0, 0]])}},
        }
        core_index = 0
        path_list = [1, 2]

        # Act
        result = find_core_congestion(core_index, network_spectrum, path_list)

        # Assert
        assert result == 1.0

    def test_find_core_congestion_with_partial_occupancy_returns_fraction(self) -> None:
        """Test core congestion with partial occupancy."""
        # Arrange
        network_spectrum = {
            (1, 2): {"cores_matrix": {"c": np.array([[1, 1, 0, 0], [0, 0, 0, 0]])}},
        }
        core_index = 0
        path_list = [1, 2]

        # Act
        result = find_core_congestion(core_index, network_spectrum, path_list)

        # Assert
        assert result == 0.5

    def test_find_core_congestion_with_multi_hop_averages_congestion(self) -> None:
        """Test core congestion averages across multiple hops."""
        # Arrange
        network_spectrum = {
            (1, 2): {"cores_matrix": {"c": np.array([[1, 1, 0, 0]])}},
            (2, 3): {"cores_matrix": {"c": np.array([[1, 0, 0, 0]])}},
        }
        core_index = 0
        path_list = [1, 2, 3]

        # Act
        result = find_core_congestion(core_index, network_spectrum, path_list)

        # Assert
        # First link: 2/4 = 0.5, Second link: 1/4 = 0.25, Average: 0.375
        assert abs(result - 0.375) < 1e-10

    def test_find_core_congestion_with_multiple_bands_considers_all(self) -> None:
        """Test core congestion considers all bands."""
        # Arrange
        network_spectrum = {
            (1, 2): {
                "cores_matrix": {
                    "c": np.array([[1, 1, 0, 0]]),
                    "l": np.array([[1, 0, 0, 0]]),
                }
            },
        }
        core_index = 0
        path_list = [1, 2]

        # Act
        result = find_core_congestion(core_index, network_spectrum, path_list)

        # Assert
        # Total slots: 8 (4 from c + 4 from l), Taken: 3, Congestion: 3/8 = 0.375
        assert abs(result - 0.375) < 1e-10


class TestGetPathModulation:
    """Tests for get_path_modulation function."""

    def test_get_path_modulation_with_short_path_returns_64qam(self) -> None:
        """Test short path returns 64-QAM modulation."""
        # Arrange
        modulation_formats = {
            "QPSK": {"max_length": 3000},
            "16-QAM": {"max_length": 2000},
            "64-QAM": {"max_length": 1000},
        }
        path_length = 500.0

        # Act
        result = get_path_modulation(
            modulation_formats=modulation_formats, path_length=path_length
        )

        # Assert
        assert result == "64-QAM"

    def test_get_path_modulation_with_medium_path_returns_16qam(self) -> None:
        """Test medium path returns 16-QAM modulation."""
        # Arrange
        modulation_formats = {
            "QPSK": {"max_length": 3000},
            "16-QAM": {"max_length": 2000},
            "64-QAM": {"max_length": 1000},
        }
        path_length = 1500.0

        # Act
        result = get_path_modulation(
            modulation_formats=modulation_formats, path_length=path_length
        )

        # Assert
        assert result == "16-QAM"

    def test_get_path_modulation_with_long_path_returns_qpsk(self) -> None:
        """Test long path returns QPSK modulation."""
        # Arrange
        modulation_formats = {
            "QPSK": {"max_length": 3000},
            "16-QAM": {"max_length": 2000},
            "64-QAM": {"max_length": 1000},
        }
        path_length = 2500.0

        # Act
        result = get_path_modulation(
            modulation_formats=modulation_formats, path_length=path_length
        )

        # Assert
        assert result == "QPSK"

    def test_get_path_modulation_with_too_long_path_returns_false(self) -> None:
        """Test path too long for any modulation returns False."""
        # Arrange
        modulation_formats = {
            "QPSK": {"max_length": 3000},
            "16-QAM": {"max_length": 2000},
            "64-QAM": {"max_length": 1000},
        }
        path_length = 3500.0

        # Act
        result = get_path_modulation(
            modulation_formats=modulation_formats, path_length=path_length
        )

        # Assert
        assert result is False

    def test_get_path_modulation_at_boundary_chooses_higher_modulation(self) -> None:
        """Test path at max boundary chooses that modulation."""
        # Arrange
        modulation_formats = {
            "QPSK": {"max_length": 3000},
            "16-QAM": {"max_length": 2000},
            "64-QAM": {"max_length": 1000},
        }
        path_length = 1000.0

        # Act
        result = get_path_modulation(
            modulation_formats=modulation_formats, path_length=path_length
        )

        # Assert
        assert result == "64-QAM"

    def test_get_path_modulation_with_legacy_parameters_works(self) -> None:
        """Test backward compatibility with legacy parameter names."""
        # Arrange
        mods_dict = {
            "QPSK": {"max_length": 3000},
            "16-QAM": {"max_length": 2000},
            "64-QAM": {"max_length": 1000},
        }
        path_len = 500.0

        # Act
        result = get_path_modulation(mods_dict=mods_dict, path_len=path_len)

        # Assert
        assert result == "64-QAM"

    def test_get_path_modulation_with_none_parameters_raises_error(self) -> None:
        """Test that None parameters raise ValueError."""
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            get_path_modulation(modulation_formats=None, path_length=500.0)

        assert "Must provide modulation_formats and path_length" in str(exc_info.value)


class TestFindPathCongestion:
    """Tests for find_path_congestion function."""

    def test_find_path_congestion_with_empty_path_returns_zeros(self) -> None:
        """Test path congestion on empty spectrum."""
        # Arrange
        network_spectrum = {
            (1, 2): {"cores_matrix": {"c": np.array([[0, 0, 0], [0, 0, 0]])}},
        }
        path_list = [1, 2]

        # Act
        congestion, capacity = find_path_congestion(path_list, network_spectrum, "c")

        # Assert
        assert congestion == 0.0
        assert capacity == 6.0  # 2 cores * 3 slots

    def test_find_path_congestion_with_full_path_returns_one(self) -> None:
        """Test path congestion on fully occupied spectrum."""
        # Arrange
        network_spectrum = {
            (1, 2): {"cores_matrix": {"c": np.array([[1, 1, 1], [1, 1, 1]])}},
        }
        path_list = [1, 2]

        # Act
        congestion, capacity = find_path_congestion(path_list, network_spectrum, "c")

        # Assert
        assert congestion == 1.0
        assert capacity == 0.0

    def test_find_path_congestion_with_partial_occupancy(self) -> None:
        """Test path congestion with partial occupancy."""
        # Arrange
        network_spectrum = {
            (1, 2): {"cores_matrix": {"c": np.array([[1, 0, 0], [0, 0, 0]])}},
        }
        path_list = [1, 2]

        # Act
        congestion, capacity = find_path_congestion(path_list, network_spectrum, "c")

        # Assert
        # 1 slot taken out of 6 total = 1/6 congestion
        assert abs(congestion - (1.0 / 6.0)) < 1e-10
        assert capacity == 5.0

    def test_find_path_congestion_with_multi_hop_averages_congestion(self) -> None:
        """Test path congestion averages over multiple hops."""
        # Arrange
        network_spectrum = {
            (1, 2): {"cores_matrix": {"c": np.array([[1, 1], [0, 0]])}},
            (2, 3): {"cores_matrix": {"c": np.array([[1, 0], [0, 0]])}},
        }
        path_list = [1, 2, 3]

        # Act
        congestion, capacity = find_path_congestion(path_list, network_spectrum, "c")

        # Assert
        # Link 1: 2/4 = 0.5, Link 2: 1/4 = 0.25, Average: 0.375
        assert abs(congestion - 0.375) < 1e-10
        # Capacity: (4-2) + (4-1) = 5
        assert capacity == 5.0

    def test_find_path_congestion_with_different_band(self) -> None:
        """Test path congestion with non-default band."""
        # Arrange
        network_spectrum = {
            (1, 2): {
                "cores_matrix": {
                    "c": np.array([[1, 1]]),
                    "l": np.array([[1, 0]]),
                }
            },
        }
        path_list = [1, 2]

        # Act
        congestion, capacity = find_path_congestion(path_list, network_spectrum, "l")

        # Assert
        assert congestion == 0.5
        assert capacity == 1.0


class TestFindPathFragmentation:
    """Tests for find_path_fragmentation function."""

    def test_find_path_fragmentation_with_empty_core_returns_zero(self) -> None:
        """Test fragmentation on empty core is zero."""
        # Arrange
        network_spectrum = {
            (1, 2): {"cores_matrix": {"c": np.array([[0, 0, 0, 0]])}},
        }
        path_list = [1, 2]

        # Act
        result = find_path_fragmentation(path_list, network_spectrum, "c")

        # Assert
        # Empty core: max_block = 4, total_free = 4, fragmentation = 1 - 4/4 = 0
        assert result == 0.0

    def test_find_path_fragmentation_with_full_core_returns_one(self) -> None:
        """Test fragmentation on fully occupied core is one."""
        # Arrange
        network_spectrum = {
            (1, 2): {"cores_matrix": {"c": np.array([[1, 1, 1, 1]])}},
        }
        path_list = [1, 2]

        # Act
        result = find_path_fragmentation(path_list, network_spectrum, "c")

        # Assert
        # Fully occupied: fragmentation = 1.0
        assert result == 1.0

    def test_find_path_fragmentation_with_fragmented_core(self) -> None:
        """Test fragmentation with fragmented free blocks."""
        # Arrange
        network_spectrum = {
            (1, 2): {"cores_matrix": {"c": np.array([[0, 1, 0, 1, 0, 0]])}},
        }
        path_list = [1, 2]

        # Act
        result = find_path_fragmentation(path_list, network_spectrum, "c")

        # Assert
        # Free slots: [0, 2, 4, 5] = 4 total, max_block = 2
        # fragmentation = 1 - 2/4 = 0.5
        assert result == 0.5

    def test_find_path_fragmentation_with_contiguous_free_block(self) -> None:
        """Test fragmentation with one contiguous free block."""
        # Arrange
        network_spectrum = {
            (1, 2): {"cores_matrix": {"c": np.array([[1, 1, 0, 0, 0, 0]])}},
        }
        path_list = [1, 2]

        # Act
        result = find_path_fragmentation(path_list, network_spectrum, "c")

        # Assert
        # Free slots: 4, max_block = 4, fragmentation = 1 - 4/4 = 0
        assert result == 0.0

    def test_find_path_fragmentation_with_multi_hop_averages(self) -> None:
        """Test fragmentation averages across multiple hops."""
        # Arrange
        network_spectrum = {
            (1, 2): {"cores_matrix": {"c": np.array([[0, 0, 0, 0]])}},
            (2, 3): {"cores_matrix": {"c": np.array([[0, 1, 0, 1]])}},
        }
        path_list = [1, 2, 3]

        # Act
        result = find_path_fragmentation(path_list, network_spectrum, "c")

        # Assert
        # Link 1: fragmentation = 0, Link 2: max_block=1, total_free=2, frag=0.5
        # Average = 0.25
        assert abs(result - 0.25) < 1e-10

    def test_find_path_fragmentation_with_multiple_cores_considers_all(self) -> None:
        """Test fragmentation considers all cores."""
        # Arrange
        network_spectrum = {
            (1, 2): {
                "cores_matrix": {
                    "c": np.array([[0, 0, 0, 0], [0, 1, 0, 1]]),
                }
            },
        }
        path_list = [1, 2]

        # Act
        result = find_path_fragmentation(path_list, network_spectrum, "c")

        # Assert
        # Core 0: frag=0, Core 1: frag=0.5, Average=0.25
        assert abs(result - 0.25) < 1e-10

    def test_find_path_fragmentation_with_different_band(self) -> None:
        """Test fragmentation with non-default band."""
        # Arrange
        network_spectrum = {
            (1, 2): {
                "cores_matrix": {
                    "c": np.array([[1, 1, 1, 1]]),
                    "l": np.array([[0, 0, 0, 0]]),
                }
            },
        }
        path_list = [1, 2]

        # Act
        result = find_path_fragmentation(path_list, network_spectrum, "l")

        # Assert
        assert result == 0.0

    def test_find_path_fragmentation_with_trailing_free_block(self) -> None:
        """Test fragmentation correctly handles trailing free block."""
        # Arrange
        network_spectrum = {
            (1, 2): {"cores_matrix": {"c": np.array([[1, 1, 0, 0, 0]])}},
        }
        path_list = [1, 2]

        # Act
        result = find_path_fragmentation(path_list, network_spectrum, "c")

        # Assert
        # Free: 3, max_block: 3, fragmentation = 1 - 3/3 = 0
        assert result == 0.0
