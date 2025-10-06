"""Unit tests for fusion/sim/utils/network.py module."""

import networkx as nx
import numpy as np
import pytest

from fusion.sim.utils.network import (
    classify_congestion,
    find_core_congestion,
    find_core_fragmentation_congestion,
    find_max_path_length,
    find_path_congestion,
    find_path_fragmentation,
    find_path_length,
    get_path_modulation,
)


@pytest.fixture
def sample_topology() -> nx.Graph:
    """Provide sample network topology for tests."""
    topology = nx.Graph()
    topology.add_edge(1, 2, length=10)
    topology.add_edge(2, 3, length=20)
    topology.add_edge(1, 3, length=50)
    topology.add_edge(3, 4, length=15)
    return topology


@pytest.fixture
def sample_modulation_formats() -> dict:
    """Provide sample modulation format configuration."""
    return {
        "QPSK": {"max_length": 2000},
        "16-QAM": {"max_length": 1000},
        "64-QAM": {"max_length": 500},
    }


@pytest.fixture
def sample_network_spectrum() -> dict:
    """Provide sample network spectrum for congestion tests."""
    return {
        (1, 2): {"cores_matrix": {"c": np.array([[0, 1, 1], [1, 1, 0]])}},
        (2, 3): {"cores_matrix": {"c": np.array([[1, 0, 0], [0, 0, 1]])}},
    }


class TestFindPathLength:
    """Tests for find_path_length function."""

    def test_find_path_length_with_single_hop_returns_correct_length(
        self, sample_topology: nx.Graph
    ) -> None:
        """Test that path length is calculated correctly for single hop."""
        # Arrange
        path_list = [1, 2]

        # Act
        result = find_path_length(path_list, sample_topology)

        # Assert
        assert result == 10

    def test_find_path_length_with_multiple_hops_sums_lengths(
        self, sample_topology: nx.Graph
    ) -> None:
        """Test that path length is sum of all hop lengths."""
        # Arrange
        path_list = [1, 2, 3, 4]

        # Act
        result = find_path_length(path_list, sample_topology)

        # Assert
        assert result == 45  # 10 + 20 + 15


class TestFindMaxPathLength:
    """Tests for find_max_path_length function."""

    def test_find_max_path_length_returns_longest_path(
        self, sample_topology: nx.Graph
    ) -> None:
        """Test that maximum path length between nodes is found."""
        # Arrange
        source, destination = 1, 3

        # Act
        result = find_max_path_length(source, destination, sample_topology)

        # Assert
        assert result == 30  # Path via node 2: 10 + 20


class TestGetPathModulation:
    """Tests for get_path_modulation function."""

    def test_get_path_modulation_with_short_path_returns_64qam(
        self, sample_modulation_formats: dict
    ) -> None:
        """Test that short path selects 64-QAM modulation."""
        # Arrange
        path_length = 400.0

        # Act
        result = get_path_modulation(sample_modulation_formats, path_length)

        # Assert
        assert result == "64-QAM"

    def test_get_path_modulation_with_medium_path_returns_16qam(
        self, sample_modulation_formats: dict
    ) -> None:
        """Test that medium path selects 16-QAM modulation."""
        # Arrange
        path_length = 800.0

        # Act
        result = get_path_modulation(sample_modulation_formats, path_length)

        # Assert
        assert result == "16-QAM"

    def test_get_path_modulation_with_long_path_returns_qpsk(
        self, sample_modulation_formats: dict
    ) -> None:
        """Test that long path selects QPSK modulation."""
        # Arrange
        path_length = 1500.0

        # Act
        result = get_path_modulation(sample_modulation_formats, path_length)

        # Assert
        assert result == "QPSK"

    def test_get_path_modulation_with_excessive_path_returns_false(
        self, sample_modulation_formats: dict
    ) -> None:
        """Test that excessively long path returns False."""
        # Arrange
        path_length = 2500.0

        # Act
        result = get_path_modulation(sample_modulation_formats, path_length)

        # Assert
        assert result is False

    def test_get_path_modulation_with_legacy_params_works(
        self, sample_modulation_formats: dict
    ) -> None:
        """Test backward compatibility with legacy parameter names."""
        # Arrange
        path_len = 800.0

        # Act
        result = get_path_modulation(
            mods_dict=sample_modulation_formats, path_len=path_len
        )

        # Assert
        assert result == "16-QAM"

    def test_get_path_modulation_without_params_raises_error(self) -> None:
        """Test that missing required parameters raises ValueError."""
        # Act & Assert
        with pytest.raises(
            ValueError, match="Must provide modulation_formats and path_length"
        ):
            get_path_modulation()


class TestFindPathCongestion:
    """Tests for find_path_congestion function."""

    def test_find_path_congestion_calculates_average_correctly(
        self, sample_network_spectrum: dict
    ) -> None:
        """Test that average congestion is calculated correctly."""
        # Arrange
        path_list = [1, 2, 3]

        # Act
        avg_cong, scaled_cap = find_path_congestion(path_list, sample_network_spectrum)

        # Assert
        # Link (1,2): 4 slots taken out of 6 total = 4/6
        # Link (2,3): 2 slots taken out of 6 total = 2/6
        # Average: (4/6 + 2/6) / 2
        expected_avg_cong = ((4 / 6) + (2 / 6)) / 2
        assert avg_cong == pytest.approx(expected_avg_cong, rel=1e-5)
        # Scaled capacity: (6-4) + (6-2) = 6
        assert scaled_cap == 6.0


class TestFindPathFragmentation:
    """Tests for find_path_fragmentation function."""

    def test_find_path_fragmentation_with_no_fragmentation_returns_zero(self) -> None:
        """Test that unfragmented spectrum returns zero fragmentation."""
        # Arrange
        network_spectrum = {
            (1, 2): {"cores_matrix": {"c": np.array([[0, 0, 0, 1, 1, 1]])}},
        }
        path_list = [1, 2]

        # Act
        result = find_path_fragmentation(path_list, network_spectrum)

        # Assert
        # Single free block and single occupied block: no fragmentation
        assert result == 0.0

    def test_find_path_fragmentation_with_fragmented_spectrum_returns_ratio(
        self,
    ) -> None:
        """Test that fragmented spectrum returns correct ratio."""
        # Arrange
        network_spectrum = {
            (1, 2): {"cores_matrix": {"c": np.array([[0, 1, 0, 1, 0]])}},
        }
        path_list = [1, 2]

        # Act
        result = find_path_fragmentation(path_list, network_spectrum)

        # Assert
        # 3 free slots in 3 separate blocks, max_block=1, total_free=3
        # fragmentation = 1 - (1/3) = 0.667
        assert result == pytest.approx(2/3, rel=1e-5)


class TestFindCoreCongestion:
    """Tests for find_core_congestion function."""

    def test_find_core_congestion_calculates_core_percentage(
        self, sample_network_spectrum: dict
    ) -> None:
        """Test that core congestion percentage is calculated correctly."""
        # Arrange
        path_list = [1, 2, 3]
        core_index = 0

        # Act
        result = find_core_congestion(core_index, sample_network_spectrum, path_list)

        # Assert
        # Core 0: Link (1,2) has 2 taken, Link (2,3) has 1 taken
        # Total: 3 taken out of 6 slots = 0.5
        expected = ((2 / 3) + (1 / 3)) / 2
        assert result == pytest.approx(expected, rel=1e-2)


class TestFindCoreFragmentationCongestion:
    """Tests for find_core_fragmentation_congestion function."""

    def test_find_core_frag_cong_with_empty_core_returns_zero(self) -> None:
        """Test that empty core returns zero fragmentation and congestion."""
        # Arrange
        network_spectrum = {
            (1, 2): {"cores_matrix": {"c": np.array([np.zeros(256)])}},
            (2, 3): {"cores_matrix": {"c": np.array([np.zeros(256)])}},
        }
        path_list = [1, 2, 3]
        core = 0
        band = "c"

        # Act
        frag, cong = find_core_fragmentation_congestion(
            network_spectrum, path_list, core, band
        )

        # Assert
        assert frag == 0.0
        assert cong == 0.0

    def test_find_core_frag_cong_with_invalid_slots_raises_error(self) -> None:
        """Test that non-256 slot array raises NotImplementedError."""
        # Arrange
        network_spectrum = {
            (1, 2): {"cores_matrix": {"c": np.array([np.zeros(128)])}},
        }
        path_list = [1, 2]
        core = 0
        band = "c"

        # Act & Assert
        with pytest.raises(
            NotImplementedError, match="Only works for 256 spectral slots"
        ):
            find_core_fragmentation_congestion(
                network_spectrum, path_list, core, band
            )


class TestClassifyCongestion:
    """Tests for classify_congestion function."""

    def test_classify_congestion_below_cutoff_returns_zero(self) -> None:
        """Test that congestion below cutoff returns level 0."""
        # Arrange
        current_congestion = 0.2
        congestion_cutoff = 0.3

        # Act
        result = classify_congestion(current_congestion, congestion_cutoff)

        # Assert
        assert result == 0

    def test_classify_congestion_above_cutoff_returns_one(self) -> None:
        """Test that congestion above cutoff returns level 1."""
        # Arrange
        current_congestion = 0.5
        congestion_cutoff = 0.3

        # Act
        result = classify_congestion(current_congestion, congestion_cutoff)

        # Assert
        assert result == 1

    def test_classify_congestion_at_cutoff_returns_zero(self) -> None:
        """Test that congestion exactly at cutoff returns level 0."""
        # Arrange
        current_congestion = 0.3
        congestion_cutoff = 0.3

        # Act
        result = classify_congestion(current_congestion, congestion_cutoff)

        # Assert
        assert result == 0
