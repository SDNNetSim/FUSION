"""Unit tests for fusion.sim.utils.network module."""

import networkx as nx
import numpy as np
import pytest

from ..network import (
    classify_congestion,
    find_core_congestion,
    find_core_fragmentation_congestion,
    find_max_path_length,
    find_path_congestion,
    find_path_fragmentation,
    find_path_length,
    get_path_modulation,
)


class TestPathLengthCalculations:
    """Tests for path length calculation functions."""

    @pytest.fixture
    def sample_topology(self) -> nx.Graph:
        """Provide a sample network topology for testing.

        :return: Sample network graph
        :rtype: nx.Graph
        """
        topology = nx.Graph()
        topology.add_edge(1, 2, length=10)
        topology.add_edge(2, 3, length=20)
        topology.add_edge(1, 3, length=50)
        topology.add_edge(3, 4, length=15)
        return topology

    def test_find_path_length_with_simple_path_returns_correct_sum(
        self, sample_topology: nx.Graph
    ) -> None:
        """Test that path length is calculated correctly for a simple path.

        :param sample_topology: Sample network topology
        :type sample_topology: nx.Graph
        """
        # Arrange
        path_list = [1, 2, 3, 4]
        expected_length = 45

        # Act
        result = find_path_length(path_list, sample_topology)

        # Assert
        assert result == expected_length

    def test_find_path_length_with_single_hop_returns_edge_length(
        self, sample_topology: nx.Graph
    ) -> None:
        """Test path length calculation for single hop.

        :param sample_topology: Sample network topology
        :type sample_topology: nx.Graph
        """
        # Arrange
        path_list = [1, 2]
        expected_length = 10

        # Act
        result = find_path_length(path_list, sample_topology)

        # Assert
        assert result == expected_length

    def test_find_max_path_length_with_multiple_paths_returns_longest(
        self, sample_topology: nx.Graph
    ) -> None:
        """Test finding the maximum path length between two nodes.

        :param sample_topology: Sample network topology
        :type sample_topology: nx.Graph
        """
        # Arrange
        source, destination = 1, 3
        expected_length = 30

        # Act
        result = find_max_path_length(source, destination, sample_topology)

        # Assert
        assert result == expected_length


class TestModulationSelection:
    """Tests for modulation format selection based on path length."""

    @pytest.fixture
    def modulation_formats(self) -> dict:
        """Provide standard modulation format specifications.

        :return: Modulation formats with max lengths
        :rtype: dict
        """
        return {
            "QPSK": {"max_length": 2000},
            "16-QAM": {"max_length": 1000},
            "64-QAM": {"max_length": 500},
        }

    def test_get_path_modulation_with_short_path_selects_64qam(
        self, modulation_formats: dict
    ) -> None:
        """Test that short paths select 64-QAM modulation.

        :param modulation_formats: Available modulation formats
        :type modulation_formats: dict
        """
        # Arrange
        path_length = 400
        expected_mod = "64-QAM"

        # Act
        result = get_path_modulation(
            modulation_formats=modulation_formats, path_length=path_length
        )

        # Assert
        assert result == expected_mod

    def test_get_path_modulation_with_medium_path_selects_16qam(
        self, modulation_formats: dict
    ) -> None:
        """Test that medium paths select 16-QAM modulation.

        :param modulation_formats: Available modulation formats
        :type modulation_formats: dict
        """
        # Arrange
        path_length = 800
        expected_mod = "16-QAM"

        # Act
        result = get_path_modulation(
            modulation_formats=modulation_formats, path_length=path_length
        )

        # Assert
        assert result == expected_mod

    def test_get_path_modulation_with_long_path_selects_qpsk(
        self, modulation_formats: dict
    ) -> None:
        """Test that long paths select QPSK modulation.

        :param modulation_formats: Available modulation formats
        :type modulation_formats: dict
        """
        # Arrange
        path_length = 1500
        expected_mod = "QPSK"

        # Act
        result = get_path_modulation(
            modulation_formats=modulation_formats, path_length=path_length
        )

        # Assert
        assert result == expected_mod

    def test_get_path_modulation_with_excessive_length_returns_false(
        self, modulation_formats: dict
    ) -> None:
        """Test that paths exceeding all formats return False.

        :param modulation_formats: Available modulation formats
        :type modulation_formats: dict
        """
        # Arrange
        path_length = 2500

        # Act
        result = get_path_modulation(
            modulation_formats=modulation_formats, path_length=path_length
        )

        # Assert
        assert result is False

    def test_get_path_modulation_with_boundary_length_selects_correct_format(
        self, modulation_formats: dict
    ) -> None:
        """Test modulation selection at boundary conditions.

        :param modulation_formats: Available modulation formats
        :type modulation_formats: dict
        """
        # Arrange - exactly at 64-QAM max
        path_length = 500
        expected_mod = "64-QAM"

        # Act
        result = get_path_modulation(
            modulation_formats=modulation_formats, path_length=path_length
        )

        # Assert
        assert result == expected_mod

    def test_get_path_modulation_with_legacy_parameters_works(
        self, modulation_formats: dict
    ) -> None:
        """Test backward compatibility with legacy parameter names.

        :param modulation_formats: Available modulation formats
        :type modulation_formats: dict
        """
        # Arrange
        path_len = 800
        expected_mod = "16-QAM"

        # Act
        result = get_path_modulation(mods_dict=modulation_formats, path_len=path_len)

        # Assert
        assert result == expected_mod

    def test_get_path_modulation_without_parameters_raises_error(self) -> None:
        """Test that missing required parameters raise ValueError."""
        # Act & Assert
        with pytest.raises(ValueError, match="Must provide modulation_formats"):
            get_path_modulation()


class TestCongestionMetrics:
    """Tests for network congestion calculation functions."""

    @pytest.fixture
    def network_spectrum(self) -> dict:
        """Provide sample network spectrum for testing.

        :return: Network spectrum database
        :rtype: dict
        """
        return {
            (1, 2): {"cores_matrix": {"c": np.array([[0, 1, 1], [1, 1, 0]])}},
            (2, 3): {"cores_matrix": {"c": np.array([[1, 0, 0], [0, 0, 1]])}},
        }

    def test_find_path_congestion_with_mixed_allocation_returns_correct_metrics(
        self, network_spectrum: dict
    ) -> None:
        """Test path congestion calculation with partially allocated links.

        :param network_spectrum: Network spectrum state
        :type network_spectrum: dict
        """
        # Arrange
        path_list = [1, 2, 3]
        expected_avg_cong = ((4 / 6) + (2 / 6)) / 2
        expected_scaled_cap = (6 - 4) + (6 - 2)

        # Act
        avg_cong, scaled_cap = find_path_congestion(path_list, network_spectrum)

        # Assert
        assert abs(avg_cong - expected_avg_cong) < 1e-5
        assert scaled_cap == expected_scaled_cap

    def test_find_path_congestion_with_empty_spectrum_returns_zero(self) -> None:
        """Test congestion calculation on completely free spectrum."""
        # Arrange
        network_spectrum = {
            (1, 2): {"cores_matrix": {"c": np.array([[0, 0, 0], [0, 0, 0]])}},
        }
        path_list = [1, 2]
        expected_congestion = 0.0

        # Act
        congestion, _ = find_path_congestion(path_list, network_spectrum)

        # Assert
        assert congestion == expected_congestion

    def test_find_path_congestion_with_full_spectrum_returns_one(self) -> None:
        """Test congestion calculation on completely occupied spectrum."""
        # Arrange
        network_spectrum = {
            (1, 2): {"cores_matrix": {"c": np.array([[1, 1, 1], [1, 1, 1]])}},
        }
        path_list = [1, 2]
        expected_congestion = 1.0

        # Act
        congestion, _ = find_path_congestion(path_list, network_spectrum)

        # Assert
        assert congestion == expected_congestion

    def test_find_core_congestion_with_specific_core_returns_correct_percentage(
        self, network_spectrum: dict
    ) -> None:
        """Test congestion calculation for a specific core.

        :param network_spectrum: Network spectrum state
        :type network_spectrum: dict
        """
        # Arrange
        path_list = [1, 2, 3]
        core_index = 0
        expected_core_cong = ((2 / 3) + (1 / 3)) / 2

        # Act
        result = find_core_congestion(core_index, network_spectrum, path_list)

        # Assert
        assert abs(result - expected_core_cong) < 0.01

    def test_classify_congestion_below_cutoff_returns_zero(self) -> None:
        """Test congestion classification below threshold."""
        # Arrange
        current_congestion = 0.2
        congestion_cutoff = 0.3
        expected_index = 0

        # Act
        result = classify_congestion(current_congestion, congestion_cutoff)

        # Assert
        assert result == expected_index

    def test_classify_congestion_above_cutoff_returns_one(self) -> None:
        """Test congestion classification above threshold."""
        # Arrange
        current_congestion = 0.5
        congestion_cutoff = 0.3
        expected_index = 1

        # Act
        result = classify_congestion(current_congestion, congestion_cutoff)

        # Assert
        assert result == expected_index

    def test_classify_congestion_at_cutoff_returns_zero(self) -> None:
        """Test congestion classification exactly at threshold."""
        # Arrange
        current_congestion = 0.3
        congestion_cutoff = 0.3
        expected_index = 0

        # Act
        result = classify_congestion(current_congestion, congestion_cutoff)

        # Assert
        assert result == expected_index


class TestFragmentationMetrics:
    """Tests for spectrum fragmentation calculation functions."""

    @pytest.fixture
    def network_spectrum_256(self) -> dict:
        """Provide network spectrum with 256-slot cores.

        :return: Network spectrum database
        :rtype: dict
        """
        return {
            (1, 2): {"cores_matrix": {"c": [np.zeros(256), np.zeros(256)]}},
            (2, 3): {"cores_matrix": {"c": [np.zeros(256), np.zeros(256)]}},
        }

    def test_find_core_fragmentation_congestion_with_empty_spectrum_returns_zero(
        self, network_spectrum_256: dict
    ) -> None:
        """Test fragmentation and congestion on empty spectrum.

        :param network_spectrum_256: Empty 256-slot spectrum
        :type network_spectrum_256: dict
        """
        # Arrange
        path_list = [1, 2, 3]
        core = 0
        band = "c"

        # Act
        frag, cong = find_core_fragmentation_congestion(
            network_spectrum_256, path_list, core, band
        )

        # Assert
        assert frag == 0
        assert cong == 0

    def test_find_core_fragmentation_congestion_with_invalid_size_raises_error(
        self,
    ) -> None:
        """Test that non-256 slot arrays raise NotImplementedError."""
        # Arrange
        network_spectrum = {
            (1, 2): {"cores_matrix": {"c": [np.zeros(128)]}},
        }
        path_list = [1, 2]
        core = 0
        band = "c"

        # Act & Assert
        with pytest.raises(NotImplementedError, match="Only works for 256"):
            find_core_fragmentation_congestion(network_spectrum, path_list, core, band)

    def test_find_path_fragmentation_with_no_free_slots_returns_one(self) -> None:
        """Test fragmentation with fully occupied spectrum."""
        # Arrange
        network_spectrum = {
            (1, 2): {"cores_matrix": {"c": np.array([[1, 1, 1, 1]])}},
        }
        path_list = [1, 2]
        expected_frag = 1.0

        # Act
        result = find_path_fragmentation(path_list, network_spectrum)

        # Assert
        assert result == expected_frag

    def test_find_path_fragmentation_with_contiguous_free_slots_returns_zero(
        self,
    ) -> None:
        """Test fragmentation with contiguous free spectrum."""
        # Arrange
        network_spectrum = {
            (1, 2): {"cores_matrix": {"c": np.array([[0, 0, 0, 0]])}},
        }
        path_list = [1, 2]
        expected_frag = 0.0

        # Act
        result = find_path_fragmentation(path_list, network_spectrum)

        # Assert
        assert result == expected_frag

    def test_find_path_fragmentation_with_fragmented_spectrum_returns_positive(
        self,
    ) -> None:
        """Test fragmentation with non-contiguous free blocks."""
        # Arrange
        network_spectrum = {
            (1, 2): {"cores_matrix": {"c": np.array([[0, 0, 1, 0, 0]])}},
        }
        path_list = [1, 2]

        # Act
        result = find_path_fragmentation(path_list, network_spectrum)

        # Assert
        assert 0 < result < 1

    def test_find_path_fragmentation_with_empty_path_returns_one(self) -> None:
        """Test fragmentation with empty path list."""
        # Arrange
        network_spectrum = {
            (1, 2): {"cores_matrix": {"c": np.array([[0, 0, 0]])}},
        }
        path_list = [1]

        # Act
        result = find_path_fragmentation(path_list, network_spectrum)

        # Assert
        assert result == 1.0
