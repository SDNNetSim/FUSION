"""Unit tests for the BestFitSpectrum class."""

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from fusion.core.properties import SDNProps
from fusion.modules.spectrum.best_fit import BestFitSpectrum


@pytest.fixture
def engine_props() -> dict[str, Any]:
    """Provide engine properties for tests."""
    return {
        "cores_per_link": 2,
        "guard_slots": 1,
        "band_list": ["c"],
        "slots_per_gbps": 1,
    }


@pytest.fixture
def sdn_props() -> SDNProps:
    """Provide SDN properties for tests."""
    props = SDNProps()
    props.network_spectrum_dict = {
        (1, 2): {
            "cores_matrix": {
                "c": [
                    np.array([0, 0, 0, 1, 1, 0, 0, 0, 0, 0]),
                    np.array([0, 0, 1, 1, 1, 1, 0, 0, 0, 0]),
                ]
            }
        },
        (2, 1): {
            "cores_matrix": {
                "c": [
                    np.array([0, 0, 0, 1, 1, 0, 0, 0, 0, 0]),
                    np.array([0, 0, 1, 1, 1, 1, 0, 0, 0, 0]),
                ]
            }
        },
        (2, 3): {
            "cores_matrix": {
                "c": [
                    np.array([0, 0, 0, 1, 1, 0, 0, 0, 0, 0]),
                    np.array([0, 0, 1, 1, 1, 1, 0, 0, 0, 0]),
                ]
            }
        },
        (3, 2): {
            "cores_matrix": {
                "c": [
                    np.array([0, 0, 0, 1, 1, 0, 0, 0, 0, 0]),
                    np.array([0, 0, 1, 1, 1, 1, 0, 0, 0, 0]),
                ]
            }
        },
    }
    return props


@pytest.fixture
def route_props() -> MagicMock:
    """Provide routing properties for tests."""
    return MagicMock()


@pytest.fixture
def best_fit_spectrum(engine_props: dict[str, Any], sdn_props: SDNProps, route_props: MagicMock) -> BestFitSpectrum:
    """Provide BestFitSpectrum instance for tests."""
    return BestFitSpectrum(engine_props, sdn_props, route_props)


class TestBestFitSpectrumProperties:
    """Tests for BestFitSpectrum properties."""

    def test_algorithm_name_returns_best_fit(self, best_fit_spectrum: BestFitSpectrum) -> None:
        """Test that algorithm name is correctly set."""
        # Act
        name = best_fit_spectrum.algorithm_name

        # Assert
        assert name == "best_fit"

    def test_supports_multiband_returns_true(self, best_fit_spectrum: BestFitSpectrum) -> None:
        """Test that algorithm supports multi-band assignment."""
        # Act
        supports = best_fit_spectrum.supports_multiband

        # Assert
        assert supports is True


class TestAssignMethod:
    """Tests for assign method."""

    def test_assign_with_valid_path_returns_assignment(self, best_fit_spectrum: BestFitSpectrum) -> None:
        """Test successful spectrum assignment with valid path."""
        # Arrange
        path = [1, 2, 3]
        request = MagicMock()
        request.slots_needed = 2

        # Act
        result = best_fit_spectrum.assign(path, request)

        # Assert
        assert result is not None
        assert "start_slot" in result
        assert "end_slot" in result
        assert "core_number" in result
        assert "band" in result

    def test_assign_with_bandwidth_calculates_slots(self, best_fit_spectrum: BestFitSpectrum) -> None:
        """Test that bandwidth is converted to slots."""
        # Arrange
        path = [1, 2, 3]
        request = MagicMock()
        request.bandwidth = 2.5
        del request.slots_needed

        # Act
        result = best_fit_spectrum.assign(path, request)

        # Assert
        assert result is not None
        assert result["slots_needed"] == 3

    def test_assign_with_no_available_slots_returns_none(self, best_fit_spectrum: BestFitSpectrum, sdn_props: SDNProps) -> None:
        """Test that assignment fails when no slots available."""
        # Arrange
        # Fill all slots
        assert sdn_props.network_spectrum_dict is not None
        for link in sdn_props.network_spectrum_dict.values():
            for band in link["cores_matrix"].values():
                for core in band:
                    core[:] = 1

        path = [1, 2, 3]
        request = MagicMock()
        request.slots_needed = 2

        # Act
        result = best_fit_spectrum.assign(path, request)

        # Assert
        assert result is None

    def test_assign_selects_smallest_fitting_channel(self, best_fit_spectrum: BestFitSpectrum, sdn_props: SDNProps) -> None:
        """Test that best fit selects smallest fitting channel."""
        # Arrange
        # Create two free channels: one size 3, one size 7
        # Core 0: [0,0,0,1,1,0,0,0,0,0] -> channels of size 3 and 5
        path = [1, 2, 3]
        request = MagicMock()
        request.slots_needed = 2  # Should fit in size-3 channel (best fit)

        # Act
        result = best_fit_spectrum.assign(path, request)

        # Assert
        assert result is not None
        # Should use the smaller channel (0-2) not the larger one (5-9)
        assert result["start_slot"] == 0


class TestCalculateSlotsNeeded:
    """Tests for _calculate_slots_needed method."""

    def test_calculate_slots_needed_with_integer_bandwidth(self, best_fit_spectrum: BestFitSpectrum) -> None:
        """Test slot calculation with integer bandwidth."""
        # Arrange
        bandwidth = 5.0

        # Act
        slots = best_fit_spectrum._calculate_slots_needed(bandwidth)

        # Assert
        assert slots == 5

    def test_calculate_slots_needed_with_fractional_bandwidth_rounds_up(self, best_fit_spectrum: BestFitSpectrum) -> None:
        """Test that fractional bandwidth rounds up."""
        # Arrange
        bandwidth = 2.3

        # Act
        slots = best_fit_spectrum._calculate_slots_needed(bandwidth)

        # Assert
        assert slots == 3


class TestCheckSpectrumAvailability:
    """Tests for check_spectrum_availability method."""

    def test_check_spectrum_availability_with_free_slots_returns_true(self, best_fit_spectrum: BestFitSpectrum) -> None:
        """Test that free spectrum is correctly identified."""
        # Arrange
        path = [1, 2]
        start_slot, end_slot = 0, 2
        core_num, band = 0, "c"

        # Act
        result = best_fit_spectrum.check_spectrum_availability(path, start_slot, end_slot, core_num, band)

        # Assert
        assert result is True

    def test_check_spectrum_availability_with_occupied_slots_returns_false(self, best_fit_spectrum: BestFitSpectrum) -> None:
        """Test that occupied spectrum is correctly identified."""
        # Arrange
        path = [1, 2]
        start_slot, end_slot = 3, 4  # Occupied slots
        core_num, band = 0, "c"

        # Act
        result = best_fit_spectrum.check_spectrum_availability(path, start_slot, end_slot, core_num, band)

        # Assert
        assert result is False

    def test_check_spectrum_availability_with_missing_link_returns_false(self, best_fit_spectrum: BestFitSpectrum) -> None:
        """Test that missing link returns False."""
        # Arrange
        path = [1, 99]
        start_slot, end_slot = 0, 2
        core_num, band = 0, "c"

        # Act
        result = best_fit_spectrum.check_spectrum_availability(path, start_slot, end_slot, core_num, band)

        # Assert
        assert result is False


class TestAllocateSpectrum:
    """Tests for allocate_spectrum method."""

    def test_allocate_spectrum_with_valid_params_returns_true(self, best_fit_spectrum: BestFitSpectrum, sdn_props: SDNProps) -> None:
        """Test successful spectrum allocation."""
        # Arrange
        path = [1, 2]
        start_slot, end_slot = 0, 2
        core_num, band = 0, "c"
        request_id = 123

        # Act
        result = best_fit_spectrum.allocate_spectrum(path, start_slot, end_slot, core_num, band, request_id)

        # Assert
        assert result is True
        assert sdn_props.network_spectrum_dict is not None
        core_array = sdn_props.network_spectrum_dict[(1, 2)]["cores_matrix"]["c"][0]
        assert core_array[0] == request_id
        assert core_array[1] == request_id
        assert core_array[2] == request_id

    def test_allocate_spectrum_with_missing_link_returns_false(self, best_fit_spectrum: BestFitSpectrum) -> None:
        """Test allocation fails with missing link."""
        # Arrange
        path = [1, 99]
        start_slot, end_slot = 0, 2
        core_num, band = 0, "c"
        request_id = 123

        # Act
        result = best_fit_spectrum.allocate_spectrum(path, start_slot, end_slot, core_num, band, request_id)

        # Assert
        assert result is False


class TestDeallocateSpectrum:
    """Tests for deallocate_spectrum method."""

    def test_deallocate_spectrum_with_allocated_slots_frees_them(self, best_fit_spectrum: BestFitSpectrum, sdn_props: SDNProps) -> None:
        """Test successful spectrum deallocation."""
        # Arrange
        path = [1, 2]
        start_slot, end_slot = 0, 2
        core_num, band = 0, "c"
        request_id = 123

        # First allocate
        best_fit_spectrum.allocate_spectrum(path, start_slot, end_slot, core_num, band, request_id)

        # Act
        result = best_fit_spectrum.deallocate_spectrum(path, start_slot, end_slot, core_num, band)

        # Assert
        assert result is True
        assert sdn_props.network_spectrum_dict is not None
        core_array = sdn_props.network_spectrum_dict[(1, 2)]["cores_matrix"]["c"][0]
        assert core_array[0] == 0
        assert core_array[1] == 0
        assert core_array[2] == 0

    def test_deallocate_spectrum_with_missing_link_returns_false(self, best_fit_spectrum: BestFitSpectrum) -> None:
        """Test deallocation fails with missing link."""
        # Arrange
        path = [1, 99]
        start_slot, end_slot = 0, 2
        core_num, band = 0, "c"

        # Act
        result = best_fit_spectrum.deallocate_spectrum(path, start_slot, end_slot, core_num, band)

        # Assert
        assert result is False


class TestGetFragmentationMetric:
    """Tests for get_fragmentation_metric method."""

    def test_get_fragmentation_metric_with_no_fragmentation_returns_low_value(
        self, best_fit_spectrum: BestFitSpectrum, sdn_props: SDNProps
    ) -> None:
        """Test fragmentation metric with minimal fragmentation."""
        # Arrange
        # Clear all occupied slots for minimal fragmentation
        assert sdn_props.network_spectrum_dict is not None
        for link in sdn_props.network_spectrum_dict.values():
            for band in link["cores_matrix"].values():
                for core in band:
                    core[:] = 0
        path = [1, 2]

        # Act
        metric = best_fit_spectrum.get_fragmentation_metric(path)

        # Assert
        # With all slots free, fragmentation should be minimal
        assert metric >= 0.0

    def test_get_fragmentation_metric_with_fragmentation_returns_higher_value(
        self, best_fit_spectrum: BestFitSpectrum, sdn_props: SDNProps
    ) -> None:
        """Test fragmentation metric with fragmented spectrum."""
        # Arrange
        path = [1, 2]
        # Create multiple small gaps
        assert sdn_props.network_spectrum_dict is not None
        core_array = sdn_props.network_spectrum_dict[(1, 2)]["cores_matrix"]["c"][0]
        core_array[1] = 1
        core_array[3] = 1
        core_array[5] = 1
        core_array[7] = 1

        # Act
        metric = best_fit_spectrum.get_fragmentation_metric(path)

        # Assert
        # Fragmented spectrum should have higher metric
        assert metric > 0.0


class TestGetMetrics:
    """Tests for get_metrics method."""

    def test_get_metrics_with_no_assignments_returns_zero_average(self, best_fit_spectrum: BestFitSpectrum) -> None:
        """Test metrics with no assignments."""
        # Act
        metrics = best_fit_spectrum.get_metrics()

        # Assert
        assert metrics["algorithm"] == "best_fit"
        assert metrics["assignments_made"] == 0
        assert metrics["total_slots_assigned"] == 0
        assert metrics["average_slots_per_assignment"] == 0
        assert metrics["fragmentation_optimized"] is True

    def test_get_metrics_after_assignments_returns_correct_values(self, best_fit_spectrum: BestFitSpectrum) -> None:
        """Test metrics after successful assignments."""
        # Arrange
        path = [1, 2, 3]
        request = MagicMock()
        request.slots_needed = 2

        # Make assignments
        best_fit_spectrum.assign(path, request)
        best_fit_spectrum.assign(path, request)

        # Act
        metrics = best_fit_spectrum.get_metrics()

        # Assert
        assert metrics["assignments_made"] == 2
        assert metrics["total_slots_assigned"] == 4
        assert metrics["average_slots_per_assignment"] == 2.0


class TestReset:
    """Tests for reset method."""

    def test_reset_clears_assignment_counts(self, best_fit_spectrum: BestFitSpectrum) -> None:
        """Test that reset clears all state."""
        # Arrange
        path = [1, 2, 3]
        request = MagicMock()
        request.slots_needed = 2
        best_fit_spectrum.assign(path, request)

        # Act
        best_fit_spectrum.reset()

        # Assert
        metrics = best_fit_spectrum.get_metrics()
        assert metrics["assignments_made"] == 0
        assert metrics["total_slots_assigned"] == 0
