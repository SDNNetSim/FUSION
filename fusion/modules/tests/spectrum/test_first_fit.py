"""Unit tests for the FirstFitSpectrum class."""

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from fusion.core.properties import SDNProps
from fusion.modules.spectrum.first_fit import FirstFitSpectrum


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
                    np.array([0, 0, 0, 0, 0, 1, 1, 1, 0, 0]),
                    np.array([0, 0, 1, 1, 1, 1, 0, 0, 0, 0]),
                ]
            }
        },
        (2, 3): {
            "cores_matrix": {
                "c": [
                    np.array([0, 0, 0, 0, 0, 1, 1, 1, 0, 0]),
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
def first_fit_spectrum(
    engine_props: dict[str, Any], sdn_props: SDNProps, route_props: MagicMock
) -> FirstFitSpectrum:
    """Provide FirstFitSpectrum instance for tests."""
    return FirstFitSpectrum(engine_props, sdn_props, route_props)


class TestFirstFitSpectrumProperties:
    """Tests for FirstFitSpectrum properties."""

    def test_algorithm_name_returns_first_fit(
        self, first_fit_spectrum: FirstFitSpectrum
    ) -> None:
        """Test that algorithm name is correctly set."""
        # Act
        name = first_fit_spectrum.algorithm_name

        # Assert
        assert name == "first_fit"

    def test_supports_multiband_returns_true(
        self, first_fit_spectrum: FirstFitSpectrum
    ) -> None:
        """Test that algorithm supports multi-band assignment."""
        # Act
        supports = first_fit_spectrum.supports_multiband

        # Assert
        assert supports is True


class TestAssignMethod:
    """Tests for assign method."""

    def test_assign_with_valid_path_returns_assignment(
        self, first_fit_spectrum: FirstFitSpectrum
    ) -> None:
        """Test successful spectrum assignment with valid path."""
        # Arrange
        path = [1, 2, 3]
        request = MagicMock()
        request.slots_needed = 2

        # Act
        result = first_fit_spectrum.assign(path, request)

        # Assert
        assert result is not None
        assert "start_slot" in result
        assert "end_slot" in result
        assert "core_number" in result
        assert "band" in result
        assert result["start_slot"] == 0

    def test_assign_with_bandwidth_calculates_slots(
        self, first_fit_spectrum: FirstFitSpectrum
    ) -> None:
        """Test that bandwidth is converted to slots."""
        # Arrange
        path = [1, 2, 3]
        request = MagicMock()
        request.bandwidth = 2.5
        del request.slots_needed  # Remove slots_needed to force bandwidth calculation

        # Act
        result = first_fit_spectrum.assign(path, request)

        # Assert
        assert result is not None
        assert result["slots_needed"] == 3  # ceil(2.5 * 1)

    def test_assign_with_empty_path_raises_error(
        self, first_fit_spectrum: FirstFitSpectrum
    ) -> None:
        """Test that empty path raises ValueError."""
        # Arrange
        path: list[Any] = []
        request = MagicMock()

        # Act & Assert
        with pytest.raises(ValueError, match="Path cannot be empty"):
            first_fit_spectrum.assign(path, request)

    def test_assign_with_none_request_raises_error(
        self, first_fit_spectrum: FirstFitSpectrum
    ) -> None:
        """Test that None request raises ValueError."""
        # Arrange
        path = [1, 2]

        # Act & Assert
        with pytest.raises(ValueError, match="Request cannot be None"):
            first_fit_spectrum.assign(path, None)  # type: ignore[arg-type]

    def test_assign_with_no_available_slots_returns_none(
        self, first_fit_spectrum: FirstFitSpectrum, sdn_props: SDNProps
    ) -> None:
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
        result = first_fit_spectrum.assign(path, request)

        # Assert
        assert result is None


class TestCalculateSlotsNeeded:
    """Tests for _calculate_slots_needed method."""

    def test_calculate_slots_needed_with_integer_bandwidth(
        self, first_fit_spectrum: FirstFitSpectrum
    ) -> None:
        """Test slot calculation with integer bandwidth."""
        # Arrange
        bandwidth = 5.0

        # Act
        slots = first_fit_spectrum._calculate_slots_needed(bandwidth)

        # Assert
        assert slots == 5

    def test_calculate_slots_needed_with_fractional_bandwidth_rounds_up(
        self, first_fit_spectrum: FirstFitSpectrum
    ) -> None:
        """Test that fractional bandwidth rounds up."""
        # Arrange
        bandwidth = 2.3

        # Act
        slots = first_fit_spectrum._calculate_slots_needed(bandwidth)

        # Assert
        assert slots == 3


class TestCheckSpectrumAvailability:
    """Tests for check_spectrum_availability method."""

    def test_check_spectrum_availability_with_free_slots_returns_true(
        self, first_fit_spectrum: FirstFitSpectrum
    ) -> None:
        """Test that free spectrum is correctly identified."""
        # Arrange
        path = [1, 2]
        start_slot, end_slot = 0, 2
        core_num, band = 0, "c"

        # Act
        result = first_fit_spectrum.check_spectrum_availability(
            path, start_slot, end_slot, core_num, band
        )

        # Assert
        assert result is True

    def test_check_spectrum_availability_with_occupied_slots_returns_false(
        self, first_fit_spectrum: FirstFitSpectrum
    ) -> None:
        """Test that occupied spectrum is correctly identified."""
        # Arrange
        path = [1, 2]
        start_slot, end_slot = 5, 7  # Occupied slots
        core_num, band = 0, "c"

        # Act
        result = first_fit_spectrum.check_spectrum_availability(
            path, start_slot, end_slot, core_num, band
        )

        # Assert
        assert result is False

    def test_check_spectrum_availability_with_missing_link_returns_false(
        self, first_fit_spectrum: FirstFitSpectrum
    ) -> None:
        """Test that missing link returns False."""
        # Arrange
        path = [1, 99]  # Non-existent link
        start_slot, end_slot = 0, 2
        core_num, band = 0, "c"

        # Act
        result = first_fit_spectrum.check_spectrum_availability(
            path, start_slot, end_slot, core_num, band
        )

        # Assert
        assert result is False


class TestAllocateSpectrum:
    """Tests for allocate_spectrum method."""

    def test_allocate_spectrum_with_valid_params_returns_true(
        self, first_fit_spectrum: FirstFitSpectrum, sdn_props: SDNProps
    ) -> None:
        """Test successful spectrum allocation."""
        # Arrange
        path = [1, 2]
        start_slot, end_slot = 0, 2
        core_num, band = 0, "c"
        request_id = 123

        # Act
        result = first_fit_spectrum.allocate_spectrum(
            path, start_slot, end_slot, core_num, band, request_id
        )

        # Assert
        assert result is True
        # Verify slots are marked with request ID
        assert sdn_props.network_spectrum_dict is not None
        core_array = sdn_props.network_spectrum_dict[(1, 2)]["cores_matrix"]["c"][0]
        assert core_array[0] == request_id
        assert core_array[1] == request_id
        assert core_array[2] == request_id

    def test_allocate_spectrum_with_missing_link_returns_false(
        self, first_fit_spectrum: FirstFitSpectrum
    ) -> None:
        """Test allocation fails with missing link."""
        # Arrange
        path = [1, 99]
        start_slot, end_slot = 0, 2
        core_num, band = 0, "c"
        request_id = 123

        # Act
        result = first_fit_spectrum.allocate_spectrum(
            path, start_slot, end_slot, core_num, band, request_id
        )

        # Assert
        assert result is False


class TestDeallocateSpectrum:
    """Tests for deallocate_spectrum method."""

    def test_deallocate_spectrum_with_allocated_slots_frees_them(
        self, first_fit_spectrum: FirstFitSpectrum, sdn_props: SDNProps
    ) -> None:
        """Test successful spectrum deallocation."""
        # Arrange
        path = [1, 2]
        start_slot, end_slot = 0, 2
        core_num, band = 0, "c"
        request_id = 123

        # First allocate
        first_fit_spectrum.allocate_spectrum(
            path, start_slot, end_slot, core_num, band, request_id
        )

        # Act - Deallocate
        result = first_fit_spectrum.deallocate_spectrum(
            path, start_slot, end_slot, core_num, band
        )

        # Assert
        assert result is True
        assert sdn_props.network_spectrum_dict is not None
        core_array = sdn_props.network_spectrum_dict[(1, 2)]["cores_matrix"]["c"][0]
        assert core_array[0] == 0
        assert core_array[1] == 0
        assert core_array[2] == 0

    def test_deallocate_spectrum_with_missing_link_returns_false(
        self, first_fit_spectrum: FirstFitSpectrum
    ) -> None:
        """Test deallocation fails with missing link."""
        # Arrange
        path = [1, 99]
        start_slot, end_slot = 0, 2
        core_num, band = 0, "c"

        # Act
        result = first_fit_spectrum.deallocate_spectrum(
            path, start_slot, end_slot, core_num, band
        )

        # Assert
        assert result is False


class TestGetFragmentationMetric:
    """Tests for get_fragmentation_metric method."""

    def test_get_fragmentation_metric_with_minimal_fragmentation(
        self, first_fit_spectrum: FirstFitSpectrum, sdn_props: SDNProps
    ) -> None:
        """Test fragmentation metric with minimal fragmentation."""
        # Arrange
        # Clear all slots - with 2 cores, metric = 1 - (1/2) = 0.5
        assert sdn_props.network_spectrum_dict is not None
        for link in sdn_props.network_spectrum_dict.values():
            for band in link["cores_matrix"].values():
                for core in band:
                    core[:] = 0
        path = [1, 2]

        # Act
        metric = first_fit_spectrum.get_fragmentation_metric(path)

        # Assert
        # With 2 free blocks (one per core), fragmentation = 1 - (1/2) = 0.5
        assert metric == 0.5

    def test_get_fragmentation_metric_with_fragmentation_returns_nonzero(
        self, first_fit_spectrum: FirstFitSpectrum, sdn_props: SDNProps
    ) -> None:
        """Test fragmentation metric with fragmented spectrum."""
        # Arrange
        path = [1, 2]
        # Create fragmentation: occupy some slots
        assert sdn_props.network_spectrum_dict is not None
        core_array = sdn_props.network_spectrum_dict[(1, 2)]["cores_matrix"]["c"][0]
        core_array[2:5] = 1  # Create gap

        # Act
        metric = first_fit_spectrum.get_fragmentation_metric(path)

        # Assert
        assert metric > 0.0  # Should show fragmentation


class TestGetMetrics:
    """Tests for get_metrics method."""

    def test_get_metrics_with_no_assignments_returns_zero_average(
        self, first_fit_spectrum: FirstFitSpectrum
    ) -> None:
        """Test metrics with no assignments."""
        # Act
        metrics = first_fit_spectrum.get_metrics()

        # Assert
        assert metrics["algorithm"] == "first_fit"
        assert metrics["assignments_made"] == 0
        assert metrics["total_slots_assigned"] == 0
        assert metrics["average_slots_per_assignment"] == 0

    def test_get_metrics_after_assignments_returns_correct_values(
        self, first_fit_spectrum: FirstFitSpectrum
    ) -> None:
        """Test metrics after successful assignments."""
        # Arrange
        path = [1, 2, 3]
        request = MagicMock()
        request.slots_needed = 2

        # Make two assignments
        first_fit_spectrum.assign(path, request)
        first_fit_spectrum.assign(path, request)

        # Act
        metrics = first_fit_spectrum.get_metrics()

        # Assert
        assert metrics["assignments_made"] == 2
        assert metrics["total_slots_assigned"] == 4
        assert metrics["average_slots_per_assignment"] == 2.0


class TestReset:
    """Tests for reset method."""

    def test_reset_clears_assignment_counts(
        self, first_fit_spectrum: FirstFitSpectrum
    ) -> None:
        """Test that reset clears all state."""
        # Arrange
        path = [1, 2, 3]
        request = MagicMock()
        request.slots_needed = 2
        first_fit_spectrum.assign(path, request)

        # Act
        first_fit_spectrum.reset()

        # Assert
        metrics = first_fit_spectrum.get_metrics()
        assert metrics["assignments_made"] == 0
        assert metrics["total_slots_assigned"] == 0
