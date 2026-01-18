"""Unit tests for the LightPathSlicingManager class."""

from typing import Any
from unittest.mock import MagicMock

import pytest

from fusion.modules.spectrum.light_path_slicing import LightPathSlicingManager


@pytest.fixture
def engine_props() -> dict[str, Any]:
    """Provide engine properties for tests."""
    # Create a proper topology mock that matches networkx structure
    topology = MagicMock()

    # Mock the __getitem__ to return edge data
    def getitem_side_effect(key: Any) -> MagicMock:
        mock_edge = MagicMock()
        mock_edge.__getitem__ = MagicMock(return_value={"length": 100})
        return mock_edge

    topology.__getitem__ = MagicMock(side_effect=getitem_side_effect)

    return {
        "max_segments": 5,
        "fixed_grid": True,
        "topology": topology,
        "mod_per_bw": {
            "50": {
                "QPSK": {"max_length": 1000},
                "16-QAM": {"max_length": 500},
                "64-QAM": {"max_length": 300},
            },
            "100": {
                "QPSK": {"max_length": 800},
                "16-QAM": {"max_length": 400},
                "64-QAM": {"max_length": 200},
            },
        },
    }


@pytest.fixture
def sdn_props() -> MagicMock:
    """Provide SDN properties for tests."""
    props = MagicMock()
    props.bandwidth = "100"
    props.was_routed = False
    props.is_sliced = False
    props.block_reason = None
    props.number_of_transponders = 0
    props.remaining_bw = 0
    props.was_partially_groomed = False
    props.path_index = 0
    props.was_partially_routed = False
    props.was_new_lp_established = []
    props.get_lightpath_id = MagicMock(return_value=1)
    return props


@pytest.fixture
def spectrum_obj() -> MagicMock:
    """Provide spectrum object for tests."""
    obj = MagicMock()
    obj.spectrum_props = MagicMock()
    obj.spectrum_props.is_free = True
    obj.spectrum_props.path_list = None
    return obj


@pytest.fixture
def slicing_manager(
    engine_props: dict[str, Any], sdn_props: MagicMock, spectrum_obj: MagicMock
) -> LightPathSlicingManager:
    """Provide LightPathSlicingManager instance for tests."""
    return LightPathSlicingManager(engine_props, sdn_props, spectrum_obj)


class TestLightPathSlicingManagerInit:
    """Tests for LightPathSlicingManager initialization."""

    def test_init_sets_properties_correctly(
        self,
        slicing_manager: LightPathSlicingManager,
        engine_props: dict[str, Any],
        sdn_props: MagicMock,
        spectrum_obj: MagicMock,
    ) -> None:
        """Test that initialization sets all properties."""
        # Assert
        assert slicing_manager.engine_props == engine_props
        assert slicing_manager.sdn_props == sdn_props
        assert slicing_manager.spectrum_obj == spectrum_obj


class TestAllocateSlicing:
    """Tests for allocate_slicing method."""

    def test_allocate_slicing_with_successful_allocation_yields_allocate(
        self, slicing_manager: LightPathSlicingManager
    ) -> None:
        """Test successful slicing allocation."""
        # Arrange
        num_segments = 2
        mod_format = "QPSK"
        path_list = [1, 2, 3]
        bandwidth = "50"

        # Act
        results = list(
            slicing_manager.allocate_slicing(
                num_segments, mod_format, path_list, bandwidth
            )
        )

        # Assert
        assert len(results) == 2
        assert all(action == "allocate" for action, _ in results)
        assert all(bw == bandwidth for _, bw in results)

    def test_allocate_slicing_with_no_spectrum_yields_release(
        self, slicing_manager: LightPathSlicingManager, spectrum_obj: MagicMock
    ) -> None:
        """Test slicing allocation failure."""
        # Arrange
        spectrum_obj.spectrum_props.is_free = False
        num_segments = 2
        mod_format = "QPSK"
        path_list = [1, 2, 3]
        bandwidth = "50"

        # Act
        results = list(
            slicing_manager.allocate_slicing(
                num_segments, mod_format, path_list, bandwidth
            )
        )

        # Assert
        assert len(results) == 1
        assert results[0][0] == "release"


class TestHandleStaticSlicing:
    """Tests for handle_static_slicing method."""

    def test_handle_static_slicing_with_valid_bandwidth_returns_true(
        self, slicing_manager: LightPathSlicingManager, sdn_props: MagicMock
    ) -> None:
        """Test successful static slicing."""
        # Arrange
        path_list = [1, 2, 3]
        forced_segments = -1  # Auto-calculate

        # Act
        gen = slicing_manager.handle_static_slicing(path_list, forced_segments)
        results = list(gen)

        # Assert
        assert len(results) > 0
        assert sdn_props.is_sliced is True

    def test_handle_static_slicing_with_excessive_segments_returns_false(
        self,
        slicing_manager: LightPathSlicingManager,
        sdn_props: MagicMock,
        engine_props: dict[str, Any],
    ) -> None:
        """Test that excessive segments are blocked."""
        # Arrange
        sdn_props.bandwidth = "500"  # Would need many segments
        engine_props["max_segments"] = 2  # Low limit
        path_list = [1, 2, 3]
        forced_segments = -1

        # Act
        gen = slicing_manager.handle_static_slicing(path_list, forced_segments)
        list(gen)  # Consume generator

        # Assert
        assert sdn_props.block_reason == "max_segments"


class TestHandleStaticSlicingDirect:
    """Tests for handle_static_slicing_direct method."""

    def test_handle_static_slicing_direct_with_successful_allocation_returns_true(
        self, slicing_manager: LightPathSlicingManager, sdn_props: MagicMock
    ) -> None:
        """Test successful direct static slicing."""
        # Arrange
        path_list = [1, 2, 3]
        forced_segments = -1
        sdn_controller = MagicMock()

        # Act
        result = slicing_manager.handle_static_slicing_direct(
            path_list, forced_segments, sdn_controller
        )

        # Assert
        assert result is True
        assert sdn_props.is_sliced is True

    def test_handle_static_slicing_direct_with_no_spectrum_returns_false(
        self,
        slicing_manager: LightPathSlicingManager,
        sdn_props: MagicMock,
        spectrum_obj: MagicMock,
    ) -> None:
        """Test direct static slicing with no spectrum available."""
        # Arrange
        spectrum_obj.spectrum_props.is_free = False
        path_list = [1, 2, 3]
        forced_segments = -1
        sdn_controller = MagicMock()

        # Configure mock to set congestion state like real controller
        def set_congestion_state(*args: Any, **kwargs: Any) -> None:
            sdn_props.was_routed = False
            sdn_props.block_reason = "congestion"

        sdn_controller._handle_congestion = MagicMock(side_effect=set_congestion_state)

        # Act
        result = slicing_manager.handle_static_slicing_direct(
            path_list, forced_segments, sdn_controller
        )

        # Assert
        assert result is False
        assert sdn_props.block_reason == "congestion"


class TestHandleDynamicSlicing:
    """Tests for handle_dynamic_slicing method."""

    def test_handle_dynamic_slicing_with_non_fixed_grid_raises_error(
        self, slicing_manager: LightPathSlicingManager, engine_props: dict[str, Any]
    ) -> None:
        """Test that non-fixed grid raises NotImplementedError."""
        # Arrange
        engine_props["fixed_grid"] = False
        path_list = [1, 2, 3]
        path_index = 0
        forced_segments = -1

        # Act & Assert
        gen = slicing_manager.handle_dynamic_slicing(
            path_list, path_index, forced_segments
        )
        with pytest.raises(
            NotImplementedError,
            match="Dynamic slicing for non-fixed grid is not implemented",
        ):
            list(gen)

    def test_handle_dynamic_slicing_with_successful_allocation_yields_allocate(
        self, slicing_manager: LightPathSlicingManager, spectrum_obj: MagicMock
    ) -> None:
        """Test successful dynamic slicing."""
        # Arrange
        spectrum_obj.get_spectrum_dynamic_slicing = MagicMock(return_value=(None, 50))
        path_list = [1, 2, 3]
        path_index = 0
        forced_segments = -1

        # Act
        results = list(
            slicing_manager.handle_dynamic_slicing(
                path_list, path_index, forced_segments
            )
        )

        # Assert
        assert len(results) > 0
        assert any(action == "allocate" for action, _ in results)


class TestHandleDynamicSlicingDirect:
    """Tests for handle_dynamic_slicing_direct method."""

    def test_handle_dynamic_slicing_direct_with_successful_allocation_returns_true(
        self,
        slicing_manager: LightPathSlicingManager,
        sdn_props: MagicMock,
        spectrum_obj: MagicMock,
    ) -> None:
        """Test successful direct dynamic slicing."""
        # Arrange
        spectrum_obj.get_spectrum_dynamic_slicing = MagicMock(return_value=(None, 50))
        spectrum_obj._update_lightpath_status = MagicMock()
        # Ensure sdn_props has proper int values for comparisons
        sdn_props.bandwidth = "100"
        sdn_props.was_partially_groomed = False
        sdn_props.remaining_bw = 0
        path_list = [1, 2, 3]
        path_index = 0
        forced_segments = -1
        sdn_controller = MagicMock()
        sdn_controller._check_snr_after_allocation = MagicMock(return_value=True)

        # Act
        result = slicing_manager.handle_dynamic_slicing_direct(
            path_list, path_index, forced_segments, sdn_controller
        )

        # Assert
        assert result is True
        assert sdn_props.is_sliced is True

    def test_handle_dynamic_slicing_direct_with_non_fixed_grid_processes(
        self,
        slicing_manager: LightPathSlicingManager,
        sdn_props: MagicMock,
        spectrum_obj: MagicMock,
        engine_props: dict[str, Any],
    ) -> None:
        """Test that non-fixed grid processes using flex-grid slicing."""
        # Arrange
        engine_props["fixed_grid"] = False
        sdn_props.bandwidth = "100"
        sdn_props.was_partially_groomed = False
        sdn_props.remaining_bw = 0
        spectrum_obj.spectrum_props.is_free = True
        spectrum_obj.spectrum_props.modulation = "QPSK"
        # Mock get_spectrum_dynamic_slicing to return (mod_format, bandwidth) tuple
        spectrum_obj.get_spectrum_dynamic_slicing = MagicMock(
            return_value=("QPSK", 50)
        )
        spectrum_obj._update_lightpath_status = MagicMock()
        path_list = [1, 2, 3]
        path_index = 0
        forced_segments = -1
        sdn_controller = MagicMock()
        sdn_controller._check_snr_after_allocation = MagicMock(return_value=True)

        # Act - flex-grid slicing is now implemented
        result = slicing_manager.handle_dynamic_slicing_direct(
            path_list, path_index, forced_segments, sdn_controller
        )

        # Assert - should return a boolean (True/False) without raising error
        assert isinstance(result, bool)


class TestAllocateSlicingDirect:
    """Tests for allocate_slicing_direct method."""

    def test_allocate_slicing_direct_with_successful_allocation(
        self, slicing_manager: LightPathSlicingManager, sdn_props: MagicMock
    ) -> None:
        """Test successful direct slicing allocation."""
        # Arrange
        num_segments = 2
        mod_format = "QPSK"
        path_list = [1, 2, 3]
        bandwidth = "50"
        sdn_controller = MagicMock()

        # Act
        slicing_manager.allocate_slicing_direct(
            num_segments, mod_format, path_list, bandwidth, sdn_controller
        )

        # Assert
        assert sdn_props.number_of_transponders == 2
        assert sdn_controller.allocate.call_count == 2
        assert sdn_controller._update_req_stats.call_count == 2

    def test_allocate_slicing_direct_with_failed_allocation(
        self,
        slicing_manager: LightPathSlicingManager,
        sdn_props: MagicMock,
        spectrum_obj: MagicMock,
    ) -> None:
        """Test direct slicing allocation failure."""
        # Arrange
        spectrum_obj.spectrum_props.is_free = False
        num_segments = 2
        mod_format = "QPSK"
        path_list = [1, 2, 3]
        bandwidth = "50"
        sdn_controller = MagicMock()

        # Configure mock to set congestion state like real controller
        def set_congestion_state(*args: Any, **kwargs: Any) -> None:
            sdn_props.was_routed = False
            sdn_props.block_reason = "congestion"
            sdn_controller.release()

        sdn_controller._handle_congestion = MagicMock(side_effect=set_congestion_state)

        # Act
        slicing_manager.allocate_slicing_direct(
            num_segments, mod_format, path_list, bandwidth, sdn_controller
        )

        # Assert
        assert sdn_props.was_routed is False
        assert sdn_props.block_reason == "congestion"
        assert sdn_controller.release.call_count == 1
