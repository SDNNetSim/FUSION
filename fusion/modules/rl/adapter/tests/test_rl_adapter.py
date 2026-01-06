"""Tests for RLSimulationAdapter.

Phase: P4.1 - RLSimulationAdapter Scaffolding
Chunk: 2 - Adapter skeleton
Chunk: 3 - get_path_options method
"""

from unittest.mock import MagicMock

import pytest

from fusion.modules.rl.adapter.path_option import PathOption
from fusion.modules.rl.adapter.rl_adapter import RLSimulationAdapter


class TestRLSimulationAdapterInit:
    """Tests for RLSimulationAdapter initialization."""

    def test_init_stores_orchestrator_reference(self) -> None:
        """Adapter should store reference to orchestrator."""
        mock_orchestrator = MagicMock()
        mock_orchestrator.routing = MagicMock()
        mock_orchestrator.spectrum = MagicMock()

        adapter = RLSimulationAdapter(mock_orchestrator)

        assert adapter.orchestrator is mock_orchestrator

    def test_init_raises_for_none_orchestrator(self) -> None:
        """Adapter should raise ValueError if orchestrator is None."""
        with pytest.raises(ValueError, match="orchestrator cannot be None"):
            RLSimulationAdapter(None)  # type: ignore[arg-type]


class TestPipelineIdentity:
    """Tests for pipeline identity invariant.

    Critical invariant: adapter.routing IS orchestrator.routing (same object).
    This ensures RL code uses the exact same pipelines as non-RL simulation.
    """

    def test_routing_pipeline_identity(self) -> None:
        """adapter.routing should BE orchestrator.routing (same object)."""
        mock_orchestrator = MagicMock()
        mock_routing = MagicMock()
        mock_orchestrator.routing = mock_routing
        mock_orchestrator.spectrum = MagicMock()

        adapter = RLSimulationAdapter(mock_orchestrator)

        # Identity check - same object, not just equal
        assert adapter.routing is mock_routing
        assert adapter.routing is mock_orchestrator.routing

    def test_spectrum_pipeline_identity(self) -> None:
        """adapter.spectrum should BE orchestrator.spectrum (same object)."""
        mock_orchestrator = MagicMock()
        mock_spectrum = MagicMock()
        mock_orchestrator.routing = MagicMock()
        mock_orchestrator.spectrum = mock_spectrum

        adapter = RLSimulationAdapter(mock_orchestrator)

        # Identity check - same object, not just equal
        assert adapter.spectrum is mock_spectrum
        assert adapter.spectrum is mock_orchestrator.spectrum

    def test_pipelines_not_copied(self) -> None:
        """Pipelines should not be copied, ensuring shared state."""
        mock_orchestrator = MagicMock()
        mock_routing = MagicMock()
        mock_spectrum = MagicMock()
        mock_orchestrator.routing = mock_routing
        mock_orchestrator.spectrum = mock_spectrum

        adapter = RLSimulationAdapter(mock_orchestrator)

        # Verify we haven't created copies
        assert id(adapter.routing) == id(mock_orchestrator.routing)
        assert id(adapter.spectrum) == id(mock_orchestrator.spectrum)


class TestAdapterProperties:
    """Tests for adapter property accessors."""

    def test_routing_property_returns_pipeline(self) -> None:
        """routing property should return the routing pipeline."""
        mock_orchestrator = MagicMock()
        mock_routing = MagicMock()
        mock_routing.some_method = MagicMock(return_value="routing_result")
        mock_orchestrator.routing = mock_routing
        mock_orchestrator.spectrum = MagicMock()

        adapter = RLSimulationAdapter(mock_orchestrator)

        # Can call methods on the returned pipeline
        result = adapter.routing.some_method()
        assert result == "routing_result"

    def test_spectrum_property_returns_pipeline(self) -> None:
        """spectrum property should return the spectrum pipeline."""
        mock_orchestrator = MagicMock()
        mock_spectrum = MagicMock()
        mock_spectrum.some_method = MagicMock(return_value="spectrum_result")
        mock_orchestrator.routing = MagicMock()
        mock_orchestrator.spectrum = mock_spectrum

        adapter = RLSimulationAdapter(mock_orchestrator)

        # Can call methods on the returned pipeline
        result = adapter.spectrum.some_method()
        assert result == "spectrum_result"

    def test_orchestrator_property_returns_orchestrator(self) -> None:
        """orchestrator property should return the orchestrator."""
        mock_orchestrator = MagicMock()
        mock_orchestrator.routing = MagicMock()
        mock_orchestrator.spectrum = MagicMock()

        adapter = RLSimulationAdapter(mock_orchestrator)

        assert adapter.orchestrator is mock_orchestrator


class TestGetPathOptions:
    """Tests for get_path_options method.

    Verifies that get_path_options:
    - Calls routing pipeline to get candidate paths
    - Calls spectrum pipeline to check feasibility
    - Returns correct number of PathOption instances
    - Populates PathOption fields correctly
    """

    def _create_mock_route_result(
        self,
        paths: list[tuple[str, ...]],
        weights_km: list[float],
        modulations: list[tuple[str, ...]],
    ) -> MagicMock:
        """Create a mock RouteResult."""
        mock = MagicMock()
        mock.paths = tuple(paths)
        mock.weights_km = tuple(weights_km)
        mock.modulations = tuple(modulations)
        mock.is_empty = len(paths) == 0
        return mock

    def _create_mock_spectrum_result(
        self,
        is_free: bool = True,
        slots_needed: int = 4,
        start_slot: int = 0,
        end_slot: int = 4,
        core: int = 0,
        band: str = "c",
    ) -> MagicMock:
        """Create a mock SpectrumResult."""
        mock = MagicMock()
        mock.is_free = is_free
        mock.slots_needed = slots_needed
        mock.start_slot = start_slot if is_free else None
        mock.end_slot = end_slot if is_free else None
        mock.core = core if is_free else None
        mock.band = band if is_free else None
        return mock

    def _create_adapter_with_mocks(self) -> tuple[RLSimulationAdapter, MagicMock, MagicMock]:
        """Create an adapter with mock orchestrator and pipelines."""
        mock_routing = MagicMock()
        mock_spectrum = MagicMock()
        mock_orchestrator = MagicMock()
        mock_orchestrator.routing = mock_routing
        mock_orchestrator.spectrum = mock_spectrum

        adapter = RLSimulationAdapter(mock_orchestrator)
        return adapter, mock_routing, mock_spectrum

    def test_returns_empty_list_when_no_routes(self) -> None:
        """get_path_options should return empty list when routing finds no paths."""
        adapter, mock_routing, _ = self._create_adapter_with_mocks()

        # Setup routing to return empty result
        mock_routing.find_routes.return_value = self._create_mock_route_result(
            paths=[], weights_km=[], modulations=[]
        )

        mock_request = MagicMock()
        mock_request.source = "A"
        mock_request.destination = "B"
        mock_request.bandwidth_gbps = 100

        mock_network_state = MagicMock()

        options = adapter.get_path_options(mock_request, mock_network_state)

        assert options == []
        mock_routing.find_routes.assert_called_once()

    def test_returns_correct_number_of_options(self) -> None:
        """get_path_options should return one PathOption per path."""
        adapter, mock_routing, mock_spectrum = self._create_adapter_with_mocks()

        # Setup routing to return 3 paths
        mock_routing.find_routes.return_value = self._create_mock_route_result(
            paths=[("A", "B"), ("A", "C", "B"), ("A", "D", "C", "B")],
            weights_km=[100.0, 150.0, 200.0],
            modulations=[("QPSK",), ("QPSK", "16-QAM"), ("QPSK",)],
        )

        # Setup spectrum to return feasible for all
        mock_spectrum.find_spectrum.return_value = self._create_mock_spectrum_result(
            is_free=True, slots_needed=4
        )

        mock_request = MagicMock()
        mock_request.source = "A"
        mock_request.destination = "B"
        mock_request.bandwidth_gbps = 100

        mock_network_state = MagicMock()

        options = adapter.get_path_options(mock_request, mock_network_state)

        assert len(options) == 3
        assert all(isinstance(opt, PathOption) for opt in options)

    def test_path_option_fields_populated_correctly(self) -> None:
        """PathOption fields should match route and spectrum results."""
        adapter, mock_routing, mock_spectrum = self._create_adapter_with_mocks()

        # Setup routing with specific values
        mock_routing.find_routes.return_value = self._create_mock_route_result(
            paths=[("A", "C", "B")],
            weights_km=[150.5],
            modulations=[("QPSK",)],
        )

        # Setup spectrum with specific values
        mock_spectrum.find_spectrum.return_value = self._create_mock_spectrum_result(
            is_free=True,
            slots_needed=4,
            start_slot=10,
            end_slot=14,
            core=0,
            band="c",
        )

        mock_request = MagicMock()
        mock_request.source = "A"
        mock_request.destination = "B"
        mock_request.bandwidth_gbps = 100

        mock_network_state = MagicMock()

        options = adapter.get_path_options(mock_request, mock_network_state)

        assert len(options) == 1
        opt = options[0]

        assert opt.path_index == 0
        assert opt.path == ("A", "C", "B")
        assert opt.weight_km == 150.5
        assert opt.num_hops == 2
        assert opt.modulation == "QPSK"
        assert opt.slots_needed == 4
        assert opt.is_feasible is True
        assert opt.spectrum_start == 10
        assert opt.spectrum_end == 14
        assert opt.core_index == 0
        assert opt.band == "c"

    def test_infeasible_path_has_correct_fields(self) -> None:
        """Infeasible paths should have is_feasible=False and None spectrum fields."""
        adapter, mock_routing, mock_spectrum = self._create_adapter_with_mocks()

        mock_routing.find_routes.return_value = self._create_mock_route_result(
            paths=[("A", "B")],
            weights_km=[100.0],
            modulations=[("QPSK",)],
        )

        # Spectrum not free
        mock_spectrum.find_spectrum.return_value = self._create_mock_spectrum_result(
            is_free=False,
            slots_needed=4,
        )

        mock_request = MagicMock()
        mock_request.source = "A"
        mock_request.destination = "B"
        mock_request.bandwidth_gbps = 100

        mock_network_state = MagicMock()

        options = adapter.get_path_options(mock_request, mock_network_state)

        assert len(options) == 1
        opt = options[0]

        assert opt.is_feasible is False
        assert opt.spectrum_start is None
        assert opt.spectrum_end is None
        assert opt.core_index is None
        assert opt.band is None

    def test_calls_routing_pipeline_with_correct_args(self) -> None:
        """Routing pipeline should be called with request parameters."""
        adapter, mock_routing, mock_spectrum = self._create_adapter_with_mocks()

        mock_routing.find_routes.return_value = self._create_mock_route_result(
            paths=[], weights_km=[], modulations=[]
        )

        mock_request = MagicMock()
        mock_request.source = "NodeA"
        mock_request.destination = "NodeB"
        mock_request.bandwidth_gbps = 200

        mock_network_state = MagicMock()

        adapter.get_path_options(mock_request, mock_network_state)

        mock_routing.find_routes.assert_called_once_with(
            source="NodeA",
            destination="NodeB",
            bandwidth_gbps=200,
            network_state=mock_network_state,
        )

    def test_calls_spectrum_pipeline_for_each_path(self) -> None:
        """Spectrum pipeline should be called once per path."""
        adapter, mock_routing, mock_spectrum = self._create_adapter_with_mocks()

        mock_routing.find_routes.return_value = self._create_mock_route_result(
            paths=[("A", "B"), ("A", "C", "B")],
            weights_km=[100.0, 150.0],
            modulations=[("QPSK",), ("16-QAM",)],
        )

        mock_spectrum.find_spectrum.return_value = self._create_mock_spectrum_result(
            is_free=True
        )

        mock_request = MagicMock()
        mock_request.source = "A"
        mock_request.destination = "B"
        mock_request.bandwidth_gbps = 100

        mock_network_state = MagicMock()

        adapter.get_path_options(mock_request, mock_network_state)

        assert mock_spectrum.find_spectrum.call_count == 2

    def test_path_without_modulation_is_infeasible(self) -> None:
        """Paths with no modulation should have is_feasible=False."""
        adapter, mock_routing, mock_spectrum = self._create_adapter_with_mocks()

        mock_routing.find_routes.return_value = self._create_mock_route_result(
            paths=[("A", "B")],
            weights_km=[100.0],
            modulations=[()],  # Empty modulations
        )

        mock_request = MagicMock()
        mock_request.source = "A"
        mock_request.destination = "B"
        mock_request.bandwidth_gbps = 100

        mock_network_state = MagicMock()

        options = adapter.get_path_options(mock_request, mock_network_state)

        assert len(options) == 1
        assert options[0].is_feasible is False
        assert options[0].modulation is None
        # Spectrum should not be called if no modulation
        mock_spectrum.find_spectrum.assert_not_called()

    def test_congestion_and_available_slots_defaults(self) -> None:
        """Congestion should be 0.0 and available_slots 1.0 (placeholders)."""
        adapter, mock_routing, mock_spectrum = self._create_adapter_with_mocks()

        mock_routing.find_routes.return_value = self._create_mock_route_result(
            paths=[("A", "B")],
            weights_km=[100.0],
            modulations=[("QPSK",)],
        )

        mock_spectrum.find_spectrum.return_value = self._create_mock_spectrum_result(
            is_free=True
        )

        mock_request = MagicMock()
        mock_request.source = "A"
        mock_request.destination = "B"
        mock_request.bandwidth_gbps = 100

        mock_network_state = MagicMock()

        options = adapter.get_path_options(mock_request, mock_network_state)

        assert options[0].congestion == 0.0
        assert options[0].available_slots == 1.0
