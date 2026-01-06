"""Tests for RLSimulationAdapter.

Phase: P4.1 - RLSimulationAdapter Scaffolding
Chunk: 2 - Adapter skeleton
"""

from unittest.mock import MagicMock

import pytest

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
