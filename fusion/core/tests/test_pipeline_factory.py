"""
Unit tests for PipelineFactory and PipelineSet.

Tests cover:
- PipelineSet creation and validation
- PipelineFactory pipeline selection logic
- Config-driven pipeline creation

Phase: P3.1 - Pipeline Factory Scaffolding
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from fusion.core.pipeline_factory import PipelineFactory, PipelineSet

if TYPE_CHECKING:
    pass


# =============================================================================
# PipelineSet Tests
# =============================================================================


class TestPipelineSet:
    """Tests for PipelineSet dataclass."""

    def test_create_with_required_only(self) -> None:
        """PipelineSet can be created with only required pipelines."""
        mock_routing = MagicMock()
        mock_spectrum = MagicMock()

        pipeline_set = PipelineSet(
            routing=mock_routing,
            spectrum=mock_spectrum,
        )

        assert pipeline_set.routing is mock_routing
        assert pipeline_set.spectrum is mock_spectrum
        assert pipeline_set.grooming is None
        assert pipeline_set.snr is None
        assert pipeline_set.slicing is None

    def test_create_with_all_pipelines(self) -> None:
        """PipelineSet can be created with all pipelines."""
        mock_routing = MagicMock()
        mock_spectrum = MagicMock()
        mock_grooming = MagicMock()
        mock_snr = MagicMock()
        mock_slicing = MagicMock()

        pipeline_set = PipelineSet(
            routing=mock_routing,
            spectrum=mock_spectrum,
            grooming=mock_grooming,
            snr=mock_snr,
            slicing=mock_slicing,
        )

        assert pipeline_set.routing is mock_routing
        assert pipeline_set.spectrum is mock_spectrum
        assert pipeline_set.grooming is mock_grooming
        assert pipeline_set.snr is mock_snr
        assert pipeline_set.slicing is mock_slicing

    def test_raises_when_routing_none(self) -> None:
        """PipelineSet raises ValueError when routing is None."""
        mock_spectrum = MagicMock()

        with pytest.raises(ValueError, match="requires a routing pipeline"):
            PipelineSet(
                routing=None,  # type: ignore[arg-type]
                spectrum=mock_spectrum,
            )

    def test_raises_when_spectrum_none(self) -> None:
        """PipelineSet raises ValueError when spectrum is None."""
        mock_routing = MagicMock()

        with pytest.raises(ValueError, match="requires a spectrum pipeline"):
            PipelineSet(
                routing=mock_routing,
                spectrum=None,  # type: ignore[arg-type]
            )

    def test_has_grooming_property(self) -> None:
        """has_grooming property returns correct value."""
        mock_routing = MagicMock()
        mock_spectrum = MagicMock()

        # Without grooming
        ps_no_groom = PipelineSet(routing=mock_routing, spectrum=mock_spectrum)
        assert ps_no_groom.has_grooming is False

        # With grooming
        ps_with_groom = PipelineSet(
            routing=mock_routing,
            spectrum=mock_spectrum,
            grooming=MagicMock(),
        )
        assert ps_with_groom.has_grooming is True

    def test_has_snr_property(self) -> None:
        """has_snr property returns correct value."""
        mock_routing = MagicMock()
        mock_spectrum = MagicMock()

        ps_no_snr = PipelineSet(routing=mock_routing, spectrum=mock_spectrum)
        assert ps_no_snr.has_snr is False

        ps_with_snr = PipelineSet(
            routing=mock_routing,
            spectrum=mock_spectrum,
            snr=MagicMock(),
        )
        assert ps_with_snr.has_snr is True

    def test_has_slicing_property(self) -> None:
        """has_slicing property returns correct value."""
        mock_routing = MagicMock()
        mock_spectrum = MagicMock()

        ps_no_slice = PipelineSet(routing=mock_routing, spectrum=mock_spectrum)
        assert ps_no_slice.has_slicing is False

        ps_with_slice = PipelineSet(
            routing=mock_routing,
            spectrum=mock_spectrum,
            slicing=MagicMock(),
        )
        assert ps_with_slice.has_slicing is True

    def test_repr(self) -> None:
        """__repr__ shows pipeline types."""
        mock_routing = MagicMock()
        mock_routing.__class__.__name__ = "RoutingAdapter"
        mock_spectrum = MagicMock()
        mock_spectrum.__class__.__name__ = "SpectrumAdapter"

        ps = PipelineSet(routing=mock_routing, spectrum=mock_spectrum)
        repr_str = repr(ps)

        assert "PipelineSet" in repr_str
        assert "RoutingAdapter" in repr_str
        assert "SpectrumAdapter" in repr_str


# =============================================================================
# PipelineFactory Routing Tests
# =============================================================================


class TestPipelineFactoryRouting:
    """Tests for PipelineFactory.create_routing."""

    def test_creates_routing_adapter_by_default(self) -> None:
        """create_routing returns RoutingAdapter for default config."""
        mock_config = MagicMock()
        mock_config.route_method = "k_shortest_path"

        with patch(
            "fusion.core.adapters.routing_adapter.RoutingAdapter"
        ) as MockAdapter:
            MockAdapter.return_value = MagicMock()

            result = PipelineFactory.create_routing(mock_config)

            MockAdapter.assert_called_once_with(mock_config)
            assert result is MockAdapter.return_value

    def test_creates_routing_adapter_for_congestion_aware(self) -> None:
        """create_routing returns RoutingAdapter for congestion_aware method."""
        mock_config = MagicMock()
        mock_config.route_method = "congestion_aware"

        with patch(
            "fusion.core.adapters.routing_adapter.RoutingAdapter"
        ) as MockAdapter:
            MockAdapter.return_value = MagicMock()

            result = PipelineFactory.create_routing(mock_config)

            MockAdapter.assert_called_once_with(mock_config)
            assert result is MockAdapter.return_value

    def test_creates_protected_routing_for_1plus1(self) -> None:
        """create_routing returns ProtectedRoutingPipeline for 1+1 protection."""
        mock_config = MagicMock()
        mock_config.route_method = "1plus1_protection"

        with patch(
            "fusion.pipelines.routing_pipeline.ProtectedRoutingPipeline"
        ) as MockPipeline:
            MockPipeline.return_value = MagicMock()

            result = PipelineFactory.create_routing(mock_config)

            MockPipeline.assert_called_once_with(mock_config)
            assert result is MockPipeline.return_value


# =============================================================================
# PipelineFactory Spectrum Tests
# =============================================================================


class TestPipelineFactorySpectrum:
    """Tests for PipelineFactory.create_spectrum."""

    def test_creates_spectrum_adapter(self) -> None:
        """create_spectrum returns SpectrumAdapter."""
        mock_config = MagicMock()
        mock_config.allocation_method = "first_fit"

        with patch(
            "fusion.core.adapters.spectrum_adapter.SpectrumAdapter"
        ) as MockAdapter:
            MockAdapter.return_value = MagicMock()

            result = PipelineFactory.create_spectrum(mock_config)

            MockAdapter.assert_called_once_with(mock_config)
            assert result is MockAdapter.return_value

    def test_creates_spectrum_adapter_for_best_fit(self) -> None:
        """create_spectrum returns SpectrumAdapter even for best_fit (for now)."""
        mock_config = MagicMock()
        mock_config.allocation_method = "best_fit"

        with patch(
            "fusion.core.adapters.spectrum_adapter.SpectrumAdapter"
        ) as MockAdapter:
            MockAdapter.return_value = MagicMock()

            result = PipelineFactory.create_spectrum(mock_config)

            MockAdapter.assert_called_once_with(mock_config)
            assert result is MockAdapter.return_value


# =============================================================================
# PipelineFactory Grooming Tests
# =============================================================================


class TestPipelineFactoryGrooming:
    """Tests for PipelineFactory.create_grooming."""

    def test_returns_none_when_disabled(self) -> None:
        """create_grooming returns None when grooming_enabled=False."""
        mock_config = MagicMock()
        mock_config.grooming_enabled = False

        result = PipelineFactory.create_grooming(mock_config)

        assert result is None

    def test_creates_grooming_adapter_when_enabled(self) -> None:
        """create_grooming returns GroomingAdapter when enabled."""
        mock_config = MagicMock()
        mock_config.grooming_enabled = True

        with patch(
            "fusion.core.adapters.grooming_adapter.GroomingAdapter"
        ) as MockAdapter:
            MockAdapter.return_value = MagicMock()

            result = PipelineFactory.create_grooming(mock_config)

            MockAdapter.assert_called_once_with(mock_config)
            assert result is MockAdapter.return_value


# =============================================================================
# PipelineFactory SNR Tests
# =============================================================================


class TestPipelineFactorySNR:
    """Tests for PipelineFactory.create_snr."""

    def test_returns_none_when_disabled(self) -> None:
        """create_snr returns None when snr_enabled=False."""
        mock_config = MagicMock()
        mock_config.snr_enabled = False

        result = PipelineFactory.create_snr(mock_config)

        assert result is None

    def test_creates_snr_adapter_when_enabled(self) -> None:
        """create_snr returns SNRAdapter when enabled."""
        mock_config = MagicMock()
        mock_config.snr_enabled = True

        with patch(
            "fusion.core.adapters.snr_adapter.SNRAdapter"
        ) as MockAdapter:
            MockAdapter.return_value = MagicMock()

            result = PipelineFactory.create_snr(mock_config)

            MockAdapter.assert_called_once_with(mock_config)
            assert result is MockAdapter.return_value


# =============================================================================
# PipelineFactory Slicing Tests
# =============================================================================


class TestPipelineFactorySlicing:
    """Tests for PipelineFactory.create_slicing."""

    def test_returns_none_when_disabled(self) -> None:
        """create_slicing returns None when slicing_enabled=False."""
        mock_config = MagicMock()
        mock_config.slicing_enabled = False

        result = PipelineFactory.create_slicing(mock_config)

        assert result is None

    def test_creates_slicing_pipeline_when_enabled(self) -> None:
        """create_slicing returns StandardSlicingPipeline when enabled."""
        mock_config = MagicMock()
        mock_config.slicing_enabled = True

        with patch(
            "fusion.pipelines.slicing_pipeline.StandardSlicingPipeline"
        ) as MockPipeline:
            MockPipeline.return_value = MagicMock()

            result = PipelineFactory.create_slicing(mock_config)

            MockPipeline.assert_called_once_with(mock_config)
            assert result is MockPipeline.return_value


# =============================================================================
# PipelineFactory PipelineSet Tests
# =============================================================================


class TestPipelineFactoryPipelineSet:
    """Tests for PipelineFactory.create_pipeline_set."""

    def test_creates_complete_pipeline_set(self) -> None:
        """create_pipeline_set creates all pipelines based on config."""
        mock_config = MagicMock()
        mock_config.route_method = "k_shortest_path"
        mock_config.allocation_method = "first_fit"
        mock_config.grooming_enabled = True
        mock_config.snr_enabled = True
        mock_config.slicing_enabled = True

        # Patch all adapter/pipeline imports
        with (
            patch.object(PipelineFactory, "create_routing") as mock_routing,
            patch.object(PipelineFactory, "create_spectrum") as mock_spectrum,
            patch.object(PipelineFactory, "create_grooming") as mock_grooming,
            patch.object(PipelineFactory, "create_snr") as mock_snr,
            patch.object(PipelineFactory, "create_slicing") as mock_slicing,
        ):
            mock_routing.return_value = MagicMock()
            mock_spectrum.return_value = MagicMock()
            mock_grooming.return_value = MagicMock()
            mock_snr.return_value = MagicMock()
            mock_slicing.return_value = MagicMock()

            result = PipelineFactory.create_pipeline_set(mock_config)

            assert isinstance(result, PipelineSet)
            assert result.routing is mock_routing.return_value
            assert result.spectrum is mock_spectrum.return_value
            assert result.grooming is mock_grooming.return_value
            assert result.snr is mock_snr.return_value
            assert result.slicing is mock_slicing.return_value

    def test_creates_minimal_pipeline_set(self) -> None:
        """create_pipeline_set creates only required pipelines when features disabled."""
        mock_config = MagicMock()
        mock_config.route_method = "k_shortest_path"
        mock_config.allocation_method = "first_fit"
        mock_config.grooming_enabled = False
        mock_config.snr_enabled = False
        mock_config.slicing_enabled = False

        with (
            patch.object(PipelineFactory, "create_routing") as mock_routing,
            patch.object(PipelineFactory, "create_spectrum") as mock_spectrum,
            patch.object(PipelineFactory, "create_grooming") as mock_grooming,
            patch.object(PipelineFactory, "create_snr") as mock_snr,
            patch.object(PipelineFactory, "create_slicing") as mock_slicing,
        ):
            mock_routing.return_value = MagicMock()
            mock_spectrum.return_value = MagicMock()
            mock_grooming.return_value = None
            mock_snr.return_value = None
            mock_slicing.return_value = None

            result = PipelineFactory.create_pipeline_set(mock_config)

            assert isinstance(result, PipelineSet)
            assert result.routing is mock_routing.return_value
            assert result.spectrum is mock_spectrum.return_value
            assert result.grooming is None
            assert result.snr is None
            assert result.slicing is None


# =============================================================================
# PipelineFactory Orchestrator Tests
# =============================================================================


class TestPipelineFactoryOrchestrator:
    """Tests for PipelineFactory.create_orchestrator."""

    @pytest.mark.skip(reason="SDNOrchestrator is implemented in P3.2")
    def test_creates_orchestrator_with_pipelines(self) -> None:
        """create_orchestrator creates SDNOrchestrator with pipeline set."""
        mock_config = MagicMock()
        mock_pipeline_set = MagicMock(spec=PipelineSet)

        with (
            patch.object(
                PipelineFactory, "create_pipeline_set"
            ) as mock_create_set,
            patch("fusion.core.orchestrator.SDNOrchestrator") as MockOrch,
        ):
            mock_create_set.return_value = mock_pipeline_set
            MockOrch.return_value = MagicMock()

            result = PipelineFactory.create_orchestrator(mock_config)

            mock_create_set.assert_called_once_with(mock_config)
            MockOrch.assert_called_once_with(mock_config, mock_pipeline_set)
            assert result is MockOrch.return_value
