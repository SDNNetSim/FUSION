"""
Unit tests for fusion.interfaces.pipelines module.

Tests the pipeline Protocol classes for FUSION architecture:

- RoutingPipeline
- SpectrumPipeline
- GroomingPipeline
- SNRPipeline
- SlicingPipeline

These tests verify:

1. Protocols are importable and usable
2. Mock implementations satisfy protocols
3. runtime_checkable works correctly
4. Type hints are valid
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fusion.interfaces import (
    GroomingPipeline,
    RoutingPipeline,
    SlicingPipeline,
    SNRPipeline,
    SpectrumPipeline,
)
from fusion.interfaces.pipelines import (
    GroomingPipeline as GroomingPipelineFromModule,
)
from fusion.interfaces.pipelines import (
    RoutingPipeline as RoutingPipelineFromModule,
)
from fusion.interfaces.pipelines import (
    SlicingPipeline as SlicingPipelineFromModule,
)
from fusion.interfaces.pipelines import (
    SNRPipeline as SNRPipelineFromModule,
)
from fusion.interfaces.pipelines import (
    SpectrumPipeline as SpectrumPipelineFromModule,
)

if TYPE_CHECKING:
    from fusion.domain.lightpath import Lightpath
    from fusion.domain.network_state import NetworkState
    from fusion.domain.request import Request
    from fusion.domain.results import (
        GroomingResult,
        RouteResult,
        SlicingResult,
        SNRRecheckResult,
        SNRResult,
        SpectrumResult,
    )


# =============================================================================
# Mock Implementations for Testing
# =============================================================================


class MockRoutingPipeline:
    """Mock implementation of RoutingPipeline for testing."""

    def find_routes(
        self,
        source: str,
        destination: str,
        bandwidth_gbps: int,
        network_state: NetworkState,
        *,
        forced_path: list[str] | None = None,
    ) -> RouteResult:
        """Return empty RouteResult."""
        from fusion.domain.results import RouteResult

        return RouteResult.empty("mock")


class MockSpectrumPipeline:
    """Mock implementation of SpectrumPipeline for testing."""

    def find_spectrum(
        self,
        path: list[str],
        modulation: str | list[str],
        bandwidth_gbps: int,
        network_state: NetworkState,
        *,
        connection_index: int | None = None,
        path_index: int = 0,
        use_dynamic_slicing: bool = False,
        snr_bandwidth: int | None = None,
        request_id: int | None = None,
        slice_bandwidth: int | None = None,
        excluded_modulations: set[str] | None = None,
    ) -> SpectrumResult:
        """Return not found result."""
        from fusion.domain.results import SpectrumResult

        return SpectrumResult.not_found(10)

    def find_protected_spectrum(
        self,
        primary_path: list[str],
        backup_path: list[str],
        modulation: str,
        bandwidth_gbps: int,
        network_state: NetworkState,
    ) -> SpectrumResult:
        """Return not found result."""
        from fusion.domain.results import SpectrumResult

        return SpectrumResult.not_found(10)


class MockGroomingPipeline:
    """Mock implementation of GroomingPipeline for testing."""

    def try_groom(
        self,
        request: Request,
        network_state: NetworkState,
    ) -> GroomingResult:
        """Return no grooming result."""
        from fusion.domain.results import GroomingResult

        return GroomingResult.no_grooming(100)

    def rollback_groom(
        self,
        request: Request,
        lightpath_ids: list[int],
        network_state: NetworkState,
    ) -> None:
        """Do nothing for rollback."""
        pass


class MockSNRPipeline:
    """Mock implementation of SNRPipeline for testing."""

    def validate(
        self,
        lightpath: Lightpath,
        network_state: NetworkState,
    ) -> SNRResult:
        """Return skipped result."""
        from fusion.domain.results import SNRResult

        return SNRResult.skipped()

    def recheck_affected(
        self,
        new_lightpath_id: int,
        network_state: NetworkState,
        *,
        affected_range_slots: int = 5,
        slicing_flag: bool = False,
    ) -> SNRRecheckResult:
        """Return success result."""
        from fusion.domain.results import SNRRecheckResult

        return SNRRecheckResult.success()


class MockSlicingPipeline:
    """Mock implementation of SlicingPipeline for testing."""

    def try_slice(
        self,
        request: Request,
        path: list[str],
        modulation: str,
        bandwidth_gbps: int,
        network_state: NetworkState,
        *,
        max_slices: int | None = None,
        spectrum_pipeline: SpectrumPipeline | None = None,
        snr_pipeline: SNRPipeline | None = None,
        connection_index: int | None = None,
        path_index: int = 0,
        snr_accumulator: list[float] | None = None,
        path_weight: float | None = None,
    ) -> SlicingResult:
        """Return failed result."""
        from fusion.domain.results import SlicingResult

        return SlicingResult.failed()

    def rollback_slices(
        self,
        lightpath_ids: list[int],
        network_state: NetworkState,
    ) -> None:
        """Do nothing for rollback."""
        pass


# =============================================================================
# Incomplete Mock Implementations (Missing Methods)
# =============================================================================


class IncompleteRoutingPipeline:
    """Routing pipeline missing the find_routes method."""

    def wrong_method(self) -> None:
        """Wrong method name."""
        pass


class IncompleteSpectrumPipeline:
    """Spectrum pipeline with only find_spectrum (missing find_protected_spectrum)."""

    def find_spectrum(
        self,
        path: list[str],
        modulation: str,
        bandwidth_gbps: int,
        network_state: NetworkState,
    ) -> SpectrumResult:
        """Return not found result."""
        from fusion.domain.results import SpectrumResult

        return SpectrumResult.not_found(10)


class IncompleteGroomingPipeline:
    """Grooming pipeline with only try_groom (missing rollback_groom)."""

    def try_groom(
        self,
        request: Request,
        network_state: NetworkState,
    ) -> GroomingResult:
        """Return no grooming result."""
        from fusion.domain.results import GroomingResult

        return GroomingResult.no_grooming(100)


class IncompleteSNRPipeline:
    """SNR pipeline with only validate (missing recheck_affected)."""

    def validate(
        self,
        lightpath: Lightpath,
        network_state: NetworkState,
    ) -> SNRResult:
        """Return skipped result."""
        from fusion.domain.results import SNRResult

        return SNRResult.skipped()


class IncompleteSlicingPipeline:
    """Slicing pipeline with only try_slice (missing rollback_slices)."""

    def try_slice(
        self,
        request: Request,
        path: list[str],
        modulation: str,
        bandwidth_gbps: int,
        network_state: NetworkState,
        *,
        max_slices: int | None = None,
        spectrum_pipeline: SpectrumPipeline | None = None,
        snr_pipeline: SNRPipeline | None = None,
    ) -> SlicingResult:
        """Return failed result."""
        from fusion.domain.results import SlicingResult

        return SlicingResult.failed()


# =============================================================================
# Protocol Import Tests
# =============================================================================


class TestProtocolImports:
    """Test that protocols are importable from correct locations."""

    def test_import_routing_pipeline_from_package(self) -> None:
        """RoutingPipeline should be importable from fusion.interfaces."""
        assert RoutingPipeline is not None
        assert RoutingPipeline is RoutingPipelineFromModule

    def test_import_spectrum_pipeline_from_package(self) -> None:
        """SpectrumPipeline should be importable from fusion.interfaces."""
        assert SpectrumPipeline is not None
        assert SpectrumPipeline is SpectrumPipelineFromModule

    def test_import_grooming_pipeline_from_package(self) -> None:
        """GroomingPipeline should be importable from fusion.interfaces."""
        assert GroomingPipeline is not None
        assert GroomingPipeline is GroomingPipelineFromModule

    def test_import_snr_pipeline_from_package(self) -> None:
        """SNRPipeline should be importable from fusion.interfaces."""
        assert SNRPipeline is not None
        assert SNRPipeline is SNRPipelineFromModule

    def test_import_slicing_pipeline_from_package(self) -> None:
        """SlicingPipeline should be importable from fusion.interfaces."""
        assert SlicingPipeline is not None
        assert SlicingPipeline is SlicingPipelineFromModule


# =============================================================================
# runtime_checkable Tests - Complete Implementations
# =============================================================================


class TestRuntimeCheckableComplete:
    """Test runtime_checkable works for complete implementations."""

    def test_mock_routing_is_routing_pipeline(self) -> None:
        """Mock routing with find_routes should pass isinstance check."""
        mock = MockRoutingPipeline()
        assert isinstance(mock, RoutingPipeline)

    def test_mock_spectrum_is_spectrum_pipeline(self) -> None:
        """Mock spectrum with both methods should pass isinstance check."""
        mock = MockSpectrumPipeline()
        assert isinstance(mock, SpectrumPipeline)

    def test_mock_grooming_is_grooming_pipeline(self) -> None:
        """Mock grooming with both methods should pass isinstance check."""
        mock = MockGroomingPipeline()
        assert isinstance(mock, GroomingPipeline)

    def test_mock_snr_is_snr_pipeline(self) -> None:
        """Mock SNR with both methods should pass isinstance check."""
        mock = MockSNRPipeline()
        assert isinstance(mock, SNRPipeline)

    def test_mock_slicing_is_slicing_pipeline(self) -> None:
        """Mock slicing with both methods should pass isinstance check."""
        mock = MockSlicingPipeline()
        assert isinstance(mock, SlicingPipeline)


# =============================================================================
# runtime_checkable Tests - Incomplete Implementations
# =============================================================================


class TestRuntimeCheckableIncomplete:
    """Test runtime_checkable fails for incomplete implementations."""

    def test_incomplete_routing_not_routing_pipeline(self) -> None:
        """Class without find_routes should fail isinstance check."""
        mock = IncompleteRoutingPipeline()
        assert not isinstance(mock, RoutingPipeline)

    def test_incomplete_spectrum_not_spectrum_pipeline(self) -> None:
        """Class missing find_protected_spectrum should fail isinstance check."""
        mock = IncompleteSpectrumPipeline()
        assert not isinstance(mock, SpectrumPipeline)

    def test_incomplete_grooming_not_grooming_pipeline(self) -> None:
        """Class missing rollback_groom should fail isinstance check."""
        mock = IncompleteGroomingPipeline()
        assert not isinstance(mock, GroomingPipeline)

    def test_incomplete_snr_not_snr_pipeline(self) -> None:
        """Class missing recheck_affected should fail isinstance check."""
        mock = IncompleteSNRPipeline()
        assert not isinstance(mock, SNRPipeline)

    def test_incomplete_slicing_not_slicing_pipeline(self) -> None:
        """Class missing rollback_slices should fail isinstance check."""
        mock = IncompleteSlicingPipeline()
        assert not isinstance(mock, SlicingPipeline)


# =============================================================================
# Protocol Method Existence Tests
# =============================================================================


class TestProtocolMethodExistence:
    """Test that protocols have expected methods defined."""

    def test_routing_pipeline_has_find_routes(self) -> None:
        """RoutingPipeline should have find_routes method."""
        assert hasattr(RoutingPipeline, "find_routes")

    def test_spectrum_pipeline_has_find_spectrum(self) -> None:
        """SpectrumPipeline should have find_spectrum method."""
        assert hasattr(SpectrumPipeline, "find_spectrum")

    def test_spectrum_pipeline_has_find_protected_spectrum(self) -> None:
        """SpectrumPipeline should have find_protected_spectrum method."""
        assert hasattr(SpectrumPipeline, "find_protected_spectrum")

    def test_grooming_pipeline_has_try_groom(self) -> None:
        """GroomingPipeline should have try_groom method."""
        assert hasattr(GroomingPipeline, "try_groom")

    def test_grooming_pipeline_has_rollback_groom(self) -> None:
        """GroomingPipeline should have rollback_groom method."""
        assert hasattr(GroomingPipeline, "rollback_groom")

    def test_snr_pipeline_has_validate(self) -> None:
        """SNRPipeline should have validate method."""
        assert hasattr(SNRPipeline, "validate")

    def test_snr_pipeline_has_recheck_affected(self) -> None:
        """SNRPipeline should have recheck_affected method."""
        assert hasattr(SNRPipeline, "recheck_affected")

    def test_slicing_pipeline_has_try_slice(self) -> None:
        """SlicingPipeline should have try_slice method."""
        assert hasattr(SlicingPipeline, "try_slice")

    def test_slicing_pipeline_has_rollback_slices(self) -> None:
        """SlicingPipeline should have rollback_slices method."""
        assert hasattr(SlicingPipeline, "rollback_slices")


# =============================================================================
# Type Annotation Tests
# =============================================================================


class TestTypeAnnotations:
    """Test that protocols can be used as type hints."""

    def test_routing_pipeline_as_type_hint(self) -> None:
        """RoutingPipeline can be used as type hint."""

        def accept_router(router: RoutingPipeline) -> None:
            pass

        mock = MockRoutingPipeline()
        accept_router(mock)  # Should not raise

    def test_spectrum_pipeline_as_type_hint(self) -> None:
        """SpectrumPipeline can be used as type hint."""

        def accept_spectrum(spectrum: SpectrumPipeline) -> None:
            pass

        mock = MockSpectrumPipeline()
        accept_spectrum(mock)

    def test_grooming_pipeline_as_type_hint(self) -> None:
        """GroomingPipeline can be used as type hint."""

        def accept_grooming(grooming: GroomingPipeline) -> None:
            pass

        mock = MockGroomingPipeline()
        accept_grooming(mock)

    def test_snr_pipeline_as_type_hint(self) -> None:
        """SNRPipeline can be used as type hint."""

        def accept_snr(snr: SNRPipeline) -> None:
            pass

        mock = MockSNRPipeline()
        accept_snr(mock)

    def test_slicing_pipeline_as_type_hint(self) -> None:
        """SlicingPipeline can be used as type hint."""

        def accept_slicing(slicing: SlicingPipeline) -> None:
            pass

        mock = MockSlicingPipeline()
        accept_slicing(mock)


# =============================================================================
# Protocol Docstring Tests
# =============================================================================


class TestProtocolDocstrings:
    """Test that protocols have proper documentation."""

    def test_routing_pipeline_has_docstring(self) -> None:
        """RoutingPipeline should have class docstring."""
        assert RoutingPipeline.__doc__ is not None
        assert len(RoutingPipeline.__doc__) > 0

    def test_spectrum_pipeline_has_docstring(self) -> None:
        """SpectrumPipeline should have class docstring."""
        assert SpectrumPipeline.__doc__ is not None
        assert len(SpectrumPipeline.__doc__) > 0

    def test_grooming_pipeline_has_docstring(self) -> None:
        """GroomingPipeline should have class docstring."""
        assert GroomingPipeline.__doc__ is not None
        assert len(GroomingPipeline.__doc__) > 0

    def test_snr_pipeline_has_docstring(self) -> None:
        """SNRPipeline should have class docstring."""
        assert SNRPipeline.__doc__ is not None
        assert len(SNRPipeline.__doc__) > 0

    def test_slicing_pipeline_has_docstring(self) -> None:
        """SlicingPipeline should have class docstring."""
        assert SlicingPipeline.__doc__ is not None
        assert len(SlicingPipeline.__doc__) > 0


# =============================================================================
# Wrong Class Tests
# =============================================================================


class TestWrongClassNotProtocol:
    """Test that unrelated classes don't match protocols."""

    def test_dict_not_any_protocol(self) -> None:
        """Plain dict should not match any protocol."""
        obj: dict[str, int] = {}
        assert not isinstance(obj, RoutingPipeline)
        assert not isinstance(obj, SpectrumPipeline)
        assert not isinstance(obj, GroomingPipeline)
        assert not isinstance(obj, SNRPipeline)
        assert not isinstance(obj, SlicingPipeline)

    def test_random_class_not_any_protocol(self) -> None:
        """Random class should not match any protocol."""

        class RandomClass:
            def random_method(self) -> None:
                pass

        obj = RandomClass()
        assert not isinstance(obj, RoutingPipeline)
        assert not isinstance(obj, SpectrumPipeline)
        assert not isinstance(obj, GroomingPipeline)
        assert not isinstance(obj, SNRPipeline)
        assert not isinstance(obj, SlicingPipeline)
