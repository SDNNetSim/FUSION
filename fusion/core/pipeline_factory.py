"""
Pipeline factory for FUSION simulation.

This module provides PipelineFactory for creating pipelines based on
SimulationConfig, and PipelineSet for holding pipeline references.

Design Principles:
    - Factory is stateless (static/class methods only)
    - Lazy imports to avoid circular dependencies
    - Config-driven pipeline selection
    - Optional pipelines return None when disabled

Usage:
    >>> config = SimulationConfig.from_engine_props(engine_props)
    >>> pipelines = PipelineFactory.create_pipeline_set(config)
    >>> orchestrator = PipelineFactory.create_orchestrator(config)

Phase: P3.1 - Pipeline Factory Scaffolding
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fusion.domain.config import SimulationConfig
    from fusion.interfaces.pipelines import (
        GroomingPipeline,
        RoutingPipeline,
        SlicingPipeline,
        SNRPipeline,
        SpectrumPipeline,
    )

logger = logging.getLogger(__name__)


# =============================================================================
# PipelineSet
# =============================================================================


@dataclass(frozen=False, slots=True)
class PipelineSet:
    """
    Container for all pipelines used by SDNOrchestrator.

    This dataclass holds references to pipeline implementations that
    will be called by the orchestrator during request handling.

    Required pipelines (always present):
        routing: Finds candidate paths between nodes
        spectrum: Finds available spectrum slots along paths

    Optional pipelines (may be None if feature disabled):
        grooming: Grooms requests onto existing lightpaths
        snr: Validates signal quality
        slicing: Slices large requests across multiple lightpaths

    Attributes:
        routing: RoutingPipeline implementation (required)
        spectrum: SpectrumPipeline implementation (required)
        grooming: GroomingPipeline or None if disabled
        snr: SNRPipeline or None if disabled
        slicing: SlicingPipeline or None if disabled

    Example:
        >>> config = SimulationConfig.from_engine_props(engine_props)
        >>> pipelines = PipelineFactory.create_pipeline_set(config)
        >>> print(pipelines.routing)
        <RoutingAdapter>
        >>> print(pipelines.grooming)
        None  # if grooming_enabled=False

    Phase: P3.1 - Pipeline Factory Scaffolding
    """

    # Required pipelines
    routing: RoutingPipeline
    spectrum: SpectrumPipeline

    # Optional pipelines (None if feature disabled)
    grooming: GroomingPipeline | None = None
    snr: SNRPipeline | None = None
    slicing: SlicingPipeline | None = None

    def __post_init__(self) -> None:
        """Validate required pipelines are present."""
        if self.routing is None:
            raise ValueError("PipelineSet requires a routing pipeline")
        if self.spectrum is None:
            raise ValueError("PipelineSet requires a spectrum pipeline")

    @property
    def has_grooming(self) -> bool:
        """Check if grooming is available."""
        return self.grooming is not None

    @property
    def has_snr(self) -> bool:
        """Check if SNR validation is available."""
        return self.snr is not None

    @property
    def has_slicing(self) -> bool:
        """Check if slicing is available."""
        return self.slicing is not None

    def __repr__(self) -> str:
        """Readable representation showing pipeline types."""
        parts = [
            f"routing={type(self.routing).__name__}",
            f"spectrum={type(self.spectrum).__name__}",
        ]
        if self.grooming:
            parts.append(f"grooming={type(self.grooming).__name__}")
        if self.snr:
            parts.append(f"snr={type(self.snr).__name__}")
        if self.slicing:
            parts.append(f"slicing={type(self.slicing).__name__}")
        return f"PipelineSet({', '.join(parts)})"


# =============================================================================
# PipelineFactory
# =============================================================================


class PipelineFactory:
    """
    Factory for creating pipelines based on SimulationConfig.

    This factory uses lazy imports to avoid circular dependencies and
    selects between legacy adapters and new pipeline implementations
    based on configuration values.

    Selection Rules:
        - Routing: ProtectedRoutingPipeline for 1+1, else RoutingAdapter
        - Spectrum: SpectrumAdapter (default)
        - Grooming: GroomingAdapter if enabled, else None
        - SNR: SNRAdapter if enabled, else None
        - Slicing: StandardSlicingPipeline if enabled, else None

    Design Notes:
        - All methods are static or class methods (stateless)
        - Imports are done inside methods (lazy) to avoid circular deps
        - Factory does not store any state

    Example:
        >>> config = SimulationConfig.from_engine_props(engine_props)
        >>> pipelines = PipelineFactory.create_pipeline_set(config)
        >>> orchestrator = PipelineFactory.create_orchestrator(config)

    Phase: P3.1 - Pipeline Factory Scaffolding
    """

    @staticmethod
    def create_routing(config: SimulationConfig) -> RoutingPipeline:
        """
        Create routing pipeline based on config.

        Selection Logic:
            - route_method == "1plus1_protection" -> ProtectedRoutingPipeline
            - Any other route_method -> RoutingAdapter (wraps legacy)

        Args:
            config: Simulation configuration

        Returns:
            RoutingPipeline implementation
        """
        route_method = getattr(config, "route_method", "k_shortest_path")

        if route_method == "1plus1_protection":
            logger.debug("Creating ProtectedRoutingPipeline for 1+1 protection")
            from fusion.pipelines.routing_pipeline import ProtectedRoutingPipeline

            return ProtectedRoutingPipeline(config)
        else:
            logger.debug(f"Creating RoutingAdapter for method: {route_method}")
            from fusion.core.adapters.routing_adapter import RoutingAdapter

            return RoutingAdapter(config)

    @staticmethod
    def create_spectrum(config: SimulationConfig) -> SpectrumPipeline:
        """
        Create spectrum pipeline based on config.

        Currently always returns SpectrumAdapter (wraps legacy).
        Future: May add BestFitSpectrumPipeline for allocation_method=="best_fit".

        Args:
            config: Simulation configuration

        Returns:
            SpectrumPipeline implementation
        """
        allocation_method = getattr(config, "allocation_method", "first_fit")

        # For now, always use adapter
        # Future: Add new implementations for specific allocation methods
        logger.debug(f"Creating SpectrumAdapter for method: {allocation_method}")
        from fusion.core.adapters.spectrum_adapter import SpectrumAdapter

        return SpectrumAdapter(config)

    @staticmethod
    def create_grooming(config: SimulationConfig) -> GroomingPipeline | None:
        """
        Create grooming pipeline if enabled.

        Selection Logic:
            - grooming_enabled == False -> None
            - grooming_enabled == True -> GroomingAdapter (wraps legacy)

        Args:
            config: Simulation configuration

        Returns:
            GroomingPipeline if enabled, None otherwise
        """
        grooming_enabled = getattr(config, "grooming_enabled", False)

        if not grooming_enabled:
            logger.debug("Grooming disabled, returning None")
            return None

        logger.debug("Creating GroomingAdapter")
        from fusion.core.adapters.grooming_adapter import GroomingAdapter

        return GroomingAdapter(config)

    @staticmethod
    def create_snr(config: SimulationConfig) -> SNRPipeline | None:
        """
        Create SNR pipeline if enabled.

        Selection Logic:
            - snr_enabled == False -> None
            - snr_enabled == True -> SNRAdapter (wraps legacy)

        Args:
            config: Simulation configuration

        Returns:
            SNRPipeline if enabled, None otherwise
        """
        snr_enabled = getattr(config, "snr_enabled", False)

        if not snr_enabled:
            logger.debug("SNR validation disabled, returning None")
            return None

        logger.debug("Creating SNRAdapter")
        from fusion.core.adapters.snr_adapter import SNRAdapter

        return SNRAdapter(config)

    @staticmethod
    def create_slicing(config: SimulationConfig) -> SlicingPipeline | None:
        """
        Create slicing pipeline if enabled.

        Selection Logic:
            - slicing_enabled == False -> None
            - slicing_enabled == True -> StandardSlicingPipeline

        Args:
            config: Simulation configuration

        Returns:
            SlicingPipeline if enabled, None otherwise
        """
        slicing_enabled = getattr(config, "slicing_enabled", False)

        if not slicing_enabled:
            logger.debug("Slicing disabled, returning None")
            return None

        logger.debug("Creating StandardSlicingPipeline")
        from fusion.pipelines.slicing_pipeline import StandardSlicingPipeline

        return StandardSlicingPipeline(config)

    @classmethod
    def create_pipeline_set(cls, config: SimulationConfig) -> PipelineSet:
        """
        Create complete pipeline set from configuration.

        Creates all pipelines based on configuration settings.
        Required pipelines (routing, spectrum) are always created.
        Optional pipelines may be None based on feature flags.

        Args:
            config: Simulation configuration

        Returns:
            PipelineSet with all pipelines configured

        Example:
            >>> config = SimulationConfig.from_engine_props(engine_props)
            >>> pipelines = PipelineFactory.create_pipeline_set(config)
            >>> print(pipelines)
            PipelineSet(routing=RoutingAdapter, spectrum=SpectrumAdapter)
        """
        logger.info("Creating pipeline set from configuration")

        return PipelineSet(
            routing=cls.create_routing(config),
            spectrum=cls.create_spectrum(config),
            grooming=cls.create_grooming(config),
            snr=cls.create_snr(config),
            slicing=cls.create_slicing(config),
        )

    @classmethod
    def create_orchestrator(cls, config: SimulationConfig) -> "SDNOrchestrator":
        """
        Create orchestrator with configured pipelines.

        Convenience method that creates both pipeline set and orchestrator.
        The orchestrator is the main entry point for request handling
        in the new v5 architecture.

        Args:
            config: Simulation configuration

        Returns:
            SDNOrchestrator instance with configured pipelines

        Example:
            >>> config = SimulationConfig.from_engine_props(engine_props)
            >>> orchestrator = PipelineFactory.create_orchestrator(config)
            >>> result = orchestrator.handle_request(request, network_state)

        Note:
            SDNOrchestrator is implemented in Phase 3.2. This method
            will raise ImportError until P3.2 is complete.
        """
        from fusion.core.orchestrator import SDNOrchestrator

        pipelines = cls.create_pipeline_set(config)
        return SDNOrchestrator(config, pipelines)


# =============================================================================
# Type alias for forward reference
# =============================================================================

if TYPE_CHECKING:
    from fusion.core.orchestrator import SDNOrchestrator
