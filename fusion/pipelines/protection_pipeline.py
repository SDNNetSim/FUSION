"""
Protection pipeline for 1+1 dedicated path protection.

This module provides ProtectionPipeline which handles:
- Finding disjoint path pairs (link or node disjoint)
- Allocating same spectrum on both primary and backup paths
- Creating protected lightpaths via NetworkState

Phase: P5.4 - Protection Pipeline
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from fusion.pipelines.disjoint_path_finder import DisjointPathFinder, DisjointnessType

if TYPE_CHECKING:
    import networkx as nx

    from fusion.domain.lightpath import Lightpath
    from fusion.domain.network_state import NetworkState

logger = logging.getLogger(__name__)


@dataclass
class ProtectedAllocationResult:
    """
    Result of protected spectrum allocation.

    Attributes:
        success: Whether allocation succeeded
        start_slot: Starting spectrum slot (same on both paths)
        end_slot: Ending spectrum slot (exclusive)
        failure_reason: Reason for failure if success=False
    """

    success: bool
    start_slot: int = -1
    end_slot: int = -1
    failure_reason: str | None = None

    @classmethod
    def no_disjoint_paths(cls) -> ProtectedAllocationResult:
        """Create result for no disjoint paths found."""
        return cls(success=False, failure_reason="no_disjoint_paths")

    @classmethod
    def no_common_spectrum(cls) -> ProtectedAllocationResult:
        """Create result for no common spectrum available."""
        return cls(success=False, failure_reason="no_common_spectrum")

    @classmethod
    def allocated(cls, start_slot: int, end_slot: int) -> ProtectedAllocationResult:
        """Create result for successful allocation."""
        return cls(success=True, start_slot=start_slot, end_slot=end_slot)


class ProtectionPipeline:
    """
    Pipeline for 1+1 dedicated path protection.

    Provides methods to:
    - Find disjoint path pairs (link or node disjoint)
    - Allocate the SAME spectrum on both primary and backup paths
    - Integrate with NetworkState for lightpath creation

    1+1 Dedicated Protection:
        - Both primary and backup paths are pre-provisioned
        - SAME spectrum slots are allocated on BOTH paths
        - Traffic is transmitted on both paths simultaneously
        - Receiver monitors primary and switches to backup on failure
        - Fast switchover (typically < 50ms)

    Attributes:
        disjoint_finder: DisjointPathFinder for path computation
        switchover_latency_ms: Protection switchover latency (default 50ms)

    Example:
        >>> pipeline = ProtectionPipeline(DisjointnessType.LINK)
        >>> result = pipeline.allocate_protected(
        ...     primary_path=["A", "B", "C"],
        ...     backup_path=["A", "D", "C"],
        ...     slots_needed=8,
        ...     network_state=state,
        ...     core=0,
        ...     band="c",
        ... )
        >>> if result.success:
        ...     print(f"Allocated slots {result.start_slot}-{result.end_slot}")
    """

    def __init__(
        self,
        disjointness: DisjointnessType = DisjointnessType.LINK,
        switchover_latency_ms: float = 50.0,
    ) -> None:
        """
        Initialize ProtectionPipeline.

        Args:
            disjointness: Type of path disjointness (LINK or NODE)
            switchover_latency_ms: Switchover latency in milliseconds
        """
        self.disjoint_finder = DisjointPathFinder(disjointness)
        self.switchover_latency_ms = switchover_latency_ms

    @property
    def disjointness(self) -> DisjointnessType:
        """Current disjointness mode."""
        return self.disjoint_finder.disjointness

    def find_protected_paths(
        self,
        topology: nx.Graph,
        source: str,
        destination: str,
    ) -> tuple[list[str], list[str]] | None:
        """
        Find a disjoint path pair for protection.

        Args:
            topology: Network topology graph
            source: Source node ID
            destination: Destination node ID

        Returns:
            Tuple of (primary_path, backup_path) or None if not possible
        """
        return self.disjoint_finder.find_disjoint_pair(topology, source, destination)

    def allocate_protected(
        self,
        primary_path: list[str],
        backup_path: list[str],
        slots_needed: int,
        network_state: NetworkState,
        core: int = 0,
        band: str = "c",
    ) -> ProtectedAllocationResult:
        """
        Allocate spectrum on both primary and backup paths.

        For 1+1 dedicated protection, allocates the SAME spectrum slots
        on both paths to enable fast switchover.

        Args:
            primary_path: Primary path node sequence
            backup_path: Backup path node sequence
            slots_needed: Number of spectrum slots needed
            network_state: Current network state
            core: Core index (default 0)
            band: Band identifier (default "c")

        Returns:
            ProtectedAllocationResult with spectrum assignment or failure reason

        Algorithm:
            1. Get spectrum availability on both paths (bitwise arrays)
            2. Compute intersection (common free blocks)
            3. Find first-fit contiguous block satisfying slots_needed
            4. Return allocation details
        """
        # Get spectrum availability on both paths
        primary_available = self._get_path_spectrum_availability(
            primary_path, network_state, core, band
        )
        backup_available = self._get_path_spectrum_availability(
            backup_path, network_state, core, band
        )

        if primary_available is None or backup_available is None:
            return ProtectedAllocationResult.no_common_spectrum()

        # Find common free blocks (bitwise AND)
        common_free = primary_available & backup_available

        # Find contiguous block using first-fit
        start_slot = self._find_first_fit_block(common_free, slots_needed)

        if start_slot < 0:
            logger.debug(
                f"No common spectrum block of {slots_needed} slots "
                f"available on both paths"
            )
            return ProtectedAllocationResult.no_common_spectrum()

        end_slot = start_slot + slots_needed

        logger.debug(
            f"Protected allocation: slots [{start_slot}:{end_slot}] "
            f"on primary {primary_path} and backup {backup_path}"
        )

        return ProtectedAllocationResult.allocated(start_slot, end_slot)

    def create_protected_lightpath(
        self,
        network_state: NetworkState,
        primary_path: list[str],
        backup_path: list[str],
        start_slot: int,
        end_slot: int,
        core: int,
        band: str,
        modulation: str,
        bandwidth_gbps: int,
        path_weight_km: float,
        guard_slots: int = 0,
    ) -> Lightpath:
        """
        Create a protected lightpath with both primary and backup paths.

        Uses NetworkState.create_lightpath() which already supports protection.
        Allocates spectrum on both paths with the same slot range.

        Args:
            network_state: Network state to create lightpath in
            primary_path: Primary path node sequence
            backup_path: Backup path node sequence
            start_slot: Starting spectrum slot
            end_slot: Ending spectrum slot (exclusive)
            core: Core index
            band: Band identifier
            modulation: Modulation format
            bandwidth_gbps: Bandwidth in Gbps
            path_weight_km: Primary path weight in km
            guard_slots: Number of guard slots (default 0)

        Returns:
            Created Lightpath with protection fields set

        Raises:
            ValueError: If spectrum not available on either path
        """
        return network_state.create_lightpath(
            path=primary_path,
            start_slot=start_slot,
            end_slot=end_slot,
            core=core,
            band=band,
            modulation=modulation,
            bandwidth_gbps=bandwidth_gbps,
            path_weight_km=path_weight_km,
            guard_slots=guard_slots,
            # Protection fields
            backup_path=backup_path,
            backup_start_slot=start_slot,  # Same slot for 1+1
            backup_end_slot=end_slot,
            backup_core=core,
            backup_band=band,
        )

    def verify_disjointness(
        self,
        path1: list[str],
        path2: list[str],
    ) -> bool:
        """
        Verify that two paths meet the disjointness requirement.

        Args:
            path1: First path
            path2: Second path

        Returns:
            True if paths are disjoint according to current mode
        """
        if self.disjointness == DisjointnessType.LINK:
            return self.disjoint_finder.are_link_disjoint(path1, path2)
        else:
            return self.disjoint_finder.are_node_disjoint(path1, path2)

    def _get_path_spectrum_availability(
        self,
        path: list[str],
        network_state: NetworkState,
        core: int,
        band: str,
    ) -> np.ndarray | None:
        """
        Get combined spectrum availability for a path (AND of all links).

        Args:
            path: Path as list of node IDs
            network_state: Network state
            core: Core index
            band: Band identifier

        Returns:
            Boolean array where True = free, or None if path invalid
        """
        if len(path) < 2:
            return None

        # Get spectrum size from first link
        try:
            first_link = (path[0], path[1])
            link_spectrum = network_state.get_link_spectrum(first_link)
            num_slots = link_spectrum.get_slot_count(band)
        except KeyError:
            return None

        # Start with all slots available
        combined = np.ones(num_slots, dtype=bool)

        # AND with each link's availability
        for i in range(len(path) - 1):
            link = (path[i], path[i + 1])
            try:
                link_spectrum = network_state.get_link_spectrum(link)
                spectrum_array = link_spectrum.get_spectrum_array(band)
                # 0 = free, nonzero = occupied
                # We want True for free slots
                link_available = spectrum_array[core] == 0
                combined &= link_available
            except (KeyError, IndexError):
                return None

        return combined

    def _find_first_fit_block(
        self,
        available: np.ndarray,
        slots_needed: int,
    ) -> int:
        """
        Find first contiguous block of free slots.

        Args:
            available: Boolean array (True = free)
            slots_needed: Number of contiguous slots needed

        Returns:
            Start slot index, or -1 if no fit found
        """
        consecutive = 0
        start = -1

        for i, free in enumerate(available):
            if free:
                if consecutive == 0:
                    start = i
                consecutive += 1
                if consecutive >= slots_needed:
                    return start
            else:
                consecutive = 0
                start = -1

        return -1

    def get_switchover_latency(self) -> float:
        """Get the protection switchover latency in milliseconds."""
        return self.switchover_latency_ms
