"""
Protection pipeline for 1+1 dedicated path protection.

This module provides ProtectionPipeline which handles:

- Finding disjoint path pairs (link or node disjoint)
- Allocating same spectrum on both primary and backup paths
- Creating protected lightpaths via NetworkState
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from fusion.pipelines.disjoint_path_finder import DisjointnessType, DisjointPathFinder

if TYPE_CHECKING:
    import networkx as nx

    from fusion.domain.lightpath import Lightpath
    from fusion.domain.network_state import NetworkState


@runtime_checkable
class LinkSpectrumLike(Protocol):
    """Protocol for link spectrum objects used in protection allocation."""

    def get_slot_count(self, band: str) -> int:
        """Get number of slots for a band."""
        ...

    def get_spectrum_array(self, band: str) -> np.ndarray:
        """Get spectrum array for a band."""
        ...


@runtime_checkable
class NetworkStateLike(Protocol):
    """Protocol for network state objects used in protection allocation."""

    def get_link_spectrum(self, link: tuple[str, str]) -> LinkSpectrumLike:
        """Get link spectrum for a link."""
        ...


logger = logging.getLogger(__name__)


@dataclass
class ProtectedAllocationResult:
    """
    Result of protected spectrum allocation.

    :ivar success: Whether allocation succeeded.
    :vartype success: bool
    :ivar start_slot: Starting spectrum slot (same on both paths).
    :vartype start_slot: int
    :ivar end_slot: Ending spectrum slot (exclusive).
    :vartype end_slot: int
    :ivar failure_reason: Reason for failure if success=False.
    :vartype failure_reason: str | None
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

    :ivar disjoint_finder: DisjointPathFinder for path computation.
    :vartype disjoint_finder: DisjointPathFinder
    :ivar switchover_latency_ms: Protection switchover latency (default 50ms).
    :vartype switchover_latency_ms: float

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

        :param disjointness: Type of path disjointness (LINK or NODE).
        :type disjointness: DisjointnessType
        :param switchover_latency_ms: Switchover latency in milliseconds.
        :type switchover_latency_ms: float
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

        :param topology: Network topology graph.
        :type topology: nx.Graph
        :param source: Source node ID.
        :type source: str
        :param destination: Destination node ID.
        :type destination: str
        :return: Tuple of (primary_path, backup_path) or None if not possible.
        :rtype: tuple[list[str], list[str]] | None
        """
        return self.disjoint_finder.find_disjoint_pair(topology, source, destination)

    def allocate_protected(
        self,
        primary_path: list[str],
        backup_path: list[str],
        slots_needed: int,
        network_state: NetworkStateLike,
        core: int = 0,
        band: str = "c",
    ) -> ProtectedAllocationResult:
        """
        Allocate spectrum on both primary and backup paths.

        For 1+1 dedicated protection, allocates the SAME spectrum slots
        on both paths to enable fast switchover.

        :param primary_path: Primary path node sequence.
        :type primary_path: list[str]
        :param backup_path: Backup path node sequence.
        :type backup_path: list[str]
        :param slots_needed: Number of spectrum slots needed.
        :type slots_needed: int
        :param network_state: Current network state.
        :type network_state: NetworkStateLike
        :param core: Core index (default 0).
        :type core: int
        :param band: Band identifier (default "c").
        :type band: str
        :return: ProtectedAllocationResult with spectrum assignment or failure reason.
        :rtype: ProtectedAllocationResult

        Algorithm:

        1. Get spectrum availability on both paths (bitwise arrays)
        2. Compute intersection (common free blocks)
        3. Find first-fit contiguous block satisfying slots_needed
        4. Return allocation details
        """
        # Get spectrum availability on both paths
        primary_available = self._get_path_spectrum_availability(primary_path, network_state, core, band)
        backup_available = self._get_path_spectrum_availability(backup_path, network_state, core, band)

        if primary_available is None or backup_available is None:
            return ProtectedAllocationResult.no_common_spectrum()

        # Find common free blocks (bitwise AND)
        common_free = primary_available & backup_available

        # Find contiguous block using first-fit
        start_slot = self._find_first_fit_block(common_free, slots_needed)

        if start_slot < 0:
            logger.debug(f"No common spectrum block of {slots_needed} slots available on both paths")
            return ProtectedAllocationResult.no_common_spectrum()

        end_slot = start_slot + slots_needed

        logger.debug(f"Protected allocation: slots [{start_slot}:{end_slot}] on primary {primary_path} and backup {backup_path}")

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

        :param network_state: Network state to create lightpath in.
        :type network_state: NetworkState
        :param primary_path: Primary path node sequence.
        :type primary_path: list[str]
        :param backup_path: Backup path node sequence.
        :type backup_path: list[str]
        :param start_slot: Starting spectrum slot.
        :type start_slot: int
        :param end_slot: Ending spectrum slot (exclusive).
        :type end_slot: int
        :param core: Core index.
        :type core: int
        :param band: Band identifier.
        :type band: str
        :param modulation: Modulation format.
        :type modulation: str
        :param bandwidth_gbps: Bandwidth in Gbps.
        :type bandwidth_gbps: int
        :param path_weight_km: Primary path weight in km.
        :type path_weight_km: float
        :param guard_slots: Number of guard slots (default 0).
        :type guard_slots: int
        :return: Created Lightpath with protection fields set.
        :rtype: Lightpath
        :raises ValueError: If spectrum not available on either path.
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

        :param path1: First path.
        :type path1: list[str]
        :param path2: Second path.
        :type path2: list[str]
        :return: True if paths are disjoint according to current mode.
        :rtype: bool
        """
        if self.disjointness == DisjointnessType.LINK:
            return self.disjoint_finder.are_link_disjoint(path1, path2)
        else:
            return self.disjoint_finder.are_node_disjoint(path1, path2)

    def _get_path_spectrum_availability(
        self,
        path: list[str],
        network_state: NetworkStateLike,
        core: int,
        band: str,
    ) -> np.ndarray | None:
        """
        Get combined spectrum availability for a path (AND of all links).

        :param path: Path as list of node IDs.
        :type path: list[str]
        :param network_state: Network state.
        :type network_state: NetworkStateLike
        :param core: Core index.
        :type core: int
        :param band: Band identifier.
        :type band: str
        :return: Boolean array where True = free, or None if path invalid.
        :rtype: np.ndarray | None
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

        :param available: Boolean array (True = free).
        :type available: np.ndarray
        :param slots_needed: Number of contiguous slots needed.
        :type slots_needed: int
        :return: Start slot index, or -1 if no fit found.
        :rtype: int
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
