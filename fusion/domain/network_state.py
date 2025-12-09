"""
Network state management for FUSION simulation.

This module provides:
- LinkSpectrum: Per-link spectrum state management
- NetworkState: Single source of truth for network state during simulation

Phase: P2.1 - NetworkState Core

Design Principles:
    - NetworkState is a state container, NOT a routing algorithm
    - One NetworkState instance per simulation
    - Passed by reference to pipelines (not stored)
    - All state mutations through explicit methods
    - Read methods are query-only (no side effects)

Spectrum Array Values:
    - 0: Slot is free
    - Positive int: Slot occupied by lightpath with that ID
    - Negative int: Guard band slot for lightpath with abs(value) ID
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Iterator

import networkx as nx
import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from fusion.domain.config import SimulationConfig
    from fusion.domain.lightpath import Lightpath


# =============================================================================
# LinkSpectrum Dataclass
# =============================================================================


@dataclass
class LinkSpectrum:
    """
    Per-link spectrum management.

    Encapsulates the spectrum allocation state for a single network link,
    supporting multi-band and multi-core configurations.

    Attributes:
        link: The link identifier as (source, destination) tuple
        cores_matrix: Spectrum arrays per band, shape (cores, slots) each
        usage_count: Number of active spectrum allocations on this link
        throughput: Cumulative bandwidth served through this link (Gbps)
        link_num: Link index for reference
        length_km: Physical link length in kilometers

    Spectrum Array Values:
        - 0: Slot is free
        - Positive int: Slot occupied by lightpath with that ID
        - Negative int: Guard band slot for lightpath with abs(value) ID

    Example:
        >>> spectrum = LinkSpectrum.from_config(("A", "B"), config, link_num=0)
        >>> spectrum.is_range_free(0, 10, core=0, band="c")
        True
        >>> spectrum.allocate_range(0, 10, core=0, band="c", lightpath_id=1)
        >>> spectrum.is_range_free(0, 10, core=0, band="c")
        False
    """

    link: tuple[str, str]
    cores_matrix: dict[str, npt.NDArray[np.int64]] = field(default_factory=dict)
    usage_count: int = 0
    throughput: float = 0.0
    link_num: int = 0
    length_km: float = 0.0

    # Class-level constants
    FREE_SLOT: ClassVar[int] = 0

    def is_range_free(
        self,
        start_slot: int,
        end_slot: int,
        core: int,
        band: str,
    ) -> bool:
        """
        Check if a spectrum range is completely free.

        Args:
            start_slot: First slot index (inclusive)
            end_slot: Last slot index (exclusive)
            core: Core index (0 to cores_per_link - 1)
            band: Band identifier ("c", "l", "s")

        Returns:
            True if all slots in range are 0 (free), False otherwise

        Raises:
            KeyError: If band is not in cores_matrix
            IndexError: If core or slot indices are out of bounds
        """
        spectrum = self.cores_matrix[band]
        return bool(np.all(spectrum[core, start_slot:end_slot] == self.FREE_SLOT))

    def allocate_range(
        self,
        start_slot: int,
        end_slot: int,
        core: int,
        band: str,
        lightpath_id: int,
        guard_slots: int = 0,
    ) -> None:
        """
        Allocate a spectrum range to a lightpath.

        Marks data slots with positive lightpath_id and guard slots with negative.
        This follows the legacy convention where guard bands use negative IDs.

        Args:
            start_slot: First slot index (inclusive)
            end_slot: Last slot index (exclusive), includes guard slots
            core: Core index
            band: Band identifier
            lightpath_id: Positive integer ID for the lightpath
            guard_slots: Number of guard band slots at end of allocation

        Raises:
            ValueError: If lightpath_id <= 0
            ValueError: If range is not free (would overwrite existing allocation)
            KeyError: If band is not in cores_matrix
            IndexError: If indices are out of bounds

        Side Effects:
            - Sets spectrum[core][start:end-guard_slots] = lightpath_id
            - Sets spectrum[core][end-guard_slots:end] = -lightpath_id (guard)
            - Increments usage_count
        """
        if lightpath_id <= 0:
            msg = f"lightpath_id must be positive, got {lightpath_id}"
            raise ValueError(msg)

        if not self.is_range_free(start_slot, end_slot, core, band):
            msg = (
                f"Spectrum range [{start_slot}:{end_slot}] on "
                f"core {core} band {band} is not free"
            )
            raise ValueError(msg)

        spectrum = self.cores_matrix[band]

        # Data slots (excluding guard band)
        data_end = end_slot - guard_slots
        spectrum[core, start_slot:data_end] = lightpath_id

        # Guard band slots (if any) - use negative ID per legacy convention
        if guard_slots > 0:
            spectrum[core, data_end:end_slot] = -lightpath_id

        self.usage_count += 1

    def release_range(
        self,
        start_slot: int,
        end_slot: int,
        core: int,
        band: str,
    ) -> None:
        """
        Release a spectrum range (set to free).

        Args:
            start_slot: First slot index (inclusive)
            end_slot: Last slot index (exclusive)
            core: Core index
            band: Band identifier

        Raises:
            KeyError: If band is not in cores_matrix
            IndexError: If indices are out of bounds

        Side Effects:
            - Sets spectrum[core][start:end] = 0
            - Decrements usage_count (clamped to >= 0)
        """
        spectrum = self.cores_matrix[band]
        spectrum[core, start_slot:end_slot] = self.FREE_SLOT
        self.usage_count = max(0, self.usage_count - 1)

    def release_by_lightpath_id(
        self,
        lightpath_id: int,
        band: str,
        core: int,
    ) -> tuple[int, int] | None:
        """
        Release all slots allocated to a specific lightpath.

        Finds and clears both data slots (positive ID) and guard slots (negative ID).

        Args:
            lightpath_id: The lightpath ID to release
            band: Band identifier
            core: Core index

        Returns:
            Tuple of (start_slot, end_slot) if found, None if not found

        Side Effects:
            - Clears all slots with +/- lightpath_id to 0
            - Decrements usage_count
        """
        spectrum = self.cores_matrix[band]

        # Find slots with this lightpath ID (data) or negative ID (guard)
        data_mask = spectrum[core] == lightpath_id
        guard_mask = spectrum[core] == -lightpath_id
        combined_mask = data_mask | guard_mask

        if not np.any(combined_mask):
            return None

        # Find the range
        indices = np.where(combined_mask)[0]
        start_slot = int(indices[0])
        end_slot = int(indices[-1]) + 1

        # Clear the slots
        spectrum[core, start_slot:end_slot] = self.FREE_SLOT
        self.usage_count = max(0, self.usage_count - 1)

        return (start_slot, end_slot)

    def get_spectrum_array(self, band: str) -> npt.NDArray[np.int64]:
        """
        Get spectrum array for a band.

        Args:
            band: Band identifier

        Returns:
            numpy array of shape (cores, slots), values are lightpath IDs or 0

        Warning:
            Returns actual array reference. Callers should NOT modify directly.
            Use allocate_range() and release_range() instead.
        """
        return self.cores_matrix[band]

    def get_slot_count(self, band: str) -> int:
        """Get number of slots in a band."""
        return int(self.cores_matrix[band].shape[1])

    def get_core_count(self) -> int:
        """Get number of cores (same across all bands)."""
        if not self.cores_matrix:
            return 0
        first_band = next(iter(self.cores_matrix))
        return int(self.cores_matrix[first_band].shape[0])

    def get_bands(self) -> list[str]:
        """Get list of available bands."""
        return list(self.cores_matrix.keys())

    def get_free_slot_count(self, band: str, core: int) -> int:
        """
        Count free slots on a specific core and band.

        Args:
            band: Band identifier
            core: Core index

        Returns:
            Number of free (value == 0) slots
        """
        spectrum = self.cores_matrix[band]
        return int(np.sum(spectrum[core] == self.FREE_SLOT))

    def get_fragmentation_ratio(self, band: str, core: int) -> float:
        """
        Calculate fragmentation ratio for a core/band.

        Fragmentation = 1 - (largest_contiguous_free / total_free)
        Value of 0 means no fragmentation, 1 means maximum fragmentation.

        Args:
            band: Band identifier
            core: Core index

        Returns:
            Fragmentation ratio between 0.0 and 1.0
        """
        spectrum = self.cores_matrix[band]
        free_mask = spectrum[core] == self.FREE_SLOT
        total_free = int(np.sum(free_mask))

        if total_free == 0:
            return 0.0  # No free slots means no fragmentation (just full)

        # Find largest contiguous free block
        # Use run-length encoding approach
        padded = np.concatenate([[False], free_mask, [False]])
        changes = np.diff(padded.astype(int))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]

        if len(starts) == 0:
            return 0.0

        largest_block = int(np.max(ends - starts))

        return 1.0 - (largest_block / total_free)

    @classmethod
    def from_config(
        cls,
        link: tuple[str, str],
        config: SimulationConfig,
        link_num: int = 0,
        length_km: float = 0.0,
    ) -> LinkSpectrum:
        """
        Create LinkSpectrum initialized from SimulationConfig.

        Args:
            link: Link identifier tuple
            config: Simulation configuration with band/core specs
            link_num: Link index
            length_km: Physical link length

        Returns:
            New LinkSpectrum with zero-initialized spectrum arrays
            matching config.band_list and config.cores_per_link
        """
        cores_matrix: dict[str, npt.NDArray[np.int64]] = {}

        for band in config.band_list:
            num_slots = config.band_slots.get(band, 320)  # Default 320 if not specified
            cores_matrix[band] = np.zeros(
                (config.cores_per_link, num_slots),
                dtype=np.int64,
            )

        return cls(
            link=link,
            cores_matrix=cores_matrix,
            link_num=link_num,
            length_km=length_km,
        )

    def to_legacy_dict(self) -> dict[str, Any]:
        """
        Convert to legacy network_spectrum_dict entry format.

        Returns:
            Dictionary compatible with legacy sdn_props.network_spectrum_dict[link]
        """
        return {
            "cores_matrix": self.cores_matrix,
            "link_num": self.link_num,
            "usage_count": self.usage_count,
            "throughput": self.throughput,
            "length": self.length_km,
        }


# =============================================================================
# NetworkState Class
# =============================================================================


class NetworkState:
    """
    Single source of truth for network state during simulation.

    Owns:
        - Network topology (read-only after init)
        - Per-link spectrum state (LinkSpectrum objects)
        - Active lightpaths (by ID)
        - Lightpath ID counter

    Design Principles:
        - One instance per simulation
        - Passed by reference to pipelines (not stored)
        - All state mutations through explicit methods
        - Read methods are query-only (no side effects)
        - NetworkState is a state container, NOT a routing algorithm

    State Authority (from Gap Analysis P2.0):
        - Spectrum arrays: NetworkState._spectrum[link].cores_matrix
        - Active lightpaths: NetworkState._lightpaths[id]
        - Lightpath ID counter: NetworkState._next_lightpath_id
        - Topology: NetworkState._topology (read-only reference)

    NOT in NetworkState (by design):
        - Request queue (SimulationEngine)
        - Request status tracking (SimulationEngine)
        - Statistics (SimStats)
        - Per-request temporary lists (SDNController)

    Example:
        >>> state = NetworkState(topology, config)
        >>> if state.is_spectrum_available(path, 0, 10, core=0, band="c"):
        ...     # Spectrum is free, can allocate (using P2.2 write methods)
        ...     pass
    """

    __slots__ = (
        "_topology",
        "_config",
        "_spectrum",
        "_lightpaths",
        "_next_lightpath_id",
    )

    def __init__(
        self,
        topology: nx.Graph,
        config: SimulationConfig,
    ) -> None:
        """
        Initialize NetworkState from topology and configuration.

        Args:
            topology: NetworkX graph representing network structure.
                      Edge attributes are preserved (length, etc.)
            config: Immutable simulation configuration

        Raises:
            ValueError: If topology is empty (no edges)
            ValueError: If config band_list is empty

        Note:
            Topology is stored as reference (not copied). Callers must
            not modify topology after passing to NetworkState.
        """
        if topology.number_of_edges() == 0:
            msg = "Topology must have at least one edge"
            raise ValueError(msg)

        if not config.band_list:
            msg = "Config must have at least one band in band_list"
            raise ValueError(msg)

        self._topology: nx.Graph = topology
        self._config: SimulationConfig = config
        self._spectrum: dict[tuple[str, str], LinkSpectrum] = {}
        self._lightpaths: dict[int, Lightpath] = {}
        self._next_lightpath_id: int = 1

        self._init_spectrum()

    def _init_spectrum(self) -> None:
        """Initialize LinkSpectrum for all edges in topology."""
        link_num = 0
        for u, v, edge_data in self._topology.edges(data=True):
            # Get link length from edge data (try multiple common attribute names)
            length_km = float(
                edge_data.get("length", edge_data.get("weight", edge_data.get("distance", 0.0)))
            )

            # Create single LinkSpectrum for both directions
            link_spectrum = LinkSpectrum.from_config(
                link=(str(u), str(v)),
                config=self._config,
                link_num=link_num,
                length_km=length_km,
            )

            # Both directions point to same object (bidirectional)
            self._spectrum[(str(u), str(v))] = link_spectrum
            self._spectrum[(str(v), str(u))] = link_spectrum

            link_num += 1

    # =========================================================================
    # Read-Only Properties
    # =========================================================================

    @property
    def topology(self) -> nx.Graph:
        """Network topology graph (read-only reference)."""
        return self._topology

    @property
    def config(self) -> SimulationConfig:
        """Simulation configuration (immutable)."""
        return self._config

    @property
    def lightpath_count(self) -> int:
        """Number of active lightpaths."""
        return len(self._lightpaths)

    @property
    def link_count(self) -> int:
        """Number of links in topology (edges, not bidirectional pairs)."""
        return self._topology.number_of_edges()

    @property
    def next_lightpath_id(self) -> int:
        """Next lightpath ID that will be assigned (read-only)."""
        return self._next_lightpath_id

    @property
    def node_count(self) -> int:
        """Number of nodes in topology."""
        return self._topology.number_of_nodes()

    # =========================================================================
    # Lightpath Read Methods
    # =========================================================================

    def get_lightpath(self, lightpath_id: int) -> Lightpath | None:
        """
        Retrieve a lightpath by ID.

        Args:
            lightpath_id: Unique lightpath identifier

        Returns:
            Lightpath object if found, None otherwise

        Note:
            Returns direct reference. Callers may read Lightpath state
            but should use NetworkState methods to modify it.
        """
        return self._lightpaths.get(lightpath_id)

    def get_lightpaths_between(
        self,
        source: str,
        destination: str,
    ) -> list[Lightpath]:
        """
        Get all lightpaths between two endpoints.

        Args:
            source: Source node ID
            destination: Destination node ID

        Returns:
            List of lightpaths connecting these endpoints (either direction).
            Empty list if none found.

        Note:
            Handles both (src, dst) and (dst, src) automatically.
        """
        result: list[Lightpath] = []
        for lp in self._lightpaths.values():
            # Check both directions
            if (lp.source == source and lp.destination == destination) or (
                lp.source == destination and lp.destination == source
            ):
                result.append(lp)
        return result

    def get_lightpaths_with_capacity(
        self,
        source: str,
        destination: str,
        min_bandwidth_gbps: int = 1,
    ) -> list[Lightpath]:
        """
        Get lightpaths between endpoints with available capacity.

        Useful for grooming: find lightpaths that can accept more traffic.

        Args:
            source: Source node ID
            destination: Destination node ID
            min_bandwidth_gbps: Minimum remaining bandwidth required

        Returns:
            List of lightpaths with at least min_bandwidth_gbps available.
            Sorted by remaining bandwidth descending (best candidates first).
        """
        candidates = self.get_lightpaths_between(source, destination)
        with_capacity = [
            lp for lp in candidates if lp.remaining_bandwidth_gbps >= min_bandwidth_gbps
        ]
        # Sort by remaining bandwidth, highest first
        return sorted(
            with_capacity,
            key=lambda lp: lp.remaining_bandwidth_gbps,
            reverse=True,
        )

    def get_lightpaths_on_link(self, link: tuple[str, str]) -> list[Lightpath]:
        """
        Get all lightpaths that traverse a specific link.

        Useful for failure impact analysis.

        Args:
            link: (source, destination) tuple

        Returns:
            List of lightpaths that include this link in their path.
        """
        result: list[Lightpath] = []
        u, v = link
        for lp in self._lightpaths.values():
            path = lp.path
            for i in range(len(path) - 1):
                if (path[i] == u and path[i + 1] == v) or (
                    path[i] == v and path[i + 1] == u
                ):
                    result.append(lp)
                    break
        return result

    def iter_lightpaths(self) -> Iterator[Lightpath]:
        """
        Iterate over all active lightpaths.

        Yields:
            Each active Lightpath in arbitrary order

        Note:
            Safe to iterate during simulation. Do not add/remove
            lightpaths during iteration.
        """
        yield from self._lightpaths.values()

    def has_lightpath(self, lightpath_id: int) -> bool:
        """Check if a lightpath exists."""
        return lightpath_id in self._lightpaths

    # =========================================================================
    # Spectrum Read Methods
    # =========================================================================

    def get_link_spectrum(self, link: tuple[str, str]) -> LinkSpectrum:
        """
        Get LinkSpectrum for a specific link.

        Args:
            link: (source, destination) tuple

        Returns:
            LinkSpectrum object for that link

        Raises:
            KeyError: If link does not exist in topology

        Note:
            Both (u, v) and (v, u) return the same LinkSpectrum object.
        """
        str_link = (str(link[0]), str(link[1]))
        if str_link not in self._spectrum:
            msg = f"Link {link} does not exist in topology"
            raise KeyError(msg)
        return self._spectrum[str_link]

    def is_spectrum_available(
        self,
        path: list[str],
        start_slot: int,
        end_slot: int,
        core: int,
        band: str,
    ) -> bool:
        """
        Check if spectrum range is free along entire path.

        Args:
            path: Ordered list of node IDs forming the path
            start_slot: First slot index (inclusive)
            end_slot: Last slot index (exclusive)
            core: Core index
            band: Band identifier

        Returns:
            True if all slots in range are free on all links in path

        Raises:
            ValueError: If path has fewer than 2 nodes
            KeyError: If any link in path doesn't exist
        """
        if len(path) < 2:
            msg = "Path must have at least 2 nodes"
            raise ValueError(msg)

        for link in self._get_path_links(path):
            link_spectrum = self.get_link_spectrum(link)
            if not link_spectrum.is_range_free(start_slot, end_slot, core, band):
                return False
        return True

    def find_first_fit(
        self,
        path: list[str],
        slots_needed: int,
        core: int,
        band: str,
    ) -> int | None:
        """
        Find first available slot range along path (first-fit algorithm).

        Args:
            path: Ordered list of node IDs
            slots_needed: Number of contiguous slots required
            core: Core index to search
            band: Band identifier

        Returns:
            Start slot index if found, None if no fit available

        Note:
            This is a read-only query method. It does NOT allocate.
            This helper exists because first-fit is commonly needed.
            More sophisticated algorithms should be in pipelines.
        """
        if len(path) < 2:
            return None

        # Get max slots from first link's spectrum
        first_link = (path[0], path[1])
        link_spectrum = self.get_link_spectrum(first_link)
        max_slots = link_spectrum.get_slot_count(band)

        # Search for contiguous free range
        for start in range(max_slots - slots_needed + 1):
            end = start + slots_needed
            if self.is_spectrum_available(path, start, end, core, band):
                return start

        return None

    def find_last_fit(
        self,
        path: list[str],
        slots_needed: int,
        core: int,
        band: str,
    ) -> int | None:
        """
        Find last available slot range along path (last-fit algorithm).

        Args:
            path: Ordered list of node IDs
            slots_needed: Number of contiguous slots required
            core: Core index to search
            band: Band identifier

        Returns:
            Start slot index if found, None if no fit available

        Note:
            This is a read-only query method. It does NOT allocate.
        """
        if len(path) < 2:
            return None

        # Get max slots from first link's spectrum
        first_link = (path[0], path[1])
        link_spectrum = self.get_link_spectrum(first_link)
        max_slots = link_spectrum.get_slot_count(band)

        # Search from end backwards
        for start in range(max_slots - slots_needed, -1, -1):
            end = start + slots_needed
            if self.is_spectrum_available(path, start, end, core, band):
                return start

        return None

    def get_spectrum_utilization(self, band: str | None = None) -> float:
        """
        Calculate overall spectrum utilization.

        Args:
            band: Specific band to check, or None for all bands

        Returns:
            Utilization ratio between 0.0 and 1.0
        """
        total_slots = 0
        used_slots = 0

        # Only count each physical link once (not bidirectional pairs)
        seen_links: set[tuple[str, str]] = set()

        for link, spectrum in self._spectrum.items():
            # Normalize link to avoid double-counting
            normalized = tuple(sorted(link))
            if normalized in seen_links:
                continue
            seen_links.add(normalized)

            bands_to_check = [band] if band else spectrum.get_bands()
            for b in bands_to_check:
                arr = spectrum.get_spectrum_array(b)
                for core_idx in range(arr.shape[0]):
                    total_slots += arr.shape[1]
                    used_slots += int(np.sum(arr[core_idx] != 0))

        if total_slots == 0:
            return 0.0
        return used_slots / total_slots

    # =========================================================================
    # Topology Read Methods
    # =========================================================================

    def get_neighbors(self, node: str) -> list[str]:
        """Get neighboring nodes."""
        return list(self._topology.neighbors(node))

    def has_link(self, link: tuple[str, str]) -> bool:
        """Check if a link exists in the topology."""
        return self._topology.has_edge(link[0], link[1])

    def get_link_length(self, link: tuple[str, str]) -> float:
        """
        Get physical length of a link in km.

        Args:
            link: (source, destination) tuple

        Returns:
            Link length in kilometers

        Raises:
            KeyError: If link does not exist
        """
        return self.get_link_spectrum(link).length_km

    # =========================================================================
    # Private Helpers
    # =========================================================================

    def _get_path_links(self, path: list[str]) -> list[tuple[str, str]]:
        """Convert path to list of link tuples."""
        links: list[tuple[str, str]] = []
        for i in range(len(path) - 1):
            links.append((path[i], path[i + 1]))
        return links

    # =========================================================================
    # Debugging / Introspection
    # =========================================================================

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"NetworkState(nodes={self.node_count}, links={self.link_count}, "
            f"lightpaths={self.lightpath_count}, next_id={self._next_lightpath_id})"
        )
