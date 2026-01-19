"""
Network state management for FUSION simulation.

This module provides:
- LinkSpectrum: Per-link spectrum state management
- NetworkState: Single source of truth for network state during simulation

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

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

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

    Spectrum array values: 0 = free, positive int = lightpath ID,
    negative int = guard band for lightpath with abs(value) ID.

    Attributes:
        link: The link identifier as (source, destination) tuple.
        cores_matrix: Spectrum arrays per band, shape (cores, slots) each.
        usage_count: Number of active spectrum allocations on this link.
        throughput: Cumulative bandwidth served through this link (Gbps).
        link_num: Link index for reference.
        length_km: Physical link length in kilometers.

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

        :param start_slot: First slot index (inclusive).
        :type start_slot: int
        :param end_slot: Last slot index (exclusive).
        :type end_slot: int
        :param core: Core index (0 to cores_per_link - 1).
        :type core: int
        :param band: Band identifier ("c", "l", "s").
        :type band: str
        :return: True if all slots in range are 0 (free), False otherwise.
        :rtype: bool
        :raises KeyError: If band is not in cores_matrix.
        :raises IndexError: If core or slot indices are out of bounds.
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
        Increments usage_count on successful allocation.

        :param start_slot: First slot index (inclusive).
        :type start_slot: int
        :param end_slot: Last slot index (exclusive), includes guard slots.
        :type end_slot: int
        :param core: Core index.
        :type core: int
        :param band: Band identifier.
        :type band: str
        :param lightpath_id: Positive integer ID for the lightpath.
        :type lightpath_id: int
        :param guard_slots: Number of guard band slots at end of allocation.
        :type guard_slots: int
        :raises ValueError: If lightpath_id <= 0 or range is not free.
        :raises KeyError: If band is not in cores_matrix.
        :raises IndexError: If indices are out of bounds.
        """
        if lightpath_id <= 0:
            msg = f"lightpath_id must be positive, got {lightpath_id}"
            raise ValueError(msg)

        if not self.is_range_free(start_slot, end_slot, core, band):
            msg = f"Spectrum range [{start_slot}:{end_slot}] on core {core} band {band} is not free"
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

        Sets spectrum slots to 0 and decrements usage_count (clamped to >= 0).

        :param start_slot: First slot index (inclusive).
        :type start_slot: int
        :param end_slot: Last slot index (exclusive).
        :type end_slot: int
        :param core: Core index.
        :type core: int
        :param band: Band identifier.
        :type band: str
        :raises KeyError: If band is not in cores_matrix.
        :raises IndexError: If indices are out of bounds.
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
        Decrements usage_count on successful release.

        :param lightpath_id: The lightpath ID to release.
        :type lightpath_id: int
        :param band: Band identifier.
        :type band: str
        :param core: Core index.
        :type core: int
        :return: Tuple of (start_slot, end_slot) if found, None if not found.
        :rtype: tuple[int, int] | None
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

        Returns actual array reference. Callers should NOT modify directly.
        Use allocate_range() and release_range() instead.

        :param band: Band identifier.
        :type band: str
        :return: numpy array of shape (cores, slots), values are lightpath IDs or 0.
        :rtype: npt.NDArray[np.int64]
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

        :param band: Band identifier.
        :type band: str
        :param core: Core index.
        :type core: int
        :return: Number of free (value == 0) slots.
        :rtype: int
        """
        spectrum = self.cores_matrix[band]
        return int(np.sum(spectrum[core] == self.FREE_SLOT))

    def get_fragmentation_ratio(self, band: str, core: int) -> float:
        """
        Calculate fragmentation ratio for a core/band.

        Fragmentation = 1 - (largest_contiguous_free / total_free).
        Value of 0 means no fragmentation, 1 means maximum fragmentation.

        :param band: Band identifier.
        :type band: str
        :param core: Core index.
        :type core: int
        :return: Fragmentation ratio between 0.0 and 1.0.
        :rtype: float
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

        :param link: Link identifier tuple.
        :type link: tuple[str, str]
        :param config: Simulation configuration with band/core specs.
        :type config: SimulationConfig
        :param link_num: Link index.
        :type link_num: int
        :param length_km: Physical link length.
        :type length_km: float
        :return: New LinkSpectrum with zero-initialized spectrum arrays
            matching config.band_list and config.cores_per_link.
        :rtype: LinkSpectrum
        """
        cores_matrix: dict[str, npt.NDArray[np.int64]] = {}

        for band in config.band_list:
            # TODO(v6.1): Hardcoded default of 320 slots - should require explicit config.
            # WARNING: This silently uses 320 slots if band not in config.band_slots.
            num_slots = config.band_slots.get(band, 320)
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

    # TODO(v6.1): Remove legacy adapter method once migration is complete.
    def to_legacy_dict(self) -> dict[str, Any]:
        """
        Convert to legacy network_spectrum_dict entry format.

        :return: Dictionary compatible with legacy sdn_props.network_spectrum_dict[link].
        :rtype: dict[str, Any]
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

    Owns network topology (read-only after init), per-link spectrum state
    (LinkSpectrum objects), active lightpaths (by ID), and lightpath ID counter.

    One instance per simulation, passed by reference to pipelines. All state
    mutations through explicit methods. Read methods are query-only (no side
    effects). This is a state container, NOT a routing algorithm.

    .. note::
        Request queue, request status tracking, statistics, and per-request
        temporary lists are managed elsewhere (SimulationEngine, SimStats,
        SDNController).

    Example:
        >>> state = NetworkState(topology, config)
        >>> if state.is_spectrum_available(path, 0, 10, core=0, band="c"):
        ...     # Spectrum is free, can allocate
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

        Topology is stored as reference (not copied). Callers must
        not modify topology after passing to NetworkState.

        :param topology: NetworkX graph representing network structure.
            Edge attributes are preserved (length, etc.).
        :type topology: nx.Graph
        :param config: Immutable simulation configuration.
        :type config: SimulationConfig
        :raises ValueError: If topology is empty (no edges) or config band_list is empty.
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
        """Initialize LinkSpectrum for all edges in topology.

        Uses topology_info["links"] for link_num assignment to match legacy
        behavior where link_num corresponds to entries in the topology JSON.
        Falls back to edge iteration order if topology_info is not available.
        """
        topology_info = self._config.topology_info
        links_info = topology_info.get("links", {}) if topology_info else {}

        if links_info:
            # Use topology_info links for link_num assignment (matches legacy)
            for link_num_key, link_data in links_info.items():
                link_num = int(link_num_key)
                source = str(link_data["source"])
                dest = str(link_data["destination"])
                length_km = float(link_data.get("length", 0.0))

                # Create single LinkSpectrum for both directions
                link_spectrum = LinkSpectrum.from_config(
                    link=(source, dest),
                    config=self._config,
                    link_num=link_num,
                    length_km=length_km,
                )

                # Both directions point to same object (bidirectional)
                self._spectrum[(source, dest)] = link_spectrum
                self._spectrum[(dest, source)] = link_spectrum
        else:
            # Fallback: iterate over topology edges (original behavior)
            link_num = 0
            for u, v, edge_data in self._topology.edges(data=True):
                # Get link length from edge data (try multiple common attribute names)
                length_km = float(edge_data.get("length", edge_data.get("weight", edge_data.get("distance", 0.0))))

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
        return int(self._topology.number_of_edges())

    @property
    def next_lightpath_id(self) -> int:
        """Next lightpath ID that will be assigned (read-only)."""
        return self._next_lightpath_id

    @property
    def node_count(self) -> int:
        """Number of nodes in topology."""
        return int(self._topology.number_of_nodes())

    # =========================================================================
    # Lightpath Read Methods
    # =========================================================================

    def get_lightpath(self, lightpath_id: int) -> Lightpath | None:
        """
        Retrieve a lightpath by ID.

        Returns direct reference. Callers may read Lightpath state
        but should use NetworkState methods to modify it.

        :param lightpath_id: Unique lightpath identifier.
        :type lightpath_id: int
        :return: Lightpath object if found, None otherwise.
        :rtype: Lightpath | None
        """
        return self._lightpaths.get(lightpath_id)

    def get_lightpaths_between(
        self,
        source: str,
        destination: str,
    ) -> list[Lightpath]:
        """
        Get all lightpaths between two endpoints.

        Handles both (src, dst) and (dst, src) automatically.

        :param source: Source node ID.
        :type source: str
        :param destination: Destination node ID.
        :type destination: str
        :return: List of lightpaths connecting these endpoints (either direction).
            Empty list if none found.
        :rtype: list[Lightpath]
        """
        result: list[Lightpath] = []
        for lp in self._lightpaths.values():
            # Check both directions
            if (lp.source == source and lp.destination == destination) or (lp.source == destination and lp.destination == source):
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

        :param source: Source node ID.
        :type source: str
        :param destination: Destination node ID.
        :type destination: str
        :param min_bandwidth_gbps: Minimum remaining bandwidth required.
        :type min_bandwidth_gbps: int
        :return: List of lightpaths with at least min_bandwidth_gbps available.
            Sorted by remaining bandwidth descending (best candidates first).
        :rtype: list[Lightpath]
        """
        candidates = self.get_lightpaths_between(source, destination)
        with_capacity = [lp for lp in candidates if lp.remaining_bandwidth_gbps >= min_bandwidth_gbps]
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

        :param link: (source, destination) tuple.
        :type link: tuple[str, str]
        :return: List of lightpaths that include this link in their path.
        :rtype: list[Lightpath]
        """
        result: list[Lightpath] = []
        u, v = link
        for lp in self._lightpaths.values():
            path = lp.path
            for i in range(len(path) - 1):
                if (path[i] == u and path[i + 1] == v) or (path[i] == v and path[i + 1] == u):
                    result.append(lp)
                    break
        return result

    def iter_lightpaths(self) -> Iterator[Lightpath]:
        """
        Iterate over all active lightpaths.

        Safe to iterate during simulation. Do not add/remove
        lightpaths during iteration.

        :yields: Each active Lightpath in arbitrary order.
        :rtype: Iterator[Lightpath]
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

        Both (u, v) and (v, u) return the same LinkSpectrum object.

        :param link: (source, destination) tuple.
        :type link: tuple[str, str]
        :return: LinkSpectrum object for that link.
        :rtype: LinkSpectrum
        :raises KeyError: If link does not exist in topology.
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

        :param path: Ordered list of node IDs forming the path.
        :type path: list[str]
        :param start_slot: First slot index (inclusive).
        :type start_slot: int
        :param end_slot: Last slot index (exclusive).
        :type end_slot: int
        :param core: Core index.
        :type core: int
        :param band: Band identifier.
        :type band: str
        :return: True if all slots in range are free on all links in path.
        :rtype: bool
        :raises ValueError: If path has fewer than 2 nodes.
        :raises KeyError: If any link in path doesn't exist.
        """
        if len(path) < 2:
            msg = "Path must have at least 2 nodes"
            raise ValueError(msg)

        for link in self._get_path_links(path):
            link_spectrum = self.get_link_spectrum(link)
            if not link_spectrum.is_range_free(start_slot, end_slot, core, band):
                return False
        return True

    # TODO(v6.1): Move find_first_fit/find_last_fit to spectrum assignment module.
    # These algorithms don't belong in a state container - NetworkState should
    # only manage state, not implement allocation strategies.
    def find_first_fit(
        self,
        path: list[str],
        slots_needed: int,
        core: int,
        band: str,
    ) -> int | None:
        """
        Find first available slot range along path (first-fit algorithm).

        This is a read-only query method. It does NOT allocate.
        More sophisticated algorithms should be in pipelines.

        :param path: Ordered list of node IDs.
        :type path: list[str]
        :param slots_needed: Number of contiguous slots required.
        :type slots_needed: int
        :param core: Core index to search.
        :type core: int
        :param band: Band identifier.
        :type band: str
        :return: Start slot index if found, None if no fit available.
        :rtype: int | None
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

        This is a read-only query method. It does NOT allocate.

        :param path: Ordered list of node IDs.
        :type path: list[str]
        :param slots_needed: Number of contiguous slots required.
        :type slots_needed: int
        :param core: Core index to search.
        :type core: int
        :param band: Band identifier.
        :type band: str
        :return: Start slot index if found, None if no fit available.
        :rtype: int | None
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

        :param band: Specific band to check, or None for all bands.
        :type band: str | None
        :return: Utilization ratio between 0.0 and 1.0.
        :rtype: float
        """
        total_slots = 0
        used_slots = 0

        # Only count each physical link once (not bidirectional pairs)
        seen_links: set[tuple[str, str]] = set()

        for link, spectrum in self._spectrum.items():
            # Normalize link to avoid double-counting
            sorted_link = sorted(link)
            normalized: tuple[str, str] = (sorted_link[0], sorted_link[1])
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
        return bool(self._topology.has_edge(link[0], link[1]))

    def get_link_length(self, link: tuple[str, str]) -> float:
        """
        Get physical length of a link in km.

        :param link: (source, destination) tuple.
        :type link: tuple[str, str]
        :return: Link length in kilometers.
        :rtype: float
        :raises KeyError: If link does not exist.
        """
        return self.get_link_spectrum(link).length_km

    def get_link_utilization(self, link: tuple[str, str]) -> float:
        """
        Get spectrum utilization for a link as a fraction (0.0 to 1.0).

        Calculates the ratio of occupied slots to total slots across all
        bands and cores. A slot is considered occupied if its value is non-zero
        (positive for lightpath data, negative for guard bands).

        :param link: (source, destination) tuple.
        :type link: tuple[str, str]
        :return: Utilization fraction between 0.0 (empty) and 1.0 (full).
        :rtype: float
        :raises KeyError: If link does not exist.
        """
        link_spectrum = self.get_link_spectrum(link)

        total_slots = 0
        occupied_slots = 0

        for band_spectrum in link_spectrum.cores_matrix.values():
            total_slots += band_spectrum.size
            occupied_slots += int(np.count_nonzero(band_spectrum))

        if total_slots == 0:
            return 0.0

        return occupied_slots / total_slots

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

    def reset(self) -> None:
        """Reset network state for new RL episode.

        Clears:
        - All spectrum allocations (sets arrays to zero)
        - Active lightpaths (clears registry)
        - Resets lightpath ID counter

        Preserves:
        - Topology structure
        - Configuration
        - Link spectrum objects (just cleared, not recreated)

        This method is used by UnifiedSimEnv.reset() to prepare for a new
        episode without needing to recreate the entire NetworkState object.

        Example:
            >>> state = NetworkState(topology, config)
            >>> # ... run episode ...
            >>> state.reset()  # Clear for new episode
        """
        # Clear all spectrum allocations
        for link_spectrum in self._spectrum.values():
            for _band, arr in link_spectrum.cores_matrix.items():
                arr.fill(0)
            link_spectrum.usage_count = 0
            link_spectrum.throughput = 0.0

        # Clear active lightpaths
        self._lightpaths.clear()

        # Reset lightpath ID counter
        self._next_lightpath_id = 1

    # =========================================================================
    # Write Methods
    # =========================================================================

    def create_lightpath(
        self,
        path: list[str],
        start_slot: int,
        end_slot: int,
        core: int,
        band: str,
        modulation: str,
        bandwidth_gbps: int,
        path_weight_km: float,
        guard_slots: int = 0,
        *,
        backup_path: list[str] | None = None,
        backup_start_slot: int | None = None,
        backup_end_slot: int | None = None,
        backup_core: int | None = None,
        backup_band: str | None = None,
        snr_db: float | None = None,
        xt_cost: float | None = None,
        connection_index: int | None = None,
        arrive_time: float | None = None,
    ) -> Lightpath:
        """
        Create and register a new lightpath, allocating spectrum.

        :param path: Ordered list of node IDs (minimum 2 nodes).
        :type path: list[str]
        :param start_slot: First spectrum slot (inclusive).
        :type start_slot: int
        :param end_slot: Last spectrum slot (exclusive, includes guard slots).
        :type end_slot: int
        :param core: Core index for allocation.
        :type core: int
        :param band: Band identifier ("c", "l", "s").
        :type band: str
        :param modulation: Modulation format name.
        :type modulation: str
        :param bandwidth_gbps: Total lightpath capacity.
        :type bandwidth_gbps: int
        :param path_weight_km: Path length/weight in kilometers.
        :type path_weight_km: float
        :param guard_slots: Number of guard band slots at end (default 0).
        :type guard_slots: int
        :param backup_path: Optional backup path for 1+1 protection.
        :type backup_path: list[str] | None
        :param backup_start_slot: Backup path start slot.
        :type backup_start_slot: int | None
        :param backup_end_slot: Backup path end slot.
        :type backup_end_slot: int | None
        :param backup_core: Backup path core index.
        :type backup_core: int | None
        :param backup_band: Backup path band.
        :type backup_band: str | None
        :param snr_db: Measured SNR value (optional).
        :type snr_db: float | None
        :param xt_cost: Crosstalk cost (optional).
        :type xt_cost: float | None
        :param connection_index: External routing index for SNR lookup (optional).
        :type connection_index: int | None
        :param arrive_time: Request arrival time for utilization tracking (optional).
        :type arrive_time: float | None
        :return: Newly created Lightpath with assigned lightpath_id.
        :rtype: Lightpath
        :raises ValueError: If parameters are invalid or spectrum unavailable.
        :raises KeyError: If any link in path doesn't exist in topology.
        """
        # Validate parameters
        self._validate_create_lightpath_params(
            path=path,
            start_slot=start_slot,
            end_slot=end_slot,
            bandwidth_gbps=bandwidth_gbps,
            backup_path=backup_path,
            backup_start_slot=backup_start_slot,
            backup_end_slot=backup_end_slot,
            backup_core=backup_core,
            backup_band=backup_band,
        )

        # Check primary path spectrum availability
        if not self.is_spectrum_available(path, start_slot, end_slot, core, band):
            msg = f"Spectrum [{start_slot}:{end_slot}] not available on primary path"
            raise ValueError(msg)

        # Check backup path spectrum availability (if protected)
        is_protected = backup_path is not None
        if is_protected and backup_path is not None:
            assert backup_start_slot is not None
            assert backup_end_slot is not None
            assert backup_core is not None
            assert backup_band is not None

            if not self.is_spectrum_available(backup_path, backup_start_slot, backup_end_slot, backup_core, backup_band):
                msg = f"Spectrum [{backup_start_slot}:{backup_end_slot}] not available on backup path"
                raise ValueError(msg)

        # Allocate primary path spectrum
        lightpath_id = self._next_lightpath_id
        self._allocate_spectrum_on_path(
            path=path,
            start_slot=start_slot,
            end_slot=end_slot,
            core=core,
            band=band,
            lightpath_id=lightpath_id,
            guard_slots=guard_slots,
        )

        # Allocate backup path spectrum (if protected)
        if is_protected:
            self._allocate_spectrum_on_path(
                path=backup_path,  # type: ignore[arg-type]
                start_slot=backup_start_slot,  # type: ignore[arg-type]
                end_slot=backup_end_slot,  # type: ignore[arg-type]
                core=backup_core,  # type: ignore[arg-type]
                band=backup_band,  # type: ignore[arg-type]
                lightpath_id=lightpath_id,
                guard_slots=guard_slots,
            )

        # Create Lightpath object
        # Note: Lightpath stores data slot range (excludes guard slots)
        data_end_slot = end_slot - guard_slots
        backup_data_end_slot = backup_end_slot - guard_slots if backup_end_slot is not None else None

        # Set initial time_bw_usage for utilization tracking
        # New lightpaths start with 0% utilization (no requests allocated yet)
        initial_time_bw_usage: dict[float, float] = {}
        if arrive_time is not None:
            initial_time_bw_usage[arrive_time] = 0.0

        lightpath = Lightpath(
            lightpath_id=lightpath_id,
            path=path,
            start_slot=start_slot,
            end_slot=data_end_slot,
            core=core,
            band=band,
            modulation=modulation,
            total_bandwidth_gbps=bandwidth_gbps,
            remaining_bandwidth_gbps=bandwidth_gbps,
            path_weight_km=path_weight_km,
            request_allocations={},
            time_bw_usage=initial_time_bw_usage,
            snr_db=snr_db,
            xt_cost=xt_cost,
            is_degraded=False,
            connection_index=connection_index,
            backup_path=backup_path,
            backup_start_slot=backup_start_slot,
            backup_end_slot=backup_data_end_slot,
            backup_core=backup_core,
            backup_band=backup_band,
            is_protected=is_protected,
            active_path="primary",
        )

        # Register lightpath
        self._lightpaths[lightpath_id] = lightpath
        self._next_lightpath_id += 1

        return lightpath

    def release_lightpath(self, lightpath_id: int) -> bool:
        """
        Release a lightpath and free its spectrum.

        :param lightpath_id: ID of lightpath to release.
        :type lightpath_id: int
        :return: True if lightpath was found and released, False if not found.
        :rtype: bool
        """
        lightpath = self._lightpaths.get(lightpath_id)
        if lightpath is None:
            return False

        # Calculate actual end slot (including guard bands)
        # Guard slots = total allocated - data slots stored in lightpath
        guard_slots = getattr(self._config, "guard_slots", 0)
        primary_end_with_guards = lightpath.end_slot + guard_slots

        # Release primary path spectrum
        self._release_spectrum_on_path(
            path=lightpath.path,
            start_slot=lightpath.start_slot,
            end_slot=primary_end_with_guards,
            core=lightpath.core,
            band=lightpath.band,
        )

        # Release backup path spectrum (if protected)
        if lightpath.is_protected and lightpath.backup_path is not None:
            assert lightpath.backup_start_slot is not None
            assert lightpath.backup_end_slot is not None
            assert lightpath.backup_core is not None
            assert lightpath.backup_band is not None

            backup_end_with_guards = lightpath.backup_end_slot + guard_slots
            self._release_spectrum_on_path(
                path=lightpath.backup_path,
                start_slot=lightpath.backup_start_slot,
                end_slot=backup_end_with_guards,
                core=lightpath.backup_core,
                band=lightpath.backup_band,
            )

        # Remove from registry
        del self._lightpaths[lightpath_id]

        return True

    # =========================================================================
    # Bandwidth Management Methods
    # =========================================================================

    def allocate_request_bandwidth(
        self,
        lightpath_id: int,
        request_id: int,
        bandwidth_gbps: int,
        timestamp: float | None = None,
    ) -> None:
        """
        Allocate bandwidth on existing lightpath to a request.

        Used by grooming to share lightpath capacity among requests.

        :param lightpath_id: ID of the lightpath.
        :type lightpath_id: int
        :param request_id: ID of the request.
        :type request_id: int
        :param bandwidth_gbps: Bandwidth to allocate.
        :type bandwidth_gbps: int
        :param timestamp: Optional arrival time for utilization tracking.
        :type timestamp: float | None
        :raises KeyError: If lightpath not found.
        :raises ValueError: If insufficient remaining bandwidth or request already allocated.
        """
        lightpath = self._lightpaths.get(lightpath_id)
        if lightpath is None:
            msg = f"Lightpath {lightpath_id} not found"
            raise KeyError(msg)

        # Delegate to Lightpath's allocate_bandwidth method
        success = lightpath.allocate_bandwidth(request_id, bandwidth_gbps, timestamp)
        if not success:
            msg = f"Insufficient bandwidth: need {bandwidth_gbps}, have {lightpath.remaining_bandwidth_gbps}"
            raise ValueError(msg)

    def release_request_bandwidth(
        self,
        lightpath_id: int,
        request_id: int,
        timestamp: float | None = None,
    ) -> int:
        """
        Release bandwidth allocated to a request.

        :param lightpath_id: ID of the lightpath.
        :type lightpath_id: int
        :param request_id: ID of the request to release.
        :type request_id: int
        :param timestamp: Optional departure time for utilization tracking.
        :type timestamp: float | None
        :return: Amount of bandwidth released.
        :rtype: int
        :raises KeyError: If lightpath or request not found.
        """
        lightpath = self._lightpaths.get(lightpath_id)
        if lightpath is None:
            msg = f"Lightpath {lightpath_id} not found"
            raise KeyError(msg)

        # Delegate to Lightpath's release_bandwidth method
        return lightpath.release_bandwidth(request_id, timestamp)

    # =========================================================================
    # Private Helpers for Write Methods
    # =========================================================================

    def _validate_create_lightpath_params(
        self,
        *,
        path: list[str],
        start_slot: int,
        end_slot: int,
        bandwidth_gbps: int,
        backup_path: list[str] | None,
        backup_start_slot: int | None,
        backup_end_slot: int | None,
        backup_core: int | None,
        backup_band: str | None,
    ) -> None:
        """Validate create_lightpath parameters."""
        if len(path) < 2:
            msg = "Path must have at least 2 nodes"
            raise ValueError(msg)

        if start_slot >= end_slot:
            msg = f"Invalid slot range: start={start_slot} >= end={end_slot}"
            raise ValueError(msg)

        if bandwidth_gbps <= 0:
            msg = f"Bandwidth must be positive, got {bandwidth_gbps}"
            raise ValueError(msg)

        # Protection validation: all or nothing
        protection_fields = [
            backup_path,
            backup_start_slot,
            backup_end_slot,
            backup_core,
            backup_band,
        ]
        has_any_protection = any(f is not None for f in protection_fields)
        has_all_protection = all(f is not None for f in protection_fields)

        if has_any_protection and not has_all_protection:
            msg = "Incomplete protection specification: provide all backup fields or none"
            raise ValueError(msg)

        if backup_path is not None and len(backup_path) < 2:
            msg = "Backup path must have at least 2 nodes"
            raise ValueError(msg)

    def _allocate_spectrum_on_path(
        self,
        path: list[str],
        start_slot: int,
        end_slot: int,
        core: int,
        band: str,
        lightpath_id: int,
        guard_slots: int = 0,
    ) -> None:
        """
        Allocate spectrum range on all links in path.

        Assumes spectrum availability has already been verified.
        """
        for link in self._get_path_links(path):
            link_spectrum = self.get_link_spectrum(link)
            link_spectrum.allocate_range(
                start_slot=start_slot,
                end_slot=end_slot,
                core=core,
                band=band,
                lightpath_id=lightpath_id,
                guard_slots=guard_slots,
            )

    def _release_spectrum_on_path(
        self,
        path: list[str],
        start_slot: int,
        end_slot: int,
        core: int,
        band: str,
    ) -> None:
        """Release spectrum range on all links in path."""
        for link in self._get_path_links(path):
            link_spectrum = self.get_link_spectrum(link)
            link_spectrum.release_range(
                start_slot=start_slot,
                end_slot=end_slot,
                core=core,
                band=band,
            )

    # =========================================================================
    # Legacy Compatibility Properties
    # TODO(v6.1): Remove legacy compatibility properties once migration is complete.
    # WARNING: These properties are TEMPORARY MIGRATION SHIMS.
    # =========================================================================

    @property
    def network_spectrum_dict(self) -> dict[tuple[str, str], dict[str, Any]]:
        """
        Legacy compatibility: Returns spectrum state in sdn_props format.

        .. warning::
            This property is a TEMPORARY MIGRATION SHIM.

        Both (u,v) and (v,u) return the SAME dict object. cores_matrix arrays
        are direct references (not copies). Edge attributes from topology are
        included.

        :return: Dictionary mapping link tuples to link state dicts.
            Format matches sdn_props.network_spectrum_dict exactly.
        :rtype: dict[tuple[str, str], dict[str, Any]]
        """
        result: dict[tuple[str, str], dict[str, Any]] = {}
        seen: set[frozenset[str]] = set()

        for link, link_spectrum in self._spectrum.items():
            # Avoid duplicate processing for bidirectional links
            link_set = frozenset(link)
            if link_set in seen:
                # Add reverse direction pointing to same dict
                reverse = (link[1], link[0])
                if link in result and reverse not in result:
                    result[reverse] = result[link]
                continue
            seen.add(link_set)

            # Build legacy format dict
            u, v = link
            link_dict: dict[str, Any] = {
                "cores_matrix": link_spectrum.cores_matrix,  # Direct reference
                "usage_count": link_spectrum.usage_count,
                "throughput": link_spectrum.throughput,
                "link_num": link_spectrum.link_num,
            }

            # Add edge attributes from topology
            if self._topology.has_edge(u, v):
                edge_data = self._topology.edges[u, v]
                for attr in ["length", "dispersion", "attenuation", "fiber_type"]:
                    if attr in edge_data:
                        link_dict[attr] = edge_data[attr]

            # Both directions point to same dict
            result[(u, v)] = link_dict
            result[(v, u)] = link_dict

        return result

    @property
    def lightpath_status_dict(self) -> dict[tuple[str, str], dict[int, dict[str, Any]]]:
        """
        Legacy compatibility: Returns lightpath state in sdn_props format.

        .. warning::
            This property is a TEMPORARY MIGRATION SHIM.

        Keys are SORTED tuples ("A", "C") not ("C", "A"). Bandwidth values are
        float (legacy format). time_bw_usage is empty dict by default. Rebuilds
        dict on each access (not cached).

        :return: Dictionary mapping sorted endpoint tuples to lightpath dicts.
            Format matches sdn_props.lightpath_status_dict exactly.
        :rtype: dict[tuple[str, str], dict[int, dict[str, Any]]]
        """
        result: dict[tuple[str, str], dict[int, dict[str, Any]]] = {}

        for lp_id, lp in self._lightpaths.items():
            # Key is SORTED tuple
            sorted_endpoints = sorted([lp.source, lp.destination])
            key: tuple[str, str] = (sorted_endpoints[0], sorted_endpoints[1])

            # Initialize group if needed
            if key not in result:
                result[key] = {}

            # Build legacy format entry with all fields from gap analysis
            result[key][lp_id] = {
                "path": lp.path,
                "lightpath_bandwidth": float(lp.total_bandwidth_gbps),
                "remaining_bandwidth": float(lp.remaining_bandwidth_gbps),
                "band": lp.band,
                "core": lp.core,
                "start_slot": lp.start_slot,
                # Convert exclusive end_slot to inclusive (legacy format)
                # Legacy uses inclusive when guard_slots=0, exclusive-like when guard_slots>0
                # For legacy compat, subtract 1.
                "end_slot": lp.end_slot - 1,
                "modulation": lp.modulation,
                "mod_format": lp.modulation,  # Alternate key for modulation
                "requests_dict": {req_id: float(bw) for req_id, bw in lp.request_allocations.items()},
                "time_bw_usage": {},  # Empty by default, populated during simulation
                "is_degraded": lp.is_degraded,
                # Additional fields from gap analysis
                "path_weight": lp.path_weight_km,
                "snr_cost": lp.snr_db,
                "xt_cost": lp.xt_cost,
            }

        return result
