"""
Network state management for FUSION simulation.

This module provides:
- LinkSpectrum: Per-link spectrum state management
- NetworkState: Single source of truth for network state during simulation

Phase: P2.1 - NetworkState Core (read methods)
Phase: P2.2 - Write Methods & Legacy Compatibility (write methods, legacy properties)

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

    # =========================================================================
    # Write Methods (P2.2)
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
    ) -> Lightpath:
        """
        Create and register a new lightpath, allocating spectrum.

        Args:
            path: Ordered list of node IDs (minimum 2 nodes)
            start_slot: First spectrum slot (inclusive)
            end_slot: Last spectrum slot (exclusive, includes guard slots)
            core: Core index for allocation
            band: Band identifier ("c", "l", "s")
            modulation: Modulation format name
            bandwidth_gbps: Total lightpath capacity
            path_weight_km: Path length/weight in kilometers
            guard_slots: Number of guard band slots at end (default 0)
            backup_path: Optional backup path for 1+1 protection
            backup_start_slot: Backup path start slot
            backup_end_slot: Backup path end slot
            backup_core: Backup path core index
            backup_band: Backup path band
            snr_db: Measured SNR value (optional)
            xt_cost: Crosstalk cost (optional)

        Returns:
            Newly created Lightpath with assigned lightpath_id

        Raises:
            ValueError: If parameters are invalid or spectrum unavailable
            KeyError: If any link in path doesn't exist in topology
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
        if is_protected:
            assert backup_start_slot is not None
            assert backup_end_slot is not None
            assert backup_core is not None
            assert backup_band is not None

            if not self.is_spectrum_available(
                backup_path, backup_start_slot, backup_end_slot, backup_core, backup_band
            ):
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
        backup_data_end_slot = (
            backup_end_slot - guard_slots if backup_end_slot is not None else None
        )

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

        Args:
            lightpath_id: ID of lightpath to release

        Returns:
            True if lightpath was found and released, False if not found
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
    # Bandwidth Management Methods (P2.2)
    # =========================================================================

    def allocate_request_bandwidth(
        self,
        lightpath_id: int,
        request_id: int,
        bandwidth_gbps: int,
    ) -> None:
        """
        Allocate bandwidth on existing lightpath to a request.

        Used by grooming to share lightpath capacity among requests.

        Args:
            lightpath_id: ID of the lightpath
            request_id: ID of the request
            bandwidth_gbps: Bandwidth to allocate

        Raises:
            KeyError: If lightpath not found
            ValueError: If insufficient remaining bandwidth or request already allocated
        """
        lightpath = self._lightpaths.get(lightpath_id)
        if lightpath is None:
            msg = f"Lightpath {lightpath_id} not found"
            raise KeyError(msg)

        # Delegate to Lightpath's allocate_bandwidth method
        success = lightpath.allocate_bandwidth(request_id, bandwidth_gbps)
        if not success:
            msg = (
                f"Insufficient bandwidth: need {bandwidth_gbps}, "
                f"have {lightpath.remaining_bandwidth_gbps}"
            )
            raise ValueError(msg)

    def release_request_bandwidth(
        self,
        lightpath_id: int,
        request_id: int,
    ) -> int:
        """
        Release bandwidth allocated to a request.

        Args:
            lightpath_id: ID of the lightpath
            request_id: ID of the request to release

        Returns:
            Amount of bandwidth released

        Raises:
            KeyError: If lightpath or request not found
        """
        lightpath = self._lightpaths.get(lightpath_id)
        if lightpath is None:
            msg = f"Lightpath {lightpath_id} not found"
            raise KeyError(msg)

        # Delegate to Lightpath's release_bandwidth method
        return lightpath.release_bandwidth(request_id)

    # =========================================================================
    # Private Helpers for Write Methods (P2.2)
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
    # Legacy Compatibility Properties (P2.2)
    # WARNING: These properties are TEMPORARY MIGRATION SHIMS.
    # They will be removed in Phase 5.
    # =========================================================================

    @property
    def network_spectrum_dict(self) -> dict[tuple[str, str], dict[str, Any]]:
        """
        Legacy compatibility: Returns spectrum state in sdn_props format.

        WARNING: This property is a TEMPORARY MIGRATION SHIM.
        It will be removed in Phase 5.

        Returns:
            Dictionary mapping link tuples to link state dicts.
            Format matches sdn_props.network_spectrum_dict exactly.

        Notes:
            - Both (u,v) and (v,u) return the SAME dict object
            - cores_matrix arrays are direct references (not copies)
            - Edge attributes from topology are included

        LEGACY_COMPAT: Phase 5 Removal Checklist
        Before removing this property, verify:
        [ ] All spectrum reads use NetworkState.is_spectrum_available()
        [ ] All spectrum writes use NetworkState.create_lightpath()
        [ ] No direct numpy array access outside NetworkState
        [ ] run_comparison.py passes without this property
        [ ] grep 'network_spectrum_dict' returns only this definition
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

        WARNING: This property is a TEMPORARY MIGRATION SHIM.
        It will be removed in Phase 5.

        Returns:
            Dictionary mapping sorted endpoint tuples to lightpath dicts.
            Format matches sdn_props.lightpath_status_dict exactly.

        Notes:
            - Keys are SORTED tuples: ("A", "C") not ("C", "A")
            - Bandwidth values are float (legacy format)
            - time_bw_usage is empty dict by default
            - Rebuilds dict on each access (not cached)

        LEGACY_COMPAT: Phase 5 Removal Checklist
        Before removing this property, verify:
        [ ] Grooming uses NetworkState.get_lightpaths_with_capacity()
        [ ] Statistics use NetworkState.iter_lightpaths()
        [ ] No manual sorted tuple key construction
        [ ] run_comparison.py passes without this property
        [ ] grep 'lightpath_status_dict' returns only this definition
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
                "end_slot": lp.end_slot,
                "modulation": lp.modulation,
                "mod_format": lp.modulation,  # Alternate key for modulation
                "requests_dict": {
                    req_id: float(bw)
                    for req_id, bw in lp.request_allocations.items()
                },
                "time_bw_usage": {},  # Empty by default, populated during simulation
                "is_degraded": lp.is_degraded,
                # Additional fields from gap analysis
                "path_weight": lp.path_weight_km,
                "snr_cost": lp.snr_db,
                "xt_cost": lp.xt_cost,
            }

        return result
