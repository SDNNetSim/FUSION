"""
Lightpath domain model with capacity management.

This module defines the Lightpath dataclass representing an allocated
optical path with spectrum assignment and capacity tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Lightpath:
    """
    Allocated optical path with capacity management.

    A Lightpath represents an established connection through the network
    with allocated spectrum resources. It tracks capacity usage for
    traffic grooming and supports 1+1 protection.

    Attributes:
        Identity:
            lightpath_id: Unique identifier
            path: Ordered list of node IDs from source to destination

        Spectrum Allocation:
            start_slot: First allocated slot index (inclusive)
            end_slot: Last allocated slot index (exclusive)
            core: Core number for MCF (0-indexed)
            band: Frequency band ("c", "l", "s")
            modulation: Modulation format ("QPSK", "16-QAM", etc.)

        Capacity:
            total_bandwidth_gbps: Maximum capacity
            remaining_bandwidth_gbps: Available for new requests
            request_allocations: Maps request_id -> allocated bandwidth

        Quality:
            snr_db: Signal-to-noise ratio (optional)
            xt_cost: Crosstalk cost (optional)
            path_weight_km: Physical path length in km
            is_degraded: True if quality has degraded

        Protection (1+1):
            backup_path: Disjoint backup path (optional)
            backup_start_slot, backup_end_slot: Backup spectrum range
            backup_core, backup_band: Backup allocation details
            is_protected: Has backup path
            active_path: Currently active path ("primary" or "backup")

    Example:
        >>> lp = Lightpath(
        ...     lightpath_id=1,
        ...     path=["0", "2", "5"],
        ...     start_slot=10,
        ...     end_slot=18,
        ...     core=0,
        ...     band="c",
        ...     modulation="QPSK",
        ...     total_bandwidth_gbps=100,
        ...     remaining_bandwidth_gbps=100,
        ... )
        >>> lp.allocate_bandwidth(42, 50)
        True
        >>> lp.utilization
        0.5
    """

    # =========================================================================
    # Identity Fields
    # =========================================================================
    lightpath_id: int
    path: list[str]

    # =========================================================================
    # Spectrum Allocation
    # =========================================================================
    start_slot: int
    end_slot: int
    core: int
    band: str
    modulation: str

    # =========================================================================
    # Capacity
    # =========================================================================
    total_bandwidth_gbps: int
    remaining_bandwidth_gbps: int
    path_weight_km: float = 0.0
    request_allocations: dict[int, int] = field(default_factory=dict)

    # =========================================================================
    # Quality Metrics
    # =========================================================================
    snr_db: float | None = None
    xt_cost: float | None = None
    is_degraded: bool = False

    # =========================================================================
    # Protection (1+1)
    # =========================================================================
    backup_path: list[str] | None = None
    backup_start_slot: int | None = None
    backup_end_slot: int | None = None
    backup_core: int | None = None
    backup_band: str | None = None
    is_protected: bool = False
    active_path: str = "primary"

    # =========================================================================
    # Validation
    # =========================================================================
    def __post_init__(self) -> None:
        """Validate lightpath after creation."""
        if len(self.path) < 2:
            raise ValueError("path must have at least 2 nodes")
        if self.start_slot >= self.end_slot:
            raise ValueError("start_slot must be < end_slot")
        if self.core < 0:
            raise ValueError("core must be >= 0")
        if self.total_bandwidth_gbps <= 0:
            raise ValueError("total_bandwidth_gbps must be > 0")
        if self.remaining_bandwidth_gbps < 0:
            raise ValueError("remaining_bandwidth_gbps must be >= 0")
        if self.remaining_bandwidth_gbps > self.total_bandwidth_gbps:
            raise ValueError(
                "remaining_bandwidth_gbps cannot exceed total_bandwidth_gbps"
            )
        if self.active_path not in ("primary", "backup"):
            raise ValueError("active_path must be 'primary' or 'backup'")

        # Protection consistency
        if self.is_protected and self.backup_path is None:
            raise ValueError("is_protected=True requires backup_path")
        if self.backup_path is not None and not self.is_protected:
            raise ValueError("backup_path requires is_protected=True")

    # =========================================================================
    # Computed Properties
    # =========================================================================
    @property
    def source(self) -> str:
        """Source node ID."""
        return self.path[0]

    @property
    def destination(self) -> str:
        """Destination node ID."""
        return self.path[-1]

    @property
    def endpoint_key(self) -> tuple[str, str]:
        """
        Canonical endpoint pair for lightpath matching.

        Returns sorted tuple to ensure bidirectional matching.
        """
        return tuple(sorted([self.source, self.destination]))  # type: ignore[return-value]

    @property
    def num_slots(self) -> int:
        """Number of spectrum slots used."""
        return self.end_slot - self.start_slot

    @property
    def num_hops(self) -> int:
        """Number of links in the path."""
        return len(self.path) - 1

    @property
    def utilization(self) -> float:
        """
        Bandwidth utilization ratio.

        Returns:
            Float between 0.0 and 1.0 representing capacity usage.
        """
        if self.total_bandwidth_gbps == 0:
            return 0.0
        used = self.total_bandwidth_gbps - self.remaining_bandwidth_gbps
        return used / self.total_bandwidth_gbps

    @property
    def has_capacity(self) -> bool:
        """True if lightpath can accept more traffic."""
        return self.remaining_bandwidth_gbps > 0

    @property
    def num_requests(self) -> int:
        """Number of requests using this lightpath."""
        return len(self.request_allocations)

    @property
    def is_empty(self) -> bool:
        """True if no requests are using this lightpath."""
        return len(self.request_allocations) == 0

    # =========================================================================
    # Capacity Management
    # =========================================================================
    def can_accommodate(self, bandwidth_gbps: int) -> bool:
        """
        Check if lightpath can accommodate the requested bandwidth.

        Args:
            bandwidth_gbps: Bandwidth to check

        Returns:
            True if sufficient capacity available
        """
        return self.remaining_bandwidth_gbps >= bandwidth_gbps

    def allocate_bandwidth(self, request_id: int, bandwidth_gbps: int) -> bool:
        """
        Allocate bandwidth to a request.

        Args:
            request_id: ID of the request
            bandwidth_gbps: Bandwidth to allocate

        Returns:
            True if allocation successful, False if insufficient capacity

        Raises:
            ValueError: If request_id already has an allocation or bandwidth <= 0
        """
        if request_id in self.request_allocations:
            raise ValueError(
                f"Request {request_id} already has allocation on this lightpath"
            )

        if bandwidth_gbps <= 0:
            raise ValueError("bandwidth_gbps must be > 0")

        if not self.can_accommodate(bandwidth_gbps):
            return False

        self.request_allocations[request_id] = bandwidth_gbps
        self.remaining_bandwidth_gbps -= bandwidth_gbps
        return True

    def release_bandwidth(self, request_id: int) -> int:
        """
        Release bandwidth from a request.

        Args:
            request_id: ID of the request to release

        Returns:
            Amount of bandwidth released

        Raises:
            KeyError: If request_id has no allocation
        """
        if request_id not in self.request_allocations:
            raise KeyError(f"Request {request_id} has no allocation on this lightpath")

        bandwidth = self.request_allocations.pop(request_id)
        self.remaining_bandwidth_gbps += bandwidth
        return bandwidth

    def get_allocation(self, request_id: int) -> int | None:
        """
        Get bandwidth allocated to a request.

        Args:
            request_id: ID of the request

        Returns:
            Allocated bandwidth or None if no allocation
        """
        return self.request_allocations.get(request_id)

    # =========================================================================
    # Protection
    # =========================================================================
    def switch_to_backup(self) -> bool:
        """
        Switch to backup path (for failure recovery).

        Returns:
            True if switch successful, False if not protected

        Raises:
            ValueError: If already on backup path
        """
        if not self.is_protected:
            return False

        if self.active_path == "backup":
            raise ValueError("Already on backup path")

        self.active_path = "backup"
        return True

    def switch_to_primary(self) -> bool:
        """
        Switch back to primary path (after recovery).

        Returns:
            True if switch successful, False if not protected

        Raises:
            ValueError: If already on primary path
        """
        if not self.is_protected:
            return False

        if self.active_path == "primary":
            raise ValueError("Already on primary path")

        self.active_path = "primary"
        return True

    @property
    def current_path(self) -> list[str]:
        """Get currently active path."""
        if self.active_path == "backup" and self.backup_path is not None:
            return self.backup_path
        return self.path

    # =========================================================================
    # Legacy Adapters
    # =========================================================================
    @classmethod
    def from_legacy_dict(
        cls,
        lightpath_id: int,
        lp_info: dict[str, Any],
    ) -> Lightpath:
        """
        Create Lightpath from legacy dictionary.

        The legacy format stores lightpath info in nested dictionaries
        indexed by endpoint pair and lightpath ID.

        Args:
            lightpath_id: Unique lightpath identifier
            lp_info: Legacy lightpath info dictionary with fields:
                - "path": list[str] - Node IDs in path
                - "start_slot": int - First slot index
                - "end_slot": int - Last slot index (exclusive)
                - "core": int - Core number
                - "band": str - Frequency band
                - "mod_format": str - Modulation format
                - "lightpath_bandwidth": float - Total bandwidth
                - "remaining_bandwidth": float - Available bandwidth
                - "requests_dict": dict - Request allocations (optional)
                - "snr_cost": float - SNR value (optional)
                - "xt_cost": float - Crosstalk (optional)
                - "path_weight": float - Path length (optional)
                - "is_degraded": bool - Degradation flag (optional)
                - "backup_path": list[str] - Backup path (optional)
                - "is_protected": bool - Protection flag (optional)
                - "active_path": str - Active path (optional)

        Returns:
            New Lightpath instance

        Example:
            >>> lp_info = {
            ...     "path": ["0", "2", "5"],
            ...     "start_slot": 10,
            ...     "end_slot": 18,
            ...     "core": 0,
            ...     "band": "c",
            ...     "mod_format": "QPSK",
            ...     "lightpath_bandwidth": 100.0,
            ...     "remaining_bandwidth": 50.0,
            ...     "requests_dict": {42: 50.0},
            ... }
            >>> lp = Lightpath.from_legacy_dict(1, lp_info)
            >>> lp.lightpath_id
            1
        """
        # Parse request allocations (convert float -> int if needed)
        requests_dict = lp_info.get("requests_dict", {})
        request_allocations = {
            int(req_id): int(bw) for req_id, bw in requests_dict.items()
        }

        # Handle protection fields
        backup_path = lp_info.get("backup_path")
        is_protected = lp_info.get("is_protected", False)

        # Ensure protection consistency - if backup_path is present, set is_protected
        if backup_path is not None and not is_protected:
            is_protected = True

        return cls(
            lightpath_id=lightpath_id,
            path=lp_info["path"],
            start_slot=int(lp_info["start_slot"]),
            end_slot=int(lp_info["end_slot"]),
            core=int(lp_info["core"]),
            band=lp_info["band"],
            modulation=lp_info.get("mod_format", lp_info.get("modulation", "")),
            total_bandwidth_gbps=int(
                lp_info.get(
                    "lightpath_bandwidth", lp_info.get("total_bandwidth_gbps", 0)
                )
            ),
            remaining_bandwidth_gbps=int(
                lp_info.get(
                    "remaining_bandwidth", lp_info.get("remaining_bandwidth_gbps", 0)
                )
            ),
            path_weight_km=float(
                lp_info.get("path_weight", lp_info.get("path_weight_km", 0.0))
            ),
            request_allocations=request_allocations,
            snr_db=lp_info.get("snr_cost", lp_info.get("snr_db")),
            xt_cost=lp_info.get("xt_cost"),
            is_degraded=lp_info.get("is_degraded", False),
            backup_path=backup_path,
            backup_start_slot=lp_info.get("backup_start_slot"),
            backup_end_slot=lp_info.get("backup_end_slot"),
            backup_core=lp_info.get("backup_core"),
            backup_band=lp_info.get("backup_band"),
            is_protected=is_protected,
            active_path=lp_info.get("active_path", "primary"),
        )

    def to_legacy_dict(self) -> dict[str, Any]:
        """
        Convert to legacy dictionary format.

        This enables interoperability with legacy code that expects
        lightpath info dictionaries during the migration period.

        Returns:
            Dictionary compatible with legacy lightpath consumers:
                - "path": list[str]
                - "start_slot": int
                - "end_slot": int
                - "core": int
                - "band": str
                - "mod_format": str
                - "lightpath_bandwidth": float
                - "remaining_bandwidth": float
                - "requests_dict": dict[int, float]
                - "snr_cost": float | None
                - "xt_cost": float | None
                - "path_weight": float
                - "is_degraded": bool
                - "backup_path": list[str] | None
                - "is_protected": bool
                - "active_path": str

        Example:
            >>> lp = Lightpath(lightpath_id=1, path=["0", "5"], ...)
            >>> legacy = lp.to_legacy_dict()
            >>> legacy["mod_format"]
            'QPSK'
        """
        legacy: dict[str, Any] = {
            # Core fields
            "path": self.path,
            "start_slot": self.start_slot,
            "end_slot": self.end_slot,
            "core": self.core,
            "band": self.band,
            "mod_format": self.modulation,
            # Capacity (legacy uses float)
            "lightpath_bandwidth": float(self.total_bandwidth_gbps),
            "remaining_bandwidth": float(self.remaining_bandwidth_gbps),
            "requests_dict": {
                req_id: float(bw) for req_id, bw in self.request_allocations.items()
            },
            # Quality
            "snr_cost": self.snr_db,
            "xt_cost": self.xt_cost,
            "path_weight": self.path_weight_km,
            "is_degraded": self.is_degraded,
            # Protection
            "backup_path": self.backup_path,
            "is_protected": self.is_protected,
            "active_path": self.active_path,
        }

        # Include backup spectrum details if protected
        if self.is_protected:
            legacy["backup_start_slot"] = self.backup_start_slot
            legacy["backup_end_slot"] = self.backup_end_slot
            legacy["backup_core"] = self.backup_core
            legacy["backup_band"] = self.backup_band

        return legacy

    def to_legacy_key(self) -> tuple[str, str]:
        """
        Generate legacy dictionary key (endpoint pair).

        Returns:
            Sorted tuple of (source, destination) for dict indexing.
        """
        return self.endpoint_key
