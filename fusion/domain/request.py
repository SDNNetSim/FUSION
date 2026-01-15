"""
Request domain model with lifecycle tracking.

This module defines:
- RequestType: Enum for request event types (arrival/release)
- RequestStatus: Enum for request lifecycle states
- BlockReason: Enum for allocation failure reasons
- Request: Mutable dataclass for network service requests

All classes support legacy conversion via from_legacy_dict/to_legacy_dict methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


# =============================================================================
# Request Type Enum
# =============================================================================


class RequestType(Enum):
    """
    Type of request event in the simulation.

    ARRIVAL indicates a request arriving that needs resource allocation.
    RELEASE indicates a request departing where resources should be freed.
    """

    ARRIVAL = "arrival"
    RELEASE = "release"

    @classmethod
    def from_legacy(cls, value: str) -> RequestType:
        """
        Convert legacy string to enum.

        :param value: Legacy string "arrival" or "release".
        :type value: str
        :return: Corresponding RequestType enum value.
        :rtype: RequestType
        """
        mapping = {
            "arrival": cls.ARRIVAL,
            "release": cls.RELEASE,
        }
        return mapping[value.lower()]

    def to_legacy(self) -> str:
        """
        Convert enum to legacy string.

        :return: String value compatible with legacy code.
        :rtype: str
        """
        return self.value


# =============================================================================
# Request Lifecycle Enums
# =============================================================================


class RequestStatus(Enum):
    """
    Request lifecycle status.

    Initial state is PENDING. Processing states are ROUTING, SPECTRUM_SEARCH,
    SNR_CHECK. Terminal success states are ALLOCATED, GROOMED, PARTIALLY_GROOMED.
    Terminal failure state is BLOCKED. Release states are RELEASING and RELEASED.

    State Machine::

        PENDING --> ROUTING --> SPECTRUM_SEARCH --> SNR_CHECK
                        |              |               |
                        v              v               v
                    BLOCKED        BLOCKED         BLOCKED
                        |              |               |
                        +--------------+---------------+
                                       |
                                       v
                    +------------------+------------------+
                    |                  |                  |
                    v                  v                  v
                ALLOCATED          GROOMED      PARTIALLY_GROOMED
                    |                  |                  |
                    +------------------+------------------+
                                       |
                                       v
                                   RELEASING
                                       |
                                       v
                                   RELEASED
    """

    # Initial state
    PENDING = auto()
    """Request created, not yet processed."""

    # Processing states
    ROUTING = auto()
    """Finding candidate paths."""

    SPECTRUM_SEARCH = auto()
    """Searching for available spectrum."""

    SNR_CHECK = auto()
    """Validating signal-to-noise ratio."""

    # Terminal success states
    ALLOCATED = auto()
    """Successfully allocated new resources."""

    GROOMED = auto()
    """Successfully groomed onto existing lightpath."""

    PARTIALLY_GROOMED = auto()
    """Partially groomed, rest allocated on new lightpath."""

    # Terminal failure state
    BLOCKED = auto()
    """Could not allocate resources (see block_reason)."""

    # Release states
    RELEASING = auto()
    """Release in progress."""

    RELEASED = auto()
    """Resources freed (terminal)."""

    def is_terminal(self) -> bool:
        """Check if this is a terminal state."""
        return self in {
            RequestStatus.ALLOCATED,
            RequestStatus.GROOMED,
            RequestStatus.PARTIALLY_GROOMED,
            RequestStatus.BLOCKED,
            RequestStatus.RELEASED,
        }

    def is_success(self) -> bool:
        """Check if this represents successful allocation."""
        return self in {
            RequestStatus.ALLOCATED,
            RequestStatus.GROOMED,
            RequestStatus.PARTIALLY_GROOMED,
        }

    def is_processing(self) -> bool:
        """Check if request is currently being processed."""
        return self in {
            RequestStatus.ROUTING,
            RequestStatus.SPECTRUM_SEARCH,
            RequestStatus.SNR_CHECK,
            RequestStatus.RELEASING,
        }

    def can_transition_to(self, target: RequestStatus) -> bool:
        """
        Check if transition to target state is valid.

        :param target: The state to transition to.
        :type target: RequestStatus
        :return: True if the transition is valid, False otherwise.
        :rtype: bool
        """
        valid_transitions: dict[RequestStatus, set[RequestStatus]] = {
            # From initial state
            RequestStatus.PENDING: {
                RequestStatus.ROUTING,
                RequestStatus.BLOCKED,
                # Allow direct to success for simplified flows
                RequestStatus.ALLOCATED,
                RequestStatus.GROOMED,
                RequestStatus.PARTIALLY_GROOMED,
            },
            # Processing states
            RequestStatus.ROUTING: {
                RequestStatus.SPECTRUM_SEARCH,
                RequestStatus.BLOCKED,
            },
            RequestStatus.SPECTRUM_SEARCH: {
                RequestStatus.SNR_CHECK,
                RequestStatus.ALLOCATED,
                RequestStatus.GROOMED,
                RequestStatus.PARTIALLY_GROOMED,
                RequestStatus.BLOCKED,
            },
            RequestStatus.SNR_CHECK: {
                RequestStatus.ALLOCATED,
                RequestStatus.GROOMED,
                RequestStatus.PARTIALLY_GROOMED,
                RequestStatus.BLOCKED,
            },
            # Success states can transition to releasing
            RequestStatus.ALLOCATED: {RequestStatus.RELEASING},
            RequestStatus.GROOMED: {RequestStatus.RELEASING},
            RequestStatus.PARTIALLY_GROOMED: {RequestStatus.RELEASING},
            # Failure state is terminal
            RequestStatus.BLOCKED: set(),
            # Release states
            RequestStatus.RELEASING: {RequestStatus.RELEASED},
            RequestStatus.RELEASED: set(),
        }
        return target in valid_transitions[self]


class ProtectionStatus(Enum):
    """
    Protection state for 1+1 protected connections.

    States include UNPROTECTED (normal request), ESTABLISHING (setting up paths),
    ACTIVE_PRIMARY (using primary), ACTIVE_BACKUP (using backup after switchover),
    SWITCHOVER_IN_PROGRESS, PRIMARY_FAILED (on backup), BACKUP_FAILED (on primary),
    and BOTH_FAILED (connection lost).
    """

    UNPROTECTED = auto()
    """No protection (normal request)."""

    ESTABLISHING = auto()
    """Setting up protection paths."""

    ACTIVE_PRIMARY = auto()
    """Protected, using primary path."""

    ACTIVE_BACKUP = auto()
    """Protected, using backup path (after switchover)."""

    SWITCHOVER_IN_PROGRESS = auto()
    """Switching between paths."""

    PRIMARY_FAILED = auto()
    """Primary path failed, on backup."""

    BACKUP_FAILED = auto()
    """Backup path failed, on primary."""

    BOTH_FAILED = auto()
    """Both paths failed (connection lost)."""

    def is_active(self) -> bool:
        """Check if connection is active (either path)."""
        return self in {
            ProtectionStatus.ACTIVE_PRIMARY,
            ProtectionStatus.ACTIVE_BACKUP,
            ProtectionStatus.PRIMARY_FAILED,
            ProtectionStatus.BACKUP_FAILED,
        }

    def is_degraded(self) -> bool:
        """Check if protection is degraded (one path failed)."""
        return self in {
            ProtectionStatus.PRIMARY_FAILED,
            ProtectionStatus.BACKUP_FAILED,
        }

    def is_failed(self) -> bool:
        """Check if connection is failed."""
        return self == ProtectionStatus.BOTH_FAILED

    def is_protected(self) -> bool:
        """Check if request has protection enabled."""
        return self != ProtectionStatus.UNPROTECTED

    @classmethod
    def from_legacy(
        cls, is_protected: bool, active_path: str | None
    ) -> ProtectionStatus:
        """
        Convert legacy protection fields to enum.

        :param is_protected: Legacy is_protected flag.
        :type is_protected: bool
        :param active_path: Legacy active_path string ("primary" or "backup").
        :type active_path: str | None
        :return: Corresponding ProtectionStatus enum value.
        :rtype: ProtectionStatus
        """
        if not is_protected:
            return cls.UNPROTECTED
        if active_path == "backup":
            return cls.ACTIVE_BACKUP
        return cls.ACTIVE_PRIMARY

    def to_legacy_active_path(self) -> str:
        """
        Convert to legacy active_path string.

        :return: "primary" or "backup" based on current state.
        :rtype: str
        """
        if self in {ProtectionStatus.ACTIVE_BACKUP, ProtectionStatus.PRIMARY_FAILED}:
            return "backup"
        return "primary"


class BlockReason(Enum):
    """
    Reasons for request blocking.

    Path-related reasons are NO_PATH and DISTANCE. Spectrum-related is CONGESTION.
    Quality-related are SNR_THRESHOLD and XT_THRESHOLD. Feature-specific are
    GROOMING_FAIL, SLICING_FAIL, and PROTECTION_FAIL. Failure-related are
    LINK_FAILURE, NODE_FAILURE, and FAILURE. Resource limits are TRANSPONDER_LIMIT
    and MAX_SEGMENTS.
    """

    # Path-related blocking
    NO_PATH = "no_path"
    """No path exists between source and destination."""

    DISTANCE = "distance"
    """Path too long for any modulation format."""

    # Spectrum-related blocking
    CONGESTION = "congestion"
    """No spectrum available on any path."""

    # Quality-related blocking
    SNR_THRESHOLD = "snr_fail"
    """SNR below required threshold."""

    SNR_RECHECK_FAIL = "snr_recheck_fail"
    """SNR recheck failed for existing lightpaths after new allocation (congestion)."""

    XT_THRESHOLD = "xt_threshold"
    """Crosstalk exceeds allowed threshold."""

    # Feature-specific blocking
    GROOMING_FAIL = "grooming_fail"
    """Grooming attempt failed."""

    SLICING_FAIL = "slicing_fail"
    """Slicing attempt failed."""

    PROTECTION_FAIL = "protection_fail"
    """Cannot establish protection path."""

    # Failure-related blocking
    LINK_FAILURE = "link_failure"
    """Link failed during processing."""

    NODE_FAILURE = "node_failure"
    """Node failed during processing."""

    FAILURE = "failure"
    """Generic failure (legacy)."""

    # Resource limits
    TRANSPONDER_LIMIT = "transponder_limit"
    """Max transponders reached."""

    MAX_SEGMENTS = "max_segments"
    """Max slicing segments reached."""

    @classmethod
    def from_legacy_string(cls, reason: str | None) -> BlockReason | None:
        """
        Convert legacy block reason string to enum.

        :param reason: Legacy string like "distance", "congestion", etc.
        :type reason: str | None
        :return: Corresponding BlockReason or None if reason is None/empty.
        :rtype: BlockReason | None
        """
        if reason is None or reason == "":
            return None

        # Legacy string mappings
        legacy_map: dict[str, BlockReason] = {
            "distance": cls.DISTANCE,
            "congestion": cls.CONGESTION,
            "xt_threshold": cls.XT_THRESHOLD,
            "failure": cls.FAILURE,
            "no_path": cls.NO_PATH,
            "no_route": cls.NO_PATH,
            "snr_fail": cls.SNR_THRESHOLD,
            "snr_failure": cls.SNR_THRESHOLD,
            "snr_recheck_fail": cls.SNR_RECHECK_FAIL,
            "grooming_fail": cls.GROOMING_FAIL,
            "slicing_fail": cls.SLICING_FAIL,
            "protection_fail": cls.PROTECTION_FAIL,
            "link_failure": cls.LINK_FAILURE,
            "node_failure": cls.NODE_FAILURE,
            "transponder_limit": cls.TRANSPONDER_LIMIT,
            "max_segments": cls.MAX_SEGMENTS,
        }

        if reason in legacy_map:
            return legacy_map[reason]

        # Try direct enum value match
        try:
            return cls(reason)
        except ValueError:
            # Default to generic FAILURE for unknown reasons
            return cls.FAILURE

    def to_legacy_string(self) -> str:
        """
        Convert to legacy block reason string.

        :return: String value compatible with legacy code.
        :rtype: str
        """
        return self.value

    def is_path_related(self) -> bool:
        """Check if blocking is path-related."""
        return self in {BlockReason.NO_PATH, BlockReason.DISTANCE}

    def is_spectrum_related(self) -> bool:
        """Check if blocking is spectrum-related."""
        return self == BlockReason.CONGESTION

    def is_quality_related(self) -> bool:
        """Check if blocking is quality-related."""
        return self in {
            BlockReason.SNR_THRESHOLD,
            BlockReason.SNR_RECHECK_FAIL,
            BlockReason.XT_THRESHOLD,
        }

    def is_feature_related(self) -> bool:
        """Check if blocking is feature-related."""
        return self in {
            BlockReason.GROOMING_FAIL,
            BlockReason.SLICING_FAIL,
            BlockReason.PROTECTION_FAIL,
        }

    def is_failure_related(self) -> bool:
        """Check if blocking is failure-related."""
        return self in {
            BlockReason.LINK_FAILURE,
            BlockReason.NODE_FAILURE,
            BlockReason.FAILURE,
        }

    def is_resource_limit(self) -> bool:
        """Check if blocking is due to resource limits."""
        return self in {BlockReason.TRANSPONDER_LIMIT, BlockReason.MAX_SEGMENTS}


# =============================================================================
# Request Dataclass
# =============================================================================


@dataclass
class Request:
    """
    Network service request with lifecycle tracking.

    A Request represents a bandwidth demand between two network nodes.
    It tracks the full lifecycle from arrival through allocation to release.

    Attributes:
        request_id: Unique identifier (immutable after creation).
        source: Source node ID (immutable after creation).
        destination: Destination node ID (immutable after creation).
        bandwidth_gbps: Requested bandwidth in Gbps (immutable after creation).
        arrive_time: Simulation time of arrival (immutable after creation).
        depart_time: Simulation time of departure (immutable after creation).
        modulation_formats: Available modulation formats for this request.
        status: Current lifecycle state (PENDING -> ROUTED/BLOCKED -> RELEASED).
        lightpath_ids: IDs of lightpaths serving this request.
        block_reason: Reason for blocking (if status == BLOCKED).
        is_groomed: Fully served by existing lightpath capacity.
        is_partially_groomed: Partially served by existing lightpath.
        is_sliced: Split across multiple lightpaths.
        protection_status: Protection state for 1+1 protected connections.
        primary_path: Primary path for protected requests.
        backup_path: Backup path for protected requests.
        last_switchover_time: Time of last protection switchover.

    Example:
        >>> request = Request(
        ...     request_id=42,
        ...     source="0",
        ...     destination="5",
        ...     bandwidth_gbps=100,
        ...     arrive_time=12.345,
        ...     depart_time=17.890,
        ... )
        >>> request.status
        <RequestStatus.PENDING: 1>
        >>> request.is_arrival
        True
    """

    # =========================================================================
    # Identity Fields (effectively immutable after creation)
    # =========================================================================
    request_id: int
    source: str
    destination: str
    bandwidth_gbps: int
    arrive_time: float
    depart_time: float
    modulation_formats: dict[str, dict[str, Any]] = field(default_factory=dict)

    # =========================================================================
    # Lifecycle State (mutable)
    # =========================================================================
    status: RequestStatus = RequestStatus.PENDING
    lightpath_ids: list[int] = field(default_factory=list)
    block_reason: BlockReason | None = None

    # =========================================================================
    # Feature Flags (mutable)
    # =========================================================================
    is_groomed: bool = False
    is_partially_groomed: bool = False
    is_sliced: bool = False

    # =========================================================================
    # Protection State (for 1+1 protected requests)
    # =========================================================================
    protection_status: ProtectionStatus = ProtectionStatus.UNPROTECTED
    primary_path: list[str] | None = None
    backup_path: list[str] | None = None
    last_switchover_time: float | None = None

    # =========================================================================
    # Validation
    # =========================================================================
    def __post_init__(self) -> None:
        """Validate request after creation."""
        if self.source == self.destination:
            raise ValueError("source and destination cannot be the same")
        if self.bandwidth_gbps <= 0:
            raise ValueError("bandwidth_gbps must be > 0")
        if self.depart_time <= self.arrive_time:
            raise ValueError("depart_time must be > arrive_time")

    # =========================================================================
    # Computed Properties
    # =========================================================================
    @property
    def is_arrival(self) -> bool:
        """True if request is pending processing (arrival event)."""
        return self.status == RequestStatus.PENDING

    @property
    def is_successful(self) -> bool:
        """True if request was successfully allocated/groomed."""
        return self.status.is_success()

    @property
    def is_blocked(self) -> bool:
        """True if request was blocked."""
        return self.status == RequestStatus.BLOCKED

    @property
    def is_released(self) -> bool:
        """True if request has been released."""
        return self.status == RequestStatus.RELEASED

    @property
    def is_processing(self) -> bool:
        """True if request is currently being processed."""
        return self.status.is_processing()

    @property
    def is_terminal(self) -> bool:
        """True if request is in a terminal state."""
        return self.status.is_terminal()

    @property
    def endpoint_key(self) -> tuple[str, str]:
        """
        Canonical endpoint pair for lightpath matching.

        :return: Sorted tuple to ensure (A, B) == (B, A) for endpoint matching.
        :rtype: tuple[str, str]
        """
        return tuple(sorted([self.source, self.destination]))  # type: ignore[return-value]

    @property
    def holding_time(self) -> float:
        """Duration of the request in simulation time units."""
        return self.depart_time - self.arrive_time

    @property
    def num_lightpaths(self) -> int:
        """Number of lightpaths serving this request."""
        return len(self.lightpath_ids)

    @property
    def is_protected(self) -> bool:
        """True if request has protection enabled."""
        return self.protection_status.is_protected()

    @property
    def active_path(self) -> str:
        """Current active path ('primary' or 'backup') for protected requests."""
        return self.protection_status.to_legacy_active_path()

    @property
    def is_protection_degraded(self) -> bool:
        """True if protection is degraded (one path failed)."""
        return self.protection_status.is_degraded()

    @property
    def is_protection_failed(self) -> bool:
        """True if both protection paths have failed."""
        return self.protection_status.is_failed()

    # =========================================================================
    # State Transition Methods
    # =========================================================================
    def mark_allocated(self, lightpath_ids: list[int]) -> None:
        """
        Mark request as successfully allocated to new lightpath(s).

        :param lightpath_ids: IDs of lightpaths allocated for this request.
        :type lightpath_ids: list[int]
        :raises ValueError: If transition is invalid or no lightpaths provided.
        """
        if not self.status.can_transition_to(RequestStatus.ALLOCATED):
            raise ValueError(f"Cannot transition from {self.status} to ALLOCATED")
        if not lightpath_ids:
            raise ValueError("Must provide at least one lightpath_id")

        self.status = RequestStatus.ALLOCATED
        self.lightpath_ids = list(lightpath_ids)  # Copy to avoid aliasing
        self.block_reason = None

    def mark_groomed(self, lightpath_ids: list[int]) -> None:
        """
        Mark request as successfully groomed onto existing lightpath(s).

        :param lightpath_ids: IDs of lightpaths used for grooming.
        :type lightpath_ids: list[int]
        :raises ValueError: If transition is invalid or no lightpaths provided.
        """
        if not self.status.can_transition_to(RequestStatus.GROOMED):
            raise ValueError(f"Cannot transition from {self.status} to GROOMED")
        if not lightpath_ids:
            raise ValueError("Must provide at least one lightpath_id")

        self.status = RequestStatus.GROOMED
        self.lightpath_ids = list(lightpath_ids)
        self.block_reason = None
        self.is_groomed = True

    def mark_partially_groomed(self, lightpath_ids: list[int]) -> None:
        """
        Mark request as partially groomed (some traffic on existing LP, rest new).

        :param lightpath_ids: IDs of all lightpaths (existing and new).
        :type lightpath_ids: list[int]
        :raises ValueError: If transition is invalid or no lightpaths provided.
        """
        if not self.status.can_transition_to(RequestStatus.PARTIALLY_GROOMED):
            raise ValueError(
                f"Cannot transition from {self.status} to PARTIALLY_GROOMED"
            )
        if not lightpath_ids:
            raise ValueError("Must provide at least one lightpath_id")

        self.status = RequestStatus.PARTIALLY_GROOMED
        self.lightpath_ids = list(lightpath_ids)
        self.block_reason = None
        self.is_partially_groomed = True

    def mark_routed(self, lightpath_ids: list[int]) -> None:
        """
        Mark request as successfully routed (alias for mark_allocated).

        This method provides backwards compatibility with code that uses
        the simplified PENDING -> ROUTED -> RELEASED flow.

        :param lightpath_ids: IDs of lightpaths allocated for this request.
        :type lightpath_ids: list[int]
        :raises ValueError: If transition is invalid or no lightpaths provided.
        """
        self.mark_allocated(lightpath_ids)

    def mark_blocked(self, reason: BlockReason) -> None:
        """
        Mark request as blocked.

        :param reason: Reason for blocking.
        :type reason: BlockReason
        :raises ValueError: If transition is invalid.
        """
        if not self.status.can_transition_to(RequestStatus.BLOCKED):
            raise ValueError(f"Cannot transition from {self.status} to BLOCKED")

        self.status = RequestStatus.BLOCKED
        self.block_reason = reason

    def mark_releasing(self) -> None:
        """
        Mark request as releasing (resources being freed).

        :raises ValueError: If transition is invalid.
        """
        if not self.status.can_transition_to(RequestStatus.RELEASING):
            raise ValueError(f"Cannot transition from {self.status} to RELEASING")

        self.status = RequestStatus.RELEASING

    def mark_released(self) -> None:
        """
        Mark request as released (resources freed).

        If currently in a success state, transitions through RELEASING first.

        :raises ValueError: If transition is invalid.
        """
        # Allow direct transition from success states by going through RELEASING
        if self.status.is_success():
            self.mark_releasing()

        if not self.status.can_transition_to(RequestStatus.RELEASED):
            raise ValueError(f"Cannot transition from {self.status} to RELEASED")

        self.status = RequestStatus.RELEASED

    def set_status(self, status: RequestStatus) -> None:
        """
        Set status with transition validation.

        Use this for processing state transitions (ROUTING, SPECTRUM_SEARCH, etc.).

        :param status: The target status.
        :type status: RequestStatus
        :raises ValueError: If transition is invalid.
        """
        if not self.status.can_transition_to(status):
            raise ValueError(f"Cannot transition from {self.status} to {status}")
        self.status = status

    # =========================================================================
    # Legacy Adapters
    # TODO(v6.1): Remove legacy adapter methods once migration is complete.
    # =========================================================================
    @classmethod
    def from_legacy_dict(
        cls,
        time_key: tuple[int, float],
        request_dict: dict[str, Any],
        request_id: int | None = None,
    ) -> Request:
        """
        Create Request from legacy request dictionary.

        The legacy format stores requests in a dict indexed by (request_id, time).
        Each entry contains fields like "req_id", "source", "bandwidth", etc.

        :param time_key: Tuple of (request_id, time) used as dict key.
        :type time_key: tuple[int, float]
        :param request_dict: Legacy request dictionary with fields req_id (int),
            source (str), destination (str), arrive (float), depart (float),
            bandwidth (str like "50Gbps"), mod_formats (dict, optional),
            request_type (str "arrival"/"release", optional).
        :type request_dict: dict[str, Any]
        :param request_id: Override request_id (default: from time_key[0] or req_id).
        :type request_id: int | None
        :return: New Request instance in PENDING state.
        :rtype: Request

        Example:
            >>> legacy = {
            ...     "req_id": 42,
            ...     "source": "0",
            ...     "destination": "5",
            ...     "arrive": 12.345,
            ...     "depart": 17.890,
            ...     "bandwidth": "100Gbps",
            ... }
            >>> request = Request.from_legacy_dict((42, 12.345), legacy)
            >>> request.request_id
            42
            >>> request.bandwidth_gbps
            100
        """
        # Determine request_id (priority: parameter > dict > time_key)
        if request_id is None:
            request_id = request_dict.get("req_id", time_key[0])

        # Parse bandwidth string to integer
        bandwidth_raw = request_dict.get("bandwidth", "0Gbps")
        if isinstance(bandwidth_raw, str):
            # Handle formats like "100Gbps", "50 Gbps", "100"
            bandwidth_gbps = int(
                bandwidth_raw.lower().replace("gbps", "").replace(" ", "")
            )
        else:
            bandwidth_gbps = int(bandwidth_raw)

        # Handle modulation formats (may be None)
        mod_formats = request_dict.get("mod_formats")
        if mod_formats is None:
            mod_formats = {}

        return cls(
            request_id=request_id,
            source=str(request_dict["source"]),
            destination=str(request_dict["destination"]),
            bandwidth_gbps=bandwidth_gbps,
            arrive_time=float(request_dict.get("arrive", time_key[1])),
            depart_time=float(request_dict["depart"]),
            modulation_formats=mod_formats,
            # State fields start at defaults
            status=RequestStatus.PENDING,
            lightpath_ids=[],
            block_reason=None,
            is_groomed=False,
            is_partially_groomed=False,
            is_sliced=False,
            # protection_status defaults to UNPROTECTED
        )

    def to_legacy_dict(self) -> dict[str, Any]:
        """
        Convert to legacy request dictionary format.

        This enables interoperability with legacy code that expects
        request dictionaries during the migration period.

        :return: Dictionary compatible with legacy request consumers with keys
            req_id (int), source (str), destination (str), arrive (float),
            depart (float), bandwidth (str like "100Gbps"), mod_formats (dict),
            request_type (str "arrival"/"release").
        :rtype: dict[str, Any]

        Example:
            >>> request = Request(
            ...     request_id=42,
            ...     source="0",
            ...     destination="5",
            ...     bandwidth_gbps=100,
            ...     arrive_time=12.345,
            ...     depart_time=17.890,
            ... )
            >>> legacy = request.to_legacy_dict()
            >>> legacy["bandwidth"]
            '100Gbps'
        """
        # Determine request_type based on status
        if self.status == RequestStatus.RELEASED:
            request_type = "release"
        else:
            # PENDING, ROUTED, or BLOCKED are all arrival-related
            request_type = "arrival"

        return {
            "req_id": self.request_id,
            "source": self.source,
            "destination": self.destination,
            "arrive": self.arrive_time,
            "depart": self.depart_time,
            "bandwidth": f"{self.bandwidth_gbps}Gbps",
            "mod_formats": self.modulation_formats,
            "request_type": request_type,
        }

    def to_legacy_time_key(self) -> tuple[int, float]:
        """
        Generate legacy dictionary key for this request.

        :return: Tuple of (request_id, time) where time depends on status:
            PENDING/ROUTED/BLOCKED uses arrive_time, RELEASED uses depart_time.
        :rtype: tuple[int, float]
        """
        if self.status == RequestStatus.RELEASED:
            return (self.request_id, self.depart_time)
        return (self.request_id, self.arrive_time)
