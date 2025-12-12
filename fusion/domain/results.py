"""
Result objects for FUSION pipeline stages.

This module defines frozen dataclasses for all pipeline outputs:
- RouteResult: Routing pipeline output (candidate paths)
- SpectrumResult: Spectrum assignment output
- GroomingResult: Grooming pipeline output
- SlicingResult: Slicing pipeline output
- SNRResult: SNR validation output
- AllocationResult: Final orchestrator output (SINGLE SOURCE OF TRUTH)

All result objects are immutable (frozen dataclasses) to ensure
consistency throughout the processing pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fusion.domain.request import BlockReason


# =============================================================================
# RouteResult
# =============================================================================


@dataclass(frozen=True)
class RouteResult:
    """
    Result of routing pipeline - candidate paths with modulation options.

    Contains ordered list of candidate paths from best to worst,
    each with associated path weight and valid modulation formats.
    For 1+1 protection, includes disjoint backup paths.

    Attributes:
        paths: List of candidate paths (each path is tuple of node IDs)
        weights_km: Path lengths/weights in kilometers
        modulations: Valid modulation formats per path
        backup_paths: Disjoint backup paths for 1+1 protection (optional)
        backup_weights_km: Backup path lengths (optional)
        backup_modulations: Modulations for backup paths (optional)
        strategy_name: Name of routing algorithm used
        metadata: Algorithm-specific metadata

    Invariants:
        - len(paths) == len(weights_km) == len(modulations)
        - Each path has at least 2 nodes
        - Weights are non-negative
        - If backup_paths exists: len(backup_paths) == len(paths)

    Example:
        >>> result = RouteResult(
        ...     paths=(("0", "2", "5"),),
        ...     weights_km=(100.0,),
        ...     modulations=(("QPSK", "16-QAM"),),
        ...     strategy_name="k_shortest_path",
        ... )
        >>> result.is_empty
        False
        >>> result.best_path
        ('0', '2', '5')
    """

    # Primary path candidates
    paths: tuple[tuple[str, ...], ...] = ()
    weights_km: tuple[float, ...] = ()
    modulations: tuple[tuple[str, ...], ...] = ()

    # Backup paths for 1+1 protection (optional)
    backup_paths: tuple[tuple[str, ...], ...] | None = None
    backup_weights_km: tuple[float, ...] | None = None
    backup_modulations: tuple[tuple[str, ...], ...] | None = None

    # Metadata
    strategy_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    # External routing index (for pre-calculated SNR data lookup)
    connection_index: int | None = None

    def __post_init__(self) -> None:
        """Validate route result after creation."""
        # Length consistency
        if len(self.paths) != len(self.weights_km):
            raise ValueError("paths and weights_km must have same length")
        if len(self.paths) != len(self.modulations):
            raise ValueError("paths and modulations must have same length")

        # Path validity
        for path in self.paths:
            if len(path) < 2:
                raise ValueError("Each path must have at least 2 nodes")

        # Weight validity
        for weight in self.weights_km:
            if weight < 0:
                raise ValueError("Weights must be non-negative")

        # Backup consistency
        if self.backup_paths is not None:
            if len(self.backup_paths) != len(self.paths):
                raise ValueError("backup_paths must match paths length")

    @property
    def is_empty(self) -> bool:
        """True if no paths found (routing failed)."""
        return len(self.paths) == 0

    @property
    def num_paths(self) -> int:
        """Number of candidate paths."""
        return len(self.paths)

    @property
    def has_protection(self) -> bool:
        """True if backup paths are available."""
        return self.backup_paths is not None and len(self.backup_paths) > 0

    @property
    def best_path(self) -> tuple[str, ...] | None:
        """Best (first) candidate path or None if empty."""
        return self.paths[0] if self.paths else None

    @property
    def best_weight(self) -> float | None:
        """Weight of best path or None if empty."""
        return self.weights_km[0] if self.weights_km else None

    def get_path(self, index: int) -> tuple[str, ...]:
        """Get path at index."""
        return self.paths[index]

    def get_modulations_for_path(self, index: int) -> tuple[str, ...]:
        """Get valid modulations for path at index."""
        return self.modulations[index]

    @classmethod
    def empty(cls, strategy_name: str = "") -> RouteResult:
        """Create empty result (no paths found)."""
        return cls(
            paths=(),
            weights_km=(),
            modulations=(),
            strategy_name=strategy_name,
        )

    @classmethod
    def from_routing_props(cls, routing_props: Any) -> RouteResult:
        """
        Create from legacy RoutingProps.

        Args:
            routing_props: Legacy RoutingProps object

        Returns:
            New RouteResult instance
        """
        # Handle empty case
        if (
            not hasattr(routing_props, "paths_matrix")
            or not routing_props.paths_matrix
        ):
            return cls.empty()

        # Convert lists to tuples for immutability
        paths = tuple(tuple(str(n) for n in p) for p in routing_props.paths_matrix)
        weights = tuple(routing_props.weights_list)
        modulations = tuple(
            tuple(mods) for mods in routing_props.modulation_formats_matrix
        )

        # Handle backup paths if present
        backup_paths = None
        backup_weights = None
        backup_mods = None

        if (
            hasattr(routing_props, "backup_paths_matrix")
            and routing_props.backup_paths_matrix
        ):
            backup_paths = tuple(
                tuple(str(n) for n in p) if p else ()
                for p in routing_props.backup_paths_matrix
            )
            if hasattr(routing_props, "backup_weights_list"):
                backup_weights = tuple(routing_props.backup_weights_list)
            if hasattr(routing_props, "backup_modulation_formats_matrix"):
                backup_mods = tuple(
                    tuple(mods) for mods in routing_props.backup_modulation_formats_matrix
                )

        return cls(
            paths=paths,
            weights_km=weights,
            modulations=modulations,
            backup_paths=backup_paths,
            backup_weights_km=backup_weights,
            backup_modulations=backup_mods,
        )


# =============================================================================
# SpectrumResult
# =============================================================================


@dataclass(frozen=True)
class SpectrumResult:
    """
    Result of spectrum assignment pipeline.

    Indicates whether contiguous spectrum was found and provides
    the allocation details (slot range, core, band, modulation).

    Attributes:
        is_free: Whether spectrum assignment was successful
        start_slot: First slot index (valid only if is_free=True)
        end_slot: Last slot index exclusive (valid only if is_free=True)
        core: Core number for MCF (valid only if is_free=True)
        band: Frequency band (valid only if is_free=True)
        modulation: Selected modulation format
        slots_needed: Number of slots required (including guard band)

        Backup spectrum (for 1+1 protection):
            backup_start_slot: Backup spectrum start
            backup_end_slot: Backup spectrum end
            backup_core: Backup core number
            backup_band: Backup frequency band

    Invariants:
        - If is_free=False, slot fields are not meaningful
        - If is_free=True, end_slot > start_slot
        - If backup fields are set, all backup fields must be set

    Example:
        >>> result = SpectrumResult(
        ...     is_free=True,
        ...     start_slot=100,
        ...     end_slot=108,
        ...     core=0,
        ...     band="c",
        ...     modulation="QPSK",
        ...     slots_needed=8,
        ... )
        >>> result.num_slots
        8
    """

    is_free: bool

    # Primary allocation (meaningful only if is_free=True)
    start_slot: int = 0
    end_slot: int = 0
    core: int = 0
    band: str = "c"
    modulation: str = ""
    slots_needed: int = 0

    # Achieved bandwidth in dynamic_lps mode (may be less than requested)
    # When None, assumed to equal the requested bandwidth
    achieved_bandwidth_gbps: int | None = None

    # Backup allocation (for 1+1 protection)
    backup_start_slot: int | None = None
    backup_end_slot: int | None = None
    backup_core: int | None = None
    backup_band: str | None = None

    def __post_init__(self) -> None:
        """Validate spectrum result after creation."""
        if self.is_free:
            if self.end_slot <= self.start_slot:
                raise ValueError("end_slot must be > start_slot when is_free=True")
            if self.slots_needed <= 0:
                raise ValueError("slots_needed must be > 0 when is_free=True")

    @property
    def num_slots(self) -> int:
        """Number of slots allocated (0 if not free)."""
        if not self.is_free:
            return 0
        return self.end_slot - self.start_slot

    @property
    def has_backup(self) -> bool:
        """True if backup spectrum is allocated."""
        return self.backup_start_slot is not None

    @classmethod
    def not_found(cls, slots_needed: int = 0) -> SpectrumResult:
        """Create result for failed spectrum assignment."""
        return cls(is_free=False, slots_needed=slots_needed)

    @classmethod
    def from_spectrum_props(cls, spectrum_props: Any) -> SpectrumResult:
        """
        Create from legacy SpectrumProps or dict.

        Args:
            spectrum_props: Legacy SpectrumProps object or dict

        Returns:
            New SpectrumResult instance
        """
        if isinstance(spectrum_props, dict):
            return cls(
                is_free=spectrum_props.get("is_free", False),
                start_slot=spectrum_props.get("start_slot", 0),
                end_slot=spectrum_props.get("end_slot", 0),
                core=spectrum_props.get("core_number", spectrum_props.get("core", 0)),
                band=spectrum_props.get("band", "c"),
                modulation=spectrum_props.get("modulation", ""),
                slots_needed=spectrum_props.get("slots_needed", 0),
            )

        # Handle SpectrumProps object
        return cls(
            is_free=getattr(spectrum_props, "is_free", False),
            start_slot=getattr(spectrum_props, "start_slot", 0) or 0,
            end_slot=getattr(spectrum_props, "end_slot", 0) or 0,
            core=getattr(spectrum_props, "core_number", 0) or 0,
            band=getattr(spectrum_props, "current_band", "c") or "c",
            modulation=getattr(spectrum_props, "modulation", "") or "",
            slots_needed=getattr(spectrum_props, "slots_needed", 0) or 0,
        )

    def to_allocation_dict(self) -> dict[str, Any]:
        """
        Convert to allocation dictionary for legacy compatibility.

        Returns:
            Dictionary with allocation details
        """
        return {
            "is_free": self.is_free,
            "start_slot": self.start_slot,
            "end_slot": self.end_slot,
            "core_number": self.core,
            "band": self.band,
            "modulation": self.modulation,
            "slots_needed": self.slots_needed,
        }


# =============================================================================
# GroomingResult
# =============================================================================


@dataclass(frozen=True)
class GroomingResult:
    """
    Result of grooming pipeline - using existing lightpath capacity.

    Indicates whether a request can be (fully or partially) served
    by existing lightpaths without creating new ones.

    Attributes:
        fully_groomed: Entire request fits in existing lightpaths
        partially_groomed: Some bandwidth groomed, rest needs new LP
        bandwidth_groomed_gbps: Amount successfully groomed
        remaining_bandwidth_gbps: Amount still needing new lightpath
        lightpaths_used: IDs of lightpaths used for grooming
        forced_path: Required path for new lightpath (if partial)

    Invariants:
        - fully_groomed and partially_groomed are mutually exclusive
        - If fully_groomed: remaining_bandwidth_gbps == 0
        - If partially_groomed or fully_groomed: len(lightpaths_used) > 0
        - bandwidth_groomed + remaining_bandwidth == original request

    Example:
        >>> result = GroomingResult.full(100, [1, 2])
        >>> result.fully_groomed
        True
        >>> result.needs_new_lightpath
        False
    """

    fully_groomed: bool = False
    partially_groomed: bool = False
    bandwidth_groomed_gbps: int = 0
    remaining_bandwidth_gbps: int = 0
    lightpaths_used: tuple[int, ...] = ()
    forced_path: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        """Validate grooming result after creation."""
        if self.fully_groomed and self.partially_groomed:
            raise ValueError("fully_groomed and partially_groomed are mutually exclusive")

        if self.fully_groomed and self.remaining_bandwidth_gbps != 0:
            raise ValueError("fully_groomed requires remaining_bandwidth_gbps == 0")

        if (self.fully_groomed or self.partially_groomed) and len(self.lightpaths_used) == 0:
            raise ValueError("groomed requests must have lightpaths_used")

    @property
    def was_groomed(self) -> bool:
        """True if any grooming occurred."""
        return self.fully_groomed or self.partially_groomed

    @property
    def needs_new_lightpath(self) -> bool:
        """True if a new lightpath is still needed."""
        return not self.fully_groomed and self.remaining_bandwidth_gbps > 0

    @classmethod
    def no_grooming(cls, bandwidth_gbps: int) -> GroomingResult:
        """Create result when no grooming is possible."""
        return cls(
            fully_groomed=False,
            partially_groomed=False,
            bandwidth_groomed_gbps=0,
            remaining_bandwidth_gbps=bandwidth_gbps,
            lightpaths_used=(),
        )

    @classmethod
    def full(cls, bandwidth_gbps: int, lightpath_ids: list[int]) -> GroomingResult:
        """Create result for fully groomed request."""
        return cls(
            fully_groomed=True,
            partially_groomed=False,
            bandwidth_groomed_gbps=bandwidth_gbps,
            remaining_bandwidth_gbps=0,
            lightpaths_used=tuple(lightpath_ids),
        )

    @classmethod
    def partial(
        cls,
        bandwidth_groomed: int,
        remaining: int,
        lightpath_ids: list[int],
        forced_path: list[str] | None = None,
    ) -> GroomingResult:
        """Create result for partially groomed request."""
        return cls(
            fully_groomed=False,
            partially_groomed=True,
            bandwidth_groomed_gbps=bandwidth_groomed,
            remaining_bandwidth_gbps=remaining,
            lightpaths_used=tuple(lightpath_ids),
            forced_path=tuple(forced_path) if forced_path else None,
        )


# =============================================================================
# SlicingResult
# =============================================================================


@dataclass(frozen=True)
class SlicingResult:
    """
    Result of slicing pipeline - splitting request across lightpaths.

    When a request is too large for a single lightpath (modulation
    limitations), it may be split into multiple smaller slices.

    Attributes:
        success: Whether slicing succeeded
        num_slices: Number of slices created
        slice_bandwidth_gbps: Bandwidth per slice
        lightpaths_created: IDs of created lightpaths
        total_bandwidth_gbps: Total bandwidth across all slices

    Invariants:
        - If success=True: num_slices > 0 and len(lightpaths_created) == num_slices
        - total_bandwidth_gbps == num_slices * slice_bandwidth_gbps

    Example:
        >>> result = SlicingResult.sliced(
        ...     num_slices=4,
        ...     slice_bandwidth=25,
        ...     lightpath_ids=[1, 2, 3, 4],
        ... )
        >>> result.is_sliced
        True
        >>> result.total_bandwidth_gbps
        100
    """

    success: bool = False
    num_slices: int = 0
    slice_bandwidth_gbps: int = 0
    lightpaths_created: tuple[int, ...] = ()
    total_bandwidth_gbps: int = 0

    def __post_init__(self) -> None:
        """Validate slicing result after creation."""
        if self.success:
            if self.num_slices == 0:
                raise ValueError("success=True requires num_slices > 0")
            if len(self.lightpaths_created) != self.num_slices:
                raise ValueError("lightpaths_created length must match num_slices")

    @property
    def is_sliced(self) -> bool:
        """True if request was sliced (multiple lightpaths)."""
        return self.success and self.num_slices > 1

    @classmethod
    def failed(cls) -> SlicingResult:
        """Create result for failed slicing."""
        return cls(success=False)

    @classmethod
    def single_lightpath(cls, bandwidth_gbps: int, lightpath_id: int) -> SlicingResult:
        """Create result for single lightpath (no slicing needed)."""
        return cls(
            success=True,
            num_slices=1,
            slice_bandwidth_gbps=bandwidth_gbps,
            lightpaths_created=(lightpath_id,),
            total_bandwidth_gbps=bandwidth_gbps,
        )

    @classmethod
    def sliced(
        cls,
        num_slices: int,
        slice_bandwidth: int,
        lightpath_ids: list[int],
    ) -> SlicingResult:
        """Create result for sliced request."""
        return cls(
            success=True,
            num_slices=num_slices,
            slice_bandwidth_gbps=slice_bandwidth,
            lightpaths_created=tuple(lightpath_ids),
            total_bandwidth_gbps=num_slices * slice_bandwidth,
        )


# =============================================================================
# SNRResult
# =============================================================================


@dataclass(frozen=True)
class SNRResult:
    """
    Result of SNR validation pipeline.

    Indicates whether the signal-to-noise ratio meets the threshold
    required for the selected modulation format.

    Attributes:
        passed: Whether SNR meets threshold (primary success indicator)
        snr_db: Calculated SNR value in dB
        required_snr_db: Threshold required for modulation format
        margin_db: SNR margin (snr_db - required_snr_db)
        failure_reason: Explanation if validation failed
        link_snr_values: Per-link SNR breakdown for debugging

    Invariants:
        - margin_db == snr_db - required_snr_db
        - If passed=False and snr_db < required_snr_db, margin_db is negative

    Example:
        >>> result = SNRResult.success(snr_db=18.5, required_snr_db=15.0)
        >>> result.passed
        True
        >>> result.margin_db
        3.5
    """

    passed: bool
    snr_db: float = 0.0
    required_snr_db: float = 0.0
    margin_db: float = 0.0
    failure_reason: str | None = None
    link_snr_values: dict[tuple[str, str], float] = field(default_factory=dict)

    @property
    def is_degraded(self) -> bool:
        """True if SNR passed but with low margin (< 1 dB)."""
        return self.passed and 0 <= self.margin_db < 1.0

    @property
    def has_link_breakdown(self) -> bool:
        """True if per-link SNR values are available."""
        return len(self.link_snr_values) > 0

    @classmethod
    def success(
        cls,
        snr_db: float,
        required_snr_db: float,
        link_snr_values: dict[tuple[str, str], float] | None = None,
    ) -> SNRResult:
        """Create successful SNR result."""
        margin = snr_db - required_snr_db
        return cls(
            passed=True,
            snr_db=snr_db,
            required_snr_db=required_snr_db,
            margin_db=margin,
            failure_reason=None,
            link_snr_values=link_snr_values or {},
        )

    @classmethod
    def failure(
        cls,
        snr_db: float,
        required_snr_db: float,
        reason: str = "SNR below threshold",
        link_snr_values: dict[tuple[str, str], float] | None = None,
    ) -> SNRResult:
        """Create failed SNR result."""
        margin = snr_db - required_snr_db
        return cls(
            passed=False,
            snr_db=snr_db,
            required_snr_db=required_snr_db,
            margin_db=margin,
            failure_reason=reason,
            link_snr_values=link_snr_values or {},
        )

    @classmethod
    def skipped(cls) -> SNRResult:
        """Create result for when SNR validation is disabled."""
        return cls(
            passed=True,
            snr_db=0.0,
            required_snr_db=0.0,
            margin_db=0.0,
            failure_reason=None,
        )

    def get_weakest_link(self) -> tuple[tuple[str, str], float] | None:
        """
        Get the link with lowest SNR.

        Returns:
            Tuple of (link, snr_db) or None if no link breakdown
        """
        if not self.link_snr_values:
            return None
        return min(self.link_snr_values.items(), key=lambda x: x[1])


# =============================================================================
# SNRRecheckResult
# =============================================================================


@dataclass(frozen=True)
class SNRRecheckResult:
    """
    Result of SNR recheck after new lightpath allocation.

    When a new lightpath is allocated, it may cause interference with
    existing lightpaths. This result captures the outcome of rechecking
    SNR for affected lightpaths.

    Attributes:
        all_pass: True if all affected lightpaths still meet SNR threshold
        degraded_lightpath_ids: IDs of lightpaths now below threshold
        violations: Mapping of lightpath_id to SNR shortfall in dB
        checked_count: Number of lightpaths that were checked

    Invariants:
        - If all_pass=True: degraded_lightpath_ids is empty
        - If all_pass=False: len(degraded_lightpath_ids) > 0

    Example:
        >>> result = SNRRecheckResult.all_pass()
        >>> result.all_pass
        True
        >>> result.num_degraded
        0
    """

    all_pass: bool
    degraded_lightpath_ids: tuple[int, ...] = ()
    violations: dict[int, float] = field(default_factory=dict)
    checked_count: int = 0

    def __post_init__(self) -> None:
        """Validate SNR recheck result after creation."""
        if self.all_pass and len(self.degraded_lightpath_ids) > 0:
            raise ValueError("all_pass=True requires no degraded lightpaths")
        if not self.all_pass and len(self.degraded_lightpath_ids) == 0:
            raise ValueError("all_pass=False requires at least one degraded lightpath")

    @property
    def num_degraded(self) -> int:
        """Number of lightpaths that are now degraded."""
        return len(self.degraded_lightpath_ids)

    @property
    def has_violations(self) -> bool:
        """True if any lightpaths are degraded."""
        return len(self.degraded_lightpath_ids) > 0

    def get_worst_violation(self) -> tuple[int, float] | None:
        """
        Get lightpath with largest SNR shortfall.

        Returns:
            Tuple of (lightpath_id, shortfall_db) or None if no violations
        """
        if not self.violations:
            return None
        return min(self.violations.items(), key=lambda x: x[1])

    @classmethod
    def success(cls, checked_count: int = 0) -> "SNRRecheckResult":
        """Create result when all affected lightpaths still pass."""
        return cls(
            all_pass=True,
            degraded_lightpath_ids=(),
            violations={},
            checked_count=checked_count,
        )

    @classmethod
    def degraded(
        cls,
        degraded_ids: list[int],
        violations: dict[int, float],
        checked_count: int = 0,
    ) -> "SNRRecheckResult":
        """Create result when some lightpaths are now degraded."""
        return cls(
            all_pass=False,
            degraded_lightpath_ids=tuple(degraded_ids),
            violations=violations,
            checked_count=checked_count if checked_count > 0 else len(degraded_ids),
        )


# =============================================================================
# AllocationResult - FINAL AUTHORITY
# =============================================================================


@dataclass(frozen=True)
class ProtectionResult:
    """
    Result of protection path establishment or switchover.

    Used for 1+1 protection scenarios to track the establishment
    of primary and backup paths, and any switchover events.

    Attributes:
        primary_established: Whether primary path was established
        backup_established: Whether backup path was established
        primary_spectrum: Spectrum allocation for primary path
        backup_spectrum: Spectrum allocation for backup path
        switchover_triggered: Whether a switchover event occurred
        switchover_success: Whether switchover completed successfully
        switchover_time_ms: Time taken for switchover in milliseconds
        failure_type: Type of failure that triggered switchover
        recovery_start: Simulation time when recovery started
        recovery_end: Simulation time when recovery completed
        recovery_type: Type of recovery ("protection" or "restoration")

    Invariants:
        - If switchover_triggered=True and switchover_success=True,
          switchover_time_ms should be set
        - If backup_established=True, backup_spectrum should be set

    Example:
        >>> result = ProtectionResult.established(
        ...     primary_spectrum=spectrum1,
        ...     backup_spectrum=spectrum2,
        ... )
        >>> result.is_fully_protected
        True
    """

    primary_established: bool = False
    backup_established: bool = False
    primary_spectrum: SpectrumResult | None = None
    backup_spectrum: SpectrumResult | None = None
    switchover_triggered: bool = False
    switchover_success: bool = False
    switchover_time_ms: float | None = None
    failure_type: str | None = None
    recovery_start: float | None = None
    recovery_end: float | None = None
    recovery_type: str | None = None

    @property
    def is_fully_protected(self) -> bool:
        """True if both primary and backup paths are established."""
        return self.primary_established and self.backup_established

    @property
    def recovery_duration_ms(self) -> float | None:
        """Duration of recovery in milliseconds, if completed."""
        if self.recovery_start is not None and self.recovery_end is not None:
            return (self.recovery_end - self.recovery_start) * 1000
        return None

    @classmethod
    def established(
        cls,
        primary_spectrum: SpectrumResult,
        backup_spectrum: SpectrumResult | None = None,
    ) -> "ProtectionResult":
        """Create result for successfully established protection."""
        return cls(
            primary_established=True,
            backup_established=backup_spectrum is not None,
            primary_spectrum=primary_spectrum,
            backup_spectrum=backup_spectrum,
        )

    @classmethod
    def primary_only(cls, primary_spectrum: SpectrumResult) -> "ProtectionResult":
        """Create result when only primary path established (backup failed)."""
        return cls(
            primary_established=True,
            backup_established=False,
            primary_spectrum=primary_spectrum,
        )

    @classmethod
    def failed(cls) -> "ProtectionResult":
        """Create result for failed protection establishment."""
        return cls(
            primary_established=False,
            backup_established=False,
        )

    @classmethod
    def switchover(
        cls,
        success: bool,
        switchover_time_ms: float,
        failure_type: str,
        recovery_type: str = "protection",
    ) -> "ProtectionResult":
        """Create result for a switchover event."""
        return cls(
            primary_established=True,
            backup_established=True,
            switchover_triggered=True,
            switchover_success=success,
            switchover_time_ms=switchover_time_ms if success else None,
            failure_type=failure_type,
            recovery_type=recovery_type,
        )


# =============================================================================
# AllocationResult - FINAL AUTHORITY
# =============================================================================


@dataclass(frozen=True)
class AllocationResult:
    """
    Final result of request allocation - SINGLE SOURCE OF TRUTH.

    This is the authoritative result returned by the orchestrator.
    The `success` field is the final word on whether a request
    was served (fully or partially).

    Attributes:
        success: FINAL AUTHORITY - True if request was served
        lightpaths_created: IDs of newly created lightpaths
        lightpaths_groomed: IDs of existing lightpaths used
        total_bandwidth_allocated_gbps: Total bandwidth allocated

        Feature flags:
            is_groomed: Request used existing lightpath capacity
            is_partially_groomed: Mix of existing and new lightpath
            is_sliced: Request split across multiple lightpaths
            is_protected: Request has 1+1 protection

        Failure info:
            block_reason: BlockReason enum if success=False

        Per-segment tracking (for sliced/multi-lightpath allocations):
            bandwidth_allocations: Bandwidth allocated per segment
            modulations: Modulation format per segment
            cores: Core number per segment
            bands: Frequency band per segment
            start_slots: Start slot per segment
            end_slots: End slot per segment
            xt_costs: Crosstalk cost per segment (from crosstalk_list)
            xt_values: Crosstalk values per segment (from xt_list)
            snr_values: SNR value per segment
            lightpath_bandwidths: Total bandwidth per lightpath

        Debug info (nested results):
            route_result: Routing pipeline output
            spectrum_result: Spectrum pipeline output
            grooming_result: Grooming pipeline output
            slicing_result: Slicing pipeline output
            snr_result: SNR validation output
            protection_result: Protection establishment output

    Invariants:
        - If success=True: len(lightpaths_created) + len(lightpaths_groomed) > 0
        - If success=True: total_bandwidth_allocated_gbps > 0
        - If success=False: block_reason is set

    Example:
        >>> result = AllocationResult.success_new_lightpath(
        ...     lightpath_id=42,
        ...     bandwidth_gbps=100,
        ... )
        >>> result.success
        True
        >>> result.num_lightpaths
        1
    """

    success: bool

    # Lightpath tracking
    lightpaths_created: tuple[int, ...] = ()
    lightpaths_groomed: tuple[int, ...] = ()
    total_bandwidth_allocated_gbps: int = 0

    # Feature flags
    is_groomed: bool = False
    is_partially_groomed: bool = False
    is_sliced: bool = False
    is_protected: bool = False

    # Failure info
    block_reason: "BlockReason | None" = None

    # Per-segment tracking (for sliced/multi-lightpath allocations)
    bandwidth_allocations: tuple[int, ...] = ()
    modulations: tuple[str, ...] = ()
    cores: tuple[int, ...] = ()
    bands: tuple[str, ...] = ()
    start_slots: tuple[int, ...] = ()
    end_slots: tuple[int, ...] = ()
    xt_costs: tuple[float, ...] = ()  # From legacy crosstalk_list
    xt_values: tuple[float, ...] = ()  # From legacy xt_list
    snr_values: tuple[float, ...] = ()
    lightpath_bandwidths: tuple[int, ...] = ()

    # Nested results (for debugging/tracing)
    route_result: RouteResult | None = None
    spectrum_result: SpectrumResult | None = None
    grooming_result: GroomingResult | None = None
    slicing_result: SlicingResult | None = None
    snr_result: SNRResult | None = None
    protection_result: ProtectionResult | None = None

    def __post_init__(self) -> None:
        """Validate allocation result after creation."""
        if self.success:
            # Must have allocated something
            total_lps = len(self.lightpaths_created) + len(self.lightpaths_groomed)
            if total_lps == 0:
                raise ValueError("success=True requires at least one lightpath")
            if self.total_bandwidth_allocated_gbps <= 0:
                raise ValueError("success=True requires total_bandwidth > 0")
        else:
            # Must have a reason for failure
            if self.block_reason is None:
                raise ValueError("success=False requires block_reason")

    @property
    def all_lightpath_ids(self) -> tuple[int, ...]:
        """All lightpath IDs (created + groomed)."""
        return self.lightpaths_created + self.lightpaths_groomed

    @property
    def num_lightpaths(self) -> int:
        """Total number of lightpaths used."""
        return len(self.lightpaths_created) + len(self.lightpaths_groomed)

    @property
    def used_grooming(self) -> bool:
        """True if any grooming was used."""
        return self.is_groomed or self.is_partially_groomed

    @classmethod
    def blocked(
        cls,
        reason: "BlockReason",
        route_result: RouteResult | None = None,
        spectrum_result: SpectrumResult | None = None,
        snr_result: SNRResult | None = None,
    ) -> "AllocationResult":
        """Create result for blocked request."""
        return cls(
            success=False,
            block_reason=reason,
            route_result=route_result,
            spectrum_result=spectrum_result,
            snr_result=snr_result,
        )

    @classmethod
    def success_new_lightpath(
        cls,
        lightpath_id: int,
        bandwidth_gbps: int,
        route_result: RouteResult | None = None,
        spectrum_result: SpectrumResult | None = None,
        snr_result: SNRResult | None = None,
        is_protected: bool = False,
    ) -> "AllocationResult":
        """Create result for successful allocation with new lightpath."""
        return cls(
            success=True,
            lightpaths_created=(lightpath_id,),
            total_bandwidth_allocated_gbps=bandwidth_gbps,
            is_protected=is_protected,
            route_result=route_result,
            spectrum_result=spectrum_result,
            snr_result=snr_result,
        )

    @classmethod
    def success_groomed(
        cls,
        lightpath_ids: list[int],
        bandwidth_gbps: int,
        grooming_result: GroomingResult | None = None,
    ) -> "AllocationResult":
        """Create result for fully groomed request."""
        return cls(
            success=True,
            lightpaths_groomed=tuple(lightpath_ids),
            total_bandwidth_allocated_gbps=bandwidth_gbps,
            is_groomed=True,
            grooming_result=grooming_result,
        )

    @classmethod
    def success_partial_groom(
        cls,
        groomed_ids: list[int],
        new_lightpath_id: int,
        total_bandwidth: int,
        grooming_result: GroomingResult | None = None,
        route_result: RouteResult | None = None,
        spectrum_result: SpectrumResult | None = None,
        snr_result: SNRResult | None = None,
    ) -> "AllocationResult":
        """Create result for partially groomed request."""
        return cls(
            success=True,
            lightpaths_created=(new_lightpath_id,),
            lightpaths_groomed=tuple(groomed_ids),
            total_bandwidth_allocated_gbps=total_bandwidth,
            is_partially_groomed=True,
            grooming_result=grooming_result,
            route_result=route_result,
            spectrum_result=spectrum_result,
            snr_result=snr_result,
        )

    @classmethod
    def success_sliced(
        cls,
        lightpath_ids: list[int],
        bandwidth_gbps: int,
        slicing_result: SlicingResult | None = None,
        route_result: RouteResult | None = None,
    ) -> "AllocationResult":
        """Create result for sliced request."""
        return cls(
            success=True,
            lightpaths_created=tuple(lightpath_ids),
            total_bandwidth_allocated_gbps=bandwidth_gbps,
            is_sliced=True,
            slicing_result=slicing_result,
            route_result=route_result,
        )
