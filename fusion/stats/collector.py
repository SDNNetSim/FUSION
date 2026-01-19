"""
StatsCollector - Centralized statistics collection for FUSION simulations.

This module provides the StatsCollector class for aggregating
simulation metrics from request processing outcomes. It consumes
domain objects (Request, AllocationResult) and produces statistics
compatible with run_comparison.py.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fusion.domain.config import SimulationConfig
    from fusion.domain.request import Request
    from fusion.domain.results import AllocationResult


@dataclass
class StatsCollector:
    """
    Centralized statistics collection for simulation runs.

    StatsCollector aggregates metrics from request processing,
    consuming domain objects (Request, AllocationResult) and
    producing statistics compatible with run_comparison.py.

    Owned by SimulationEngine, receives recording calls after
    each request is processed.

    :ivar config: Simulation configuration (for context)
    :vartype config: SimulationConfig

    **Request Counters:**

    :ivar total_requests: Total arrivals processed
    :vartype total_requests: int
    :ivar successful_requests: Requests that were routed
    :vartype successful_requests: int
    :ivar blocked_requests: Requests that were blocked
    :vartype blocked_requests: int

    **Block Reasons:**

    :ivar block_reasons: Maps reason string to count
    :vartype block_reasons: dict[str, int]

    **Feature Tracking:**

    :ivar groomed_requests: Fully groomed requests
    :vartype groomed_requests: int
    :ivar partially_groomed_requests: Partially groomed requests
    :vartype partially_groomed_requests: int
    :ivar sliced_requests: Sliced requests
    :vartype sliced_requests: int
    :ivar protected_requests: Protected requests
    :vartype protected_requests: int

    **Modulation Tracking:**

    :ivar modulations_used: Maps format name to count
    :vartype modulations_used: dict[str, int]

    **SNR Tracking:**

    :ivar snr_values: List of SNR measurements
    :vartype snr_values: list[float]

    **Bandwidth Tracking:**

    :ivar total_bandwidth_requested_gbps: Sum of request bandwidths
    :vartype total_bandwidth_requested_gbps: int
    :ivar total_bandwidth_allocated_gbps: Sum of allocated bandwidths
    :vartype total_bandwidth_allocated_gbps: int

    **Protection Tracking:**

    :ivar switchover_count: Number of protection switchovers
    :vartype switchover_count: int
    :ivar protection_failures: Number of protection failures
    :vartype protection_failures: int
    :ivar failure_induced_blocks: Blocks due to network failures
    :vartype failure_induced_blocks: int
    :ivar switchover_times: List of switchover event times
    :vartype switchover_times: list[float]

    **Path Tracking:**

    :ivar hop_counts: List of path hop counts
    :vartype hop_counts: list[int]
    :ivar path_lengths_km: List of path lengths in km
    :vartype path_lengths_km: list[float]

    Example::

        >>> from fusion.domain.config import SimulationConfig
        >>> config = SimulationConfig(...)
        >>> collector = StatsCollector(config)
        >>> collector.record_arrival(request, result)
        >>> print(f"BP: {collector.blocking_probability:.2%}")
    """

    config: SimulationConfig

    # =========================================================================
    # Request Counters
    # =========================================================================
    total_requests: int = 0
    successful_requests: int = 0
    blocked_requests: int = 0

    # =========================================================================
    # Block Reasons (uses string keys for legacy compatibility)
    # =========================================================================
    block_reasons: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # =========================================================================
    # Feature Tracking
    # =========================================================================
    groomed_requests: int = 0
    partially_groomed_requests: int = 0
    sliced_requests: int = 0
    protected_requests: int = 0

    # =========================================================================
    # Modulation Tracking
    # =========================================================================
    modulations_used: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # =========================================================================
    # SNR Tracking
    # =========================================================================
    snr_values: list[float] = field(default_factory=list)

    # =========================================================================
    # Bandwidth Tracking
    # =========================================================================
    total_bandwidth_requested_gbps: int = 0
    total_bandwidth_allocated_gbps: int = 0

    # =========================================================================
    # Protection Tracking (from StatsProps)
    # =========================================================================
    switchover_count: int = 0
    protection_failures: int = 0
    failure_induced_blocks: int = 0
    switchover_times: list[float] = field(default_factory=list)

    # =========================================================================
    # Path Tracking
    # =========================================================================
    hop_counts: list[int] = field(default_factory=list)
    path_lengths_km: list[float] = field(default_factory=list)

    # =========================================================================
    # Core Usage Tracking
    # =========================================================================
    core_usage: dict[int, int] = field(default_factory=lambda: defaultdict(int))

    # =========================================================================
    # Band Usage Tracking
    # =========================================================================
    band_usage: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # =========================================================================
    # Crosstalk (XT) Tracking (from StatsProps.crosstalk_list)
    # =========================================================================
    xt_values: list[float] = field(default_factory=list)

    # =========================================================================
    # Computed Properties
    # =========================================================================
    @property
    def blocking_probability(self) -> float:
        """
        Calculate blocking probability.

        :return: Ratio of blocked to total requests (0.0 to 1.0).
            Returns 0.0 if no requests have been processed.
        :rtype: float
        """
        if self.total_requests == 0:
            return 0.0
        return self.blocked_requests / self.total_requests

    @property
    def success_rate(self) -> float:
        """
        Calculate success rate (complement of blocking probability).

        :return: Ratio of successful to total requests (0.0 to 1.0).
        :rtype: float
        """
        return 1.0 - self.blocking_probability

    @property
    def average_snr(self) -> float:
        """
        Calculate average SNR across all measurements.

        :return: Mean SNR in dB, or 0.0 if no measurements.
        :rtype: float
        """
        if not self.snr_values:
            return 0.0
        return sum(self.snr_values) / len(self.snr_values)

    @property
    def min_snr(self) -> float:
        """
        Get minimum SNR value.

        :return: Minimum SNR in dB, or 0.0 if no measurements.
        :rtype: float
        """
        if not self.snr_values:
            return 0.0
        return min(self.snr_values)

    @property
    def max_snr(self) -> float:
        """
        Get maximum SNR value.

        :return: Maximum SNR in dB, or 0.0 if no measurements.
        :rtype: float
        """
        if not self.snr_values:
            return 0.0
        return max(self.snr_values)

    @property
    def grooming_ratio(self) -> float:
        """
        Ratio of groomed requests to successful requests.

        Includes both fully and partially groomed requests.

        :return: Ratio (0.0 to 1.0), or 0.0 if no successful requests.
        :rtype: float
        """
        if self.successful_requests == 0:
            return 0.0
        total_groomed = self.groomed_requests + self.partially_groomed_requests
        return total_groomed / self.successful_requests

    @property
    def slicing_ratio(self) -> float:
        """
        Ratio of sliced requests to successful requests.

        :return: Ratio (0.0 to 1.0), or 0.0 if no successful requests.
        :rtype: float
        """
        if self.successful_requests == 0:
            return 0.0
        return self.sliced_requests / self.successful_requests

    @property
    def protection_ratio(self) -> float:
        """
        Ratio of protected requests to successful requests.

        :return: Ratio (0.0 to 1.0), or 0.0 if no successful requests.
        :rtype: float
        """
        if self.successful_requests == 0:
            return 0.0
        return self.protected_requests / self.successful_requests

    @property
    def bandwidth_utilization(self) -> float:
        """
        Ratio of allocated bandwidth to requested bandwidth.

        :return: Ratio (0.0 to 1.0), or 0.0 if no bandwidth requested.
        :rtype: float
        """
        if self.total_bandwidth_requested_gbps == 0:
            return 0.0
        return self.total_bandwidth_allocated_gbps / self.total_bandwidth_requested_gbps

    @property
    def average_hop_count(self) -> float:
        """
        Calculate average path hop count.

        :return: Mean hop count, or 0.0 if no paths recorded.
        :rtype: float
        """
        if not self.hop_counts:
            return 0.0
        return sum(self.hop_counts) / len(self.hop_counts)

    @property
    def average_path_length_km(self) -> float:
        """
        Calculate average path length in kilometers.

        :return: Mean path length, or 0.0 if no paths recorded.
        :rtype: float
        """
        if not self.path_lengths_km:
            return 0.0
        return sum(self.path_lengths_km) / len(self.path_lengths_km)

    # =========================================================================
    # Recording Methods
    # =========================================================================
    def record_arrival(
        self,
        request: Request,
        result: AllocationResult,
    ) -> None:
        """
        Record the outcome of a request arrival.

        This is the primary recording method, called by SimulationEngine
        after each arrival is processed. Updates all relevant counters
        based on the allocation result.

        :param request: The request that was processed
        :type request: Request
        :param result: The allocation result (success or blocked)
        :type result: AllocationResult

        Example::

            >>> stats.record_arrival(request, result)
            >>> stats.total_requests
            1
        """
        # Update request counter
        self.total_requests += 1
        self.total_bandwidth_requested_gbps += request.bandwidth_gbps

        if result.success:
            self._record_success(request, result)
        else:
            self._record_block(result)

    def _record_success(
        self,
        request: Request,
        result: AllocationResult,
    ) -> None:
        """
        Record a successful allocation.

        :param request: The routed request
        :type request: Request
        :param result: The successful allocation result
        :type result: AllocationResult
        """
        self.successful_requests += 1
        self.total_bandwidth_allocated_gbps += result.total_bandwidth_allocated_gbps

        # Feature tracking
        if result.is_groomed:
            self.groomed_requests += 1
        if result.is_partially_groomed:
            self.partially_groomed_requests += 1
        if result.is_sliced:
            self.sliced_requests += 1
        if result.is_protected:
            self.protected_requests += 1

        # Modulation tracking from spectrum result
        if result.spectrum_result and result.spectrum_result.modulation:
            modulation = result.spectrum_result.modulation
            self.modulations_used[modulation] += 1

        # Also track modulations from per-segment allocations
        for mod in result.modulations:
            if mod and mod not in self.modulations_used:
                self.modulations_used[mod] = 0
            if mod:
                self.modulations_used[mod] += 1

        # Core usage tracking
        if result.spectrum_result:
            self.core_usage[result.spectrum_result.core] += 1
        for core in result.cores:
            self.core_usage[core] += 1

        # Band usage tracking
        if result.spectrum_result:
            self.band_usage[result.spectrum_result.band] += 1
        for band in result.bands:
            if band:
                self.band_usage[band] += 1

        # SNR tracking from result
        if result.snr_result and result.snr_result.passed:
            self.snr_values.append(result.snr_result.snr_db)

        # Also track per-segment SNR values
        for snr in result.snr_values:
            if snr > 0:  # Filter out zero/invalid values
                self.snr_values.append(snr)

        # Crosstalk tracking from per-segment allocations
        for xt in result.xt_costs:
            if xt > 0:  # Filter out zero/invalid values
                self.xt_values.append(xt)
        for xt in result.xt_values:
            if xt > 0:
                self.xt_values.append(xt)

        # Path tracking from route result
        if result.route_result and result.route_result.best_path:
            path = result.route_result.best_path
            self.hop_counts.append(len(path) - 1)  # Hops = nodes - 1
            if result.route_result.best_weight is not None:
                self.path_lengths_km.append(result.route_result.best_weight)

        # Protection tracking
        if result.protection_result:
            if result.protection_result.switchover_triggered:
                self.switchover_count += 1
                if result.protection_result.switchover_time_ms:
                    self.switchover_times.append(result.protection_result.switchover_time_ms)
            if not result.protection_result.is_fully_protected and result.is_protected:
                self.protection_failures += 1

    def _record_block(self, result: AllocationResult) -> None:
        """
        Record a blocked request.

        :param result: The failed allocation result
        :type result: AllocationResult
        """
        self.blocked_requests += 1

        if result.block_reason is not None:
            reason_str = result.block_reason.value
            self.block_reasons[reason_str] += 1

            # Track failure-induced blocks
            if result.block_reason.is_failure_related():
                self.failure_induced_blocks += 1

    def record_snr(self, snr_db: float) -> None:
        """
        Record an SNR measurement.

        Called during SNR validation to track signal quality.
        Can be called multiple times per request (e.g., per link).

        :param snr_db: SNR value in dB
        :type snr_db: float

        Example::

            >>> stats.record_snr(18.5)
            >>> stats.record_snr(15.5)
            >>> stats.average_snr
            17.0
        """
        self.snr_values.append(snr_db)

    def record_xt(self, xt_value: float) -> None:
        """
        Record a crosstalk measurement.

        Called during XT validation to track inter-core crosstalk.
        Can be called multiple times per request.

        :param xt_value: Crosstalk value (typically negative dB or linear)
        :type xt_value: float

        Example::

            >>> stats.record_xt(-30.5)
            >>> stats.record_xt(-28.0)
        """
        self.xt_values.append(xt_value)

    def record_release(self, request: Request) -> None:
        """
        Record a request release.

        Called when a request departs and resources are freed.
        In Phase 1, this is a placeholder for future utilization tracking.

        :param request: The request being released
        :type request: Request

        .. note::

            Currently a no-op in Phase 1. Future phases may track:

            - Active request count over time
            - Resource utilization during request lifetime
            - Departure rate metrics
        """
        # Phase 1: No-op (placeholder for future utilization tracking)
        pass

    def record_switchover(
        self,
        switchover_time_ms: float,
        success: bool = True,
    ) -> None:
        """
        Record a protection switchover event.

        :param switchover_time_ms: Time taken for switchover in milliseconds
        :type switchover_time_ms: float
        :param success: Whether the switchover was successful
        :type success: bool
        """
        self.switchover_count += 1
        if success:
            self.switchover_times.append(switchover_time_ms)
        else:
            self.protection_failures += 1

    # =========================================================================
    # Export Methods
    # =========================================================================
    def to_comparison_format(self) -> dict[str, Any]:
        """
        Export statistics in run_comparison.py format.

        This method produces a dictionary compatible with the existing
        comparison scripts and result analysis tools.

        :return: Dictionary with keys including blocking_probability (float 0.0-1.0),
            success_rate (float 0.0-1.0), total_requests (int), successful_requests (int),
            blocked_requests (int), block_reasons (dict[str, int]), grooming_ratio (float),
            slicing_ratio (float), protection_ratio (float), groomed_requests (int),
            sliced_requests (int), protected_requests (int), modulations_used (dict[str, int]),
            average_snr (float), snr_values (list[float]), bandwidth_utilization (float),
            total_bandwidth_requested_gbps (int), total_bandwidth_allocated_gbps (int)
        :rtype: dict[str, Any]

        Example::

            >>> results = stats.to_comparison_format()
            >>> results["blocking_probability"]
            0.15
        """
        return {
            # Primary metrics
            "blocking_probability": self.blocking_probability,
            "success_rate": self.success_rate,
            # Request counts
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "blocked_requests": self.blocked_requests,
            # Block reasons (convert defaultdict to regular dict)
            "block_reasons": dict(self.block_reasons),
            # Feature ratios
            "grooming_ratio": self.grooming_ratio,
            "slicing_ratio": self.slicing_ratio,
            "protection_ratio": self.protection_ratio,
            # Feature counts
            "groomed_requests": self.groomed_requests,
            "partially_groomed_requests": self.partially_groomed_requests,
            "sliced_requests": self.sliced_requests,
            "protected_requests": self.protected_requests,
            # Modulation usage (convert defaultdict to regular dict)
            "modulations_used": dict(self.modulations_used),
            # SNR metrics
            "average_snr": self.average_snr,
            "min_snr": self.min_snr,
            "max_snr": self.max_snr,
            "snr_values": self.snr_values.copy(),  # Copy to prevent mutation
            # Bandwidth metrics
            "bandwidth_utilization": self.bandwidth_utilization,
            "total_bandwidth_requested_gbps": self.total_bandwidth_requested_gbps,
            "total_bandwidth_allocated_gbps": self.total_bandwidth_allocated_gbps,
            # Path metrics
            "average_hop_count": self.average_hop_count,
            "average_path_length_km": self.average_path_length_km,
            "hop_counts": self.hop_counts.copy(),
            "path_lengths_km": self.path_lengths_km.copy(),
            # Core and band usage
            "core_usage": dict(self.core_usage),
            "band_usage": dict(self.band_usage),
            # Crosstalk metrics
            "xt_values": self.xt_values.copy(),
            "average_xt": sum(self.xt_values) / len(self.xt_values) if self.xt_values else 0.0,
            # Protection metrics
            "switchover_count": self.switchover_count,
            "protection_failures": self.protection_failures,
            "failure_induced_blocks": self.failure_induced_blocks,
            "switchover_times": self.switchover_times.copy(),
            # Configuration context
            "config": {
                "network_name": self.config.network_name,
                "erlang": self.config.erlang,
                "num_requests": self.config.num_requests,
                "k_paths": self.config.k_paths,
                "route_method": self.config.route_method,
                "allocation_method": self.config.allocation_method,
                "grooming_enabled": self.config.grooming_enabled,
                "slicing_enabled": self.config.slicing_enabled,
                "snr_enabled": self.config.snr_enabled,
                "cores_per_link": self.config.cores_per_link,
                "guard_slots": self.config.guard_slots,
            },
        }

    def to_legacy_stats_dict(self) -> dict[str, Any]:
        """
        Export in legacy StatsProps-compatible format.

        This method produces output matching the legacy StatsProps
        structure for gradual migration of existing code.

        :return: Dictionary matching legacy stats structure
        :rtype: dict[str, Any]
        """
        return {
            # Match legacy key names
            "simulation_blocking_list": [self.blocking_probability],
            "simulation_bitrate_blocking_list": [1.0 - self.bandwidth_utilization if self.total_bandwidth_requested_gbps > 0 else 0.0],
            "block_reasons_dict": dict(self.block_reasons),
            "modulations_used_dict": dict(self.modulations_used),
            "snr_list": self.snr_values.copy(),
            "crosstalk_list": self.xt_values.copy(),
            "hops_list": [float(h) for h in self.hop_counts],
            "lengths_list": self.path_lengths_km.copy(),
            "cores_dict": dict(self.core_usage),
            "bands_list": list(self.band_usage.keys()),
            # Protection stats
            "protection_switchovers": self.switchover_count,
            "protection_failures": self.protection_failures,
            "failure_induced_blocks": self.failure_induced_blocks,
            "switchover_times": self.switchover_times.copy(),
        }

    def reset(self) -> None:
        """
        Reset all counters and lists to initial state.

        Called between simulation iterations or when starting
        a new simulation run.

        Example::

            >>> stats.record_arrival(request, result)
            >>> stats.total_requests
            1
            >>> stats.reset()
            >>> stats.total_requests
            0
        """
        # Request counters
        self.total_requests = 0
        self.successful_requests = 0
        self.blocked_requests = 0

        # Block reasons (clear and reset to defaultdict behavior)
        self.block_reasons.clear()

        # Feature tracking
        self.groomed_requests = 0
        self.partially_groomed_requests = 0
        self.sliced_requests = 0
        self.protected_requests = 0

        # Modulation tracking
        self.modulations_used.clear()

        # SNR tracking
        self.snr_values.clear()

        # Bandwidth tracking
        self.total_bandwidth_requested_gbps = 0
        self.total_bandwidth_allocated_gbps = 0

        # Protection tracking
        self.switchover_count = 0
        self.protection_failures = 0
        self.failure_induced_blocks = 0
        self.switchover_times.clear()

        # Path tracking
        self.hop_counts.clear()
        self.path_lengths_km.clear()

        # Core and band usage
        self.core_usage.clear()
        self.band_usage.clear()

        # Crosstalk tracking
        self.xt_values.clear()

    def merge(self, other: StatsCollector) -> None:
        """
        Merge statistics from another collector into this one.

        Useful for combining statistics from parallel simulation runs.

        :param other: Another StatsCollector to merge from
        :type other: StatsCollector
        """
        # Request counters
        self.total_requests += other.total_requests
        self.successful_requests += other.successful_requests
        self.blocked_requests += other.blocked_requests

        # Block reasons
        for reason, count in other.block_reasons.items():
            self.block_reasons[reason] += count

        # Feature tracking
        self.groomed_requests += other.groomed_requests
        self.partially_groomed_requests += other.partially_groomed_requests
        self.sliced_requests += other.sliced_requests
        self.protected_requests += other.protected_requests

        # Modulation tracking
        for mod, count in other.modulations_used.items():
            self.modulations_used[mod] += count

        # SNR tracking
        self.snr_values.extend(other.snr_values)

        # Bandwidth tracking
        self.total_bandwidth_requested_gbps += other.total_bandwidth_requested_gbps
        self.total_bandwidth_allocated_gbps += other.total_bandwidth_allocated_gbps

        # Protection tracking
        self.switchover_count += other.switchover_count
        self.protection_failures += other.protection_failures
        self.failure_induced_blocks += other.failure_induced_blocks
        self.switchover_times.extend(other.switchover_times)

        # Path tracking
        self.hop_counts.extend(other.hop_counts)
        self.path_lengths_km.extend(other.path_lengths_km)

        # Core and band usage
        for core, count in other.core_usage.items():
            self.core_usage[core] += count
        for band, count in other.band_usage.items():
            self.band_usage[band] += count

        # Crosstalk tracking
        self.xt_values.extend(other.xt_values)

    # =========================================================================
    # String Representations
    # =========================================================================
    def __str__(self) -> str:
        """Human-readable summary of statistics."""
        return (
            f"StatsCollector("
            f"total={self.total_requests}, "
            f"success={self.successful_requests}, "
            f"blocked={self.blocked_requests}, "
            f"BP={self.blocking_probability:.2%})"
        )

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return (
            f"StatsCollector("
            f"config={self.config.network_name!r}, "
            f"total_requests={self.total_requests}, "
            f"successful_requests={self.successful_requests}, "
            f"blocked_requests={self.blocked_requests}, "
            f"block_reasons={dict(self.block_reasons)}, "
            f"modulations_used={dict(self.modulations_used)})"
        )
