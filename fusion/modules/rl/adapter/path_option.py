"""PathOption dataclass for RL action selection.

This module defines the PathOption dataclass that represents a candidate path
with its metadata. It is used by the RLSimulationAdapter for:
- Action masking (via is_feasible field)
- Observation building (path features)
- Action application (path selection)

RLSimulationAdapter Scaffolding
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class PathOption:
    """
    Represents a candidate path for RL action selection.

    This dataclass encapsulates all information about a candidate path
    that an RL agent needs to make a routing decision. The is_feasible
    field is computed from REAL spectrum pipeline checks, not mocked.

    :ivar path_index: Index in the k-paths list (0 to k-1)
    :vartype path_index: int
    :ivar path: Sequence of node IDs representing the path (immutable tuple)
    :vartype path: tuple[str, ...]
    :ivar weight_km: Total path length in kilometers
    :vartype weight_km: float
    :ivar num_hops: Number of links in the path
    :vartype num_hops: int
    :ivar modulation: Selected modulation format (e.g., "QPSK", "16-QAM")
    :vartype modulation: str | None
    :ivar slots_needed: Number of contiguous spectrum slots required
    :vartype slots_needed: int
    :ivar is_feasible: True if path can be allocated (from real spectrum check)
    :vartype is_feasible: bool
    :ivar congestion: Congestion metric in [0, 1], higher = more congested
    :vartype congestion: float
    :ivar available_slots: Ratio of available slots on most constrained link
    :vartype available_slots: float
    :ivar spectrum_start: Start slot if feasible, None otherwise
    :vartype spectrum_start: int | None
    :ivar spectrum_end: End slot if feasible, None otherwise
    :vartype spectrum_end: int | None
    :ivar core_index: Core index if multi-core, None for single-core
    :vartype core_index: int | None
    :ivar band: Spectrum band (e.g., "C", "L") if multi-band, None otherwise
    :vartype band: str | None
    :ivar frag_indicator: Path fragmentation indicator [0, 1], 0 = no fragmentation
    :vartype frag_indicator: float
    :ivar failure_mask: Whether path passes through failed links (disaster scenarios)
    :vartype failure_mask: bool
    :ivar dist_to_disaster: Normalized distance from path to disaster centroid [0, 1]
    :vartype dist_to_disaster: float
    :ivar min_residual_slots: Minimum residual slots along path, normalized [0, 1]
    :vartype min_residual_slots: float
    :ivar backup_path: Backup path for 1+1 protection, None if unprotected
    :vartype backup_path: tuple[str, ...] | None
    :ivar backup_feasible: Whether backup path has available spectrum
    :vartype backup_feasible: bool | None
    :ivar backup_weight_km: Backup path length in kilometers
    :vartype backup_weight_km: float | None
    :ivar backup_modulation: Backup path modulation format
    :vartype backup_modulation: str | None
    :ivar is_protected: Whether this is a 1+1 protected path pair
    :vartype is_protected: bool
    """

    # Core fields (required)
    path_index: int
    path: tuple[str, ...]
    weight_km: float
    num_hops: int
    modulation: str | None
    slots_needed: int
    is_feasible: bool
    congestion: float
    available_slots: float

    # Spectrum fields (optional)
    spectrum_start: int | None = None
    spectrum_end: int | None = None
    core_index: int | None = None
    band: str | None = None

    # Disaster-aware fields (optional, for offline RL compatibility)
    frag_indicator: float = 0.0
    failure_mask: bool = False
    dist_to_disaster: float = 1.0
    min_residual_slots: float = 1.0

    # Protection fields (optional, for 1+1 protection)
    backup_path: tuple[str, ...] | None = None
    backup_feasible: bool | None = None
    backup_weight_km: float | None = None
    backup_modulation: str | None = None
    is_protected: bool = False

    def __post_init__(self) -> None:
        """
        Validate invariants after initialization.

        :raises ValueError: If any field violates its constraints
        """
        if self.path_index < 0:
            raise ValueError("path_index must be non-negative")
        if self.weight_km < 0:
            raise ValueError("weight_km must be non-negative")
        if self.congestion < 0 or self.congestion > 1:
            raise ValueError("congestion must be in [0, 1]")
        if self.available_slots < 0 or self.available_slots > 1:
            raise ValueError("available_slots must be in [0, 1]")
        # Validate protection fields consistency
        if self.is_protected:
            if self.backup_path is None:
                raise ValueError("is_protected=True requires backup_path")
            if self.backup_feasible is None:
                raise ValueError("is_protected=True requires backup_feasible")

    @property
    def both_paths_feasible(self) -> bool:
        """
        Check if both primary and backup paths are feasible.

        For unprotected paths, returns the primary path's feasibility.
        For protected paths, returns True only if both paths have spectrum.

        :return: True if allocation can proceed (considering protection status)
        :rtype: bool
        """
        if not self.is_protected:
            return self.is_feasible
        return self.is_feasible and (self.backup_feasible is True)

    @property
    def total_weight_km(self) -> float:
        """
        Total path length (primary + backup if protected).

        :return: Combined length for protected paths, primary length otherwise
        :rtype: float
        """
        if self.is_protected and self.backup_weight_km is not None:
            return self.weight_km + self.backup_weight_km
        return self.weight_km

    @property
    def backup_hop_count(self) -> int | None:
        """
        Number of hops in backup path, or None if unprotected.

        :return: Number of hops in the backup path
        :rtype: int | None
        """
        if self.backup_path is None:
            return None
        return len(self.backup_path) - 1

    @classmethod
    def from_pipeline_results(
        cls,
        path_index: int,
        route_result: Any,
        spectrum_result: Any | None,
        congestion: float,
        available_slots: float,
    ) -> PathOption:
        """
        Create PathOption from pipeline results.

        Factory method to construct a PathOption from routing and spectrum
        pipeline results. This provides a clean interface between the
        pipeline layer and the RL adapter.

        :param path_index: Index in k-paths list
        :type path_index: int
        :param route_result: Result from routing pipeline with path, weight_km,
            modulation attributes
        :type route_result: Any
        :param spectrum_result: Result from spectrum pipeline with is_free,
            start_slot, end_slot, core, band attributes (or None)
        :type spectrum_result: Any | None
        :param congestion: Pre-computed congestion metric [0, 1]
        :type congestion: float
        :param available_slots: Pre-computed available slots ratio [0, 1]
        :type available_slots: float
        :return: PathOption instance
        :rtype: PathOption
        """
        # Extract path - handle both list and tuple
        path = route_result.path
        if isinstance(path, list):
            path = tuple(path)

        # Determine modulation
        modulation: str | None = None
        if hasattr(route_result, "modulation"):
            modulation = route_result.modulation
        elif hasattr(route_result, "modulations") and route_result.modulations:
            # May be a tuple of modulations, take first
            mods = route_result.modulations
            modulation = mods[0] if mods else None

        # Determine slots needed
        slots_needed = 0
        if spectrum_result is not None and hasattr(spectrum_result, "slots_needed"):
            slots_needed = spectrum_result.slots_needed
        elif hasattr(route_result, "slots_needed"):
            slots_needed = route_result.slots_needed

        # Determine feasibility and spectrum details
        is_feasible = False
        spectrum_start = None
        spectrum_end = None
        core_index = None
        band = None

        if spectrum_result is not None:
            is_feasible = getattr(spectrum_result, "is_free", False)
            if is_feasible:
                spectrum_start = getattr(spectrum_result, "start_slot", None)
                spectrum_end = getattr(spectrum_result, "end_slot", None)
                core_index = getattr(spectrum_result, "core", None)
                if core_index is None:
                    core_index = getattr(spectrum_result, "core_index", None)
                band = getattr(spectrum_result, "band", None)

        return cls(
            path_index=path_index,
            path=path,
            weight_km=getattr(route_result, "weight_km", 0.0),
            num_hops=len(path) - 1 if len(path) > 1 else 0,
            modulation=modulation,
            slots_needed=slots_needed,
            is_feasible=is_feasible,
            congestion=congestion,
            available_slots=available_slots,
            spectrum_start=spectrum_start,
            spectrum_end=spectrum_end,
            core_index=core_index,
            band=band,
        )

    @classmethod
    def from_protected_route(
        cls,
        path_index: int,
        primary_path: list[str],
        backup_path: list[str],
        primary_weight: float,
        backup_weight: float,
        primary_feasible: bool,
        backup_feasible: bool,
        primary_modulation: str | None,
        backup_modulation: str | None,
        slots_needed: int,
        congestion: float,
        available_slots: float = 1.0,
    ) -> PathOption:
        """
        Factory method to create PathOption for 1+1 protected path pair.

        This method provides a convenient way to construct a protected
        PathOption from separate primary and backup path information.

        :param path_index: Index in the options list
        :type path_index: int
        :param primary_path: Primary path node sequence
        :type primary_path: list[str]
        :param backup_path: Backup (disjoint) path node sequence
        :type backup_path: list[str]
        :param primary_weight: Primary path length in km
        :type primary_weight: float
        :param backup_weight: Backup path length in km
        :type backup_weight: float
        :param primary_feasible: Primary path spectrum availability
        :type primary_feasible: bool
        :param backup_feasible: Backup path spectrum availability
        :type backup_feasible: bool
        :param primary_modulation: Primary path modulation format
        :type primary_modulation: str | None
        :param backup_modulation: Backup path modulation format
        :type backup_modulation: str | None
        :param slots_needed: Spectrum slots required (same for both paths)
        :type slots_needed: int
        :param congestion: Combined congestion metric
        :type congestion: float
        :param available_slots: Available slots ratio (default 1.0)
        :type available_slots: float
        :return: PathOption configured for 1+1 protection
        :rtype: PathOption
        """
        return cls(
            path_index=path_index,
            path=tuple(primary_path),
            weight_km=primary_weight,
            num_hops=len(primary_path) - 1 if len(primary_path) > 1 else 0,
            modulation=primary_modulation,
            slots_needed=slots_needed,
            is_feasible=primary_feasible,
            congestion=congestion,
            available_slots=available_slots,
            backup_path=tuple(backup_path),
            backup_feasible=backup_feasible,
            backup_weight_km=backup_weight,
            backup_modulation=backup_modulation,
            is_protected=True,
        )

    @classmethod
    def from_unprotected_route(
        cls,
        path_index: int,
        path: list[str],
        weight_km: float,
        is_feasible: bool,
        modulation: str | None,
        slots_needed: int,
        congestion: float,
        available_slots: float = 1.0,
    ) -> PathOption:
        """
        Factory method to create PathOption for unprotected route.

        :param path_index: Index in the options list
        :type path_index: int
        :param path: Path node sequence
        :type path: list[str]
        :param weight_km: Path length in km
        :type weight_km: float
        :param is_feasible: Spectrum availability
        :type is_feasible: bool
        :param modulation: Modulation format
        :type modulation: str | None
        :param slots_needed: Required spectrum slots
        :type slots_needed: int
        :param congestion: Path congestion metric
        :type congestion: float
        :param available_slots: Available slots ratio (default 1.0)
        :type available_slots: float
        :return: PathOption configured for unprotected route
        :rtype: PathOption
        """
        return cls(
            path_index=path_index,
            path=tuple(path),
            weight_km=weight_km,
            num_hops=len(path) - 1 if len(path) > 1 else 0,
            modulation=modulation,
            slots_needed=slots_needed,
            is_feasible=is_feasible,
            congestion=congestion,
            available_slots=available_slots,
        )


# Type aliases for clarity
PathOptionList: TypeAlias = list[PathOption]
ActionMask: TypeAlias = np.ndarray  # Shape: (k_paths,), dtype: bool


def compute_action_mask(options: list[PathOption], k_paths: int) -> np.ndarray:
    """
    Generate action mask from path options.

    Creates a boolean mask where True indicates a valid (feasible) action.
    This is used by action-masked RL algorithms like MaskablePPO.

    :param options: List of PathOption instances
    :type options: list[PathOption]
    :param k_paths: Total number of possible actions (action space size)
    :type k_paths: int
    :return: Boolean array of shape (k_paths,) where True = action is valid
    :rtype: np.ndarray
    """
    mask = np.zeros(k_paths, dtype=bool)
    for opt in options:
        if opt.path_index < k_paths:
            mask[opt.path_index] = opt.is_feasible
    return mask
