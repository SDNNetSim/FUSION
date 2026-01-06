"""PathOption dataclass for RL action selection.

This module defines the PathOption dataclass that represents a candidate path
with its metadata. It is used by the RLSimulationAdapter for:
- Action masking (via is_feasible field)
- Observation building (path features)
- Action application (path selection)

Phase: P4.1 - RLSimulationAdapter Scaffolding
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class PathOption:
    """Represents a candidate path for RL action selection.

    This dataclass encapsulates all information about a candidate path
    that an RL agent needs to make a routing decision. The is_feasible
    field is computed from REAL spectrum pipeline checks, not mocked.

    Attributes:
        path_index: Index in the k-paths list (0 to k-1)
        path: Sequence of node IDs representing the path (immutable tuple)
        weight_km: Total path length in kilometers
        num_hops: Number of links in the path
        modulation: Selected modulation format (e.g., "QPSK", "16-QAM")
        slots_needed: Number of contiguous spectrum slots required
        is_feasible: True if path can be allocated (from real spectrum check)
        congestion: Congestion metric in [0, 1], higher = more congested
        available_slots: Ratio of available slots on most constrained link
        spectrum_start: Start slot if feasible, None otherwise
        spectrum_end: End slot if feasible, None otherwise
        core_index: Core index if multi-core, None for single-core
        band: Spectrum band (e.g., "C", "L") if multi-band, None otherwise
        frag_indicator: Path fragmentation indicator [0, 1], 0 = no fragmentation
        failure_mask: Whether path passes through failed links (disaster scenarios)
        dist_to_disaster: Normalized distance from path to disaster centroid [0, 1]
        min_residual_slots: Minimum residual slots along path, normalized [0, 1]
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

    def __post_init__(self) -> None:
        """Validate invariants after initialization."""
        if self.path_index < 0:
            raise ValueError("path_index must be non-negative")
        if self.weight_km < 0:
            raise ValueError("weight_km must be non-negative")
        if self.congestion < 0 or self.congestion > 1:
            raise ValueError("congestion must be in [0, 1]")
        if self.available_slots < 0 or self.available_slots > 1:
            raise ValueError("available_slots must be in [0, 1]")

    @classmethod
    def from_pipeline_results(
        cls,
        path_index: int,
        route_result: Any,
        spectrum_result: Any | None,
        congestion: float,
        available_slots: float,
    ) -> PathOption:
        """Create PathOption from V4 pipeline results.

        Factory method to construct a PathOption from routing and spectrum
        pipeline results. This provides a clean interface between the
        pipeline layer and the RL adapter.

        Args:
            path_index: Index in k-paths list
            route_result: Result from routing pipeline with path, weight_km,
                modulation attributes
            spectrum_result: Result from spectrum pipeline with is_free,
                start_slot, end_slot, core, band attributes (or None)
            congestion: Pre-computed congestion metric [0, 1]
            available_slots: Pre-computed available slots ratio [0, 1]

        Returns:
            PathOption instance
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


# Type aliases for clarity
PathOptionList: TypeAlias = list[PathOption]
ActionMask: TypeAlias = np.ndarray  # Shape: (k_paths,), dtype: bool


def compute_action_mask(options: list[PathOption], k_paths: int) -> np.ndarray:
    """Generate action mask from path options.

    Creates a boolean mask where True indicates a valid (feasible) action.
    This is used by action-masked RL algorithms like MaskablePPO.

    Args:
        options: List of PathOption instances
        k_paths: Total number of possible actions (action space size)

    Returns:
        Boolean array of shape (k_paths,) where True = action is valid
    """
    mask = np.zeros(k_paths, dtype=bool)
    for opt in options:
        if opt.path_index < k_paths:
            mask[opt.path_index] = opt.is_feasible
    return mask
