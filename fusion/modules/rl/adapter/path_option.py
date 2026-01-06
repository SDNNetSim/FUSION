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
from typing import TYPE_CHECKING, TypeAlias

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
    """

    path_index: int
    path: tuple[str, ...]
    weight_km: float
    num_hops: int
    modulation: str | None
    slots_needed: int
    is_feasible: bool
    congestion: float
    available_slots: float
    spectrum_start: int | None = None
    spectrum_end: int | None = None
    core_index: int | None = None
    band: str | None = None

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
