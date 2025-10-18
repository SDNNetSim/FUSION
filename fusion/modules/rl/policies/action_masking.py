"""
Action masking utilities for safe RL deployment.

This module provides utilities for computing action masks based on
network conditions and applying fallback policies when all actions
are masked.
"""

from typing import Any

from .base import PathPolicy


def compute_action_mask(
    k_paths: list[list[int]], k_path_features: list[dict[str, Any]], slots_needed: int
) -> list[bool]:
    """
    Compute feasibility mask for K candidate paths.

    A path is masked (infeasible) if:
    - failure_mask == 1 (path uses failed link)
    - min_residual_slots < slots_needed (insufficient spectrum)

    :param k_paths: K candidate paths
    :type k_paths: list[list[int]]
    :param k_path_features: Features for each path
    :type k_path_features: list[dict[str, Any]]
    :param slots_needed: Required contiguous slots
    :type slots_needed: int
    :return: Boolean mask (True = feasible, False = masked)
    :rtype: list[bool]

    Example:
        >>> mask = compute_action_mask(
        ...     k_paths=[[0,1,2], [0,3,2], [0,4,5,2]],
        ...     k_path_features=features,
        ...     slots_needed=4
        ... )
        >>> print(mask)
        [False, True, True]  # First path infeasible
    """
    mask = []

    for features in k_path_features:
        # Check failure condition
        if features["failure_mask"] == 1:
            mask.append(False)
            continue

        # Check spectrum availability
        if features["min_residual_slots"] < slots_needed:
            mask.append(False)
            continue

        # Path is feasible
        mask.append(True)

    return mask


def apply_fallback_policy(
    state: dict[str, Any], fallback_policy: PathPolicy, action_mask: list[bool]
) -> int:
    """
    Apply fallback policy when all actions are masked.

    Attempts to use fallback policy (typically KSP-FF) with
    relaxed constraints or alternative path set.

    :param state: Current state
    :type state: dict[str, Any]
    :param fallback_policy: Fallback policy (KSP-FF or 1+1)
    :type fallback_policy: PathPolicy
    :param action_mask: Current action mask
    :type action_mask: list[bool]
    :return: Fallback path index or -1 if all blocked
    :rtype: int

    Example:
        >>> from .ksp_ff_policy import KSPFFPolicy
        >>> fallback = KSPFFPolicy()
        >>> idx = apply_fallback_policy(state, fallback, action_mask)
        >>> if idx == -1:
        ...     print("Request blocked")
    """
    # Try fallback with full feasibility mask
    # (fallback may use different logic)
    # Returns -1 if fallback also cannot find a path
    return fallback_policy.select_path(state, [True] * len(state["paths"]))
