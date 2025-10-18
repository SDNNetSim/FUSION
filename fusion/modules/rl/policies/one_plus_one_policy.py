"""
1+1 protection policy (baseline).

This module implements the 1+1 protection policy that selects from
pre-computed disjoint paths based on failure state.
"""

from typing import Any

from .base import PathPolicy


class OnePlusOnePolicy(PathPolicy):
    """
    1+1 policy: use primary if feasible, else backup.

    Selects from pre-computed disjoint paths based on
    failure state.

    Example:
        >>> policy = OnePlusOnePolicy()
        >>> # Primary path masked (failed)
        >>> action_mask = [False, True]
        >>> selected = policy.select_path(state, action_mask)
        >>> print(selected)
        1  # Backup path
    """

    def select_path(self, state: dict[str, Any], action_mask: list[bool]) -> int:
        """
        Select primary (index 0) if feasible, else backup (index 1).

        :param state: Current state
        :type state: dict[str, Any]
        :param action_mask: Feasibility mask
        :type action_mask: list[bool]
        :return: 0 (primary), 1 (backup), or -1 if both paths masked
        :rtype: int
        """
        # Try primary first
        if action_mask[0]:
            return 0

        # Fall back to backup
        if len(action_mask) > 1 and action_mask[1]:
            return 1

        return -1  # Both paths masked - request should be blocked
