"""
K-Shortest Path First-Fit policy (baseline).

This module implements the standard KSP-FF heuristic baseline used in most
optical network studies. It always selects the first feasible path from the
K shortest paths.
"""

from typing import Any

from .base import AllPathsMaskedError, PathPolicy


class KSPFFPolicy(PathPolicy):
    """
    KSP-FF baseline: always select first feasible path.

    This is the standard heuristic baseline used in most
    optical network studies.

    Example:
        >>> policy = KSPFFPolicy()
        >>> action_mask = [False, True, True, False]
        >>> selected = policy.select_path(state, action_mask)
        >>> print(selected)
        1  # First feasible path
    """

    def select_path(self, state: dict[str, Any], action_mask: list[bool]) -> int:
        """
        Select first unmasked path.

        :param state: Current state (not used by KSP-FF)
        :type state: dict[str, Any]
        :param action_mask: Feasibility mask
        :type action_mask: list[bool]
        :return: Index of first feasible path
        :rtype: int
        :raises AllPathsMaskedError: If all paths masked
        """
        for i, is_feasible in enumerate(action_mask):
            if is_feasible:
                return i

        raise AllPathsMaskedError("All K paths are infeasible")
