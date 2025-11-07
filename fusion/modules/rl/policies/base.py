"""
Abstract interface for path selection policies.

This module defines the base PathPolicy interface that all routing policies
(heuristic and RL-based) must implement to ensure consistent integration
with the SDN controller.
"""

from abc import ABC, abstractmethod
from typing import Any


class PathPolicy(ABC):
    """
    Abstract interface for path selection policies.

    All policies (heuristic and RL-based) must implement this interface
    to ensure consistent integration with the SDN controller.
    """

    @abstractmethod
    def select_path(self, state: dict[str, Any], action_mask: list[bool]) -> int:
        """
        Select a path index from K candidates.

        :param state: State dictionary with request and path features
        :type state: dict[str, Any]
        :param action_mask: Feasibility mask for K paths (True = feasible)
        :type action_mask: list[bool]
        :return: Selected path index (0 to K-1), or -1 if all paths are masked
        :rtype: int

        When all paths are masked (infeasible), returns -1 to indicate
        the request should be blocked. This is a normal occurrence in
        network simulations and contributes to blocking probability metrics.

        State format:
            {
                'src': int,
                'dst': int,
                'slots_needed': int,
                'est_remaining_time': float,
                'is_disaster': int (0 or 1),
                'paths': [
                    {
                        'path_hops': int,
                        'min_residual_slots': int,
                        'frag_indicator': float,
                        'failure_mask': int,
                        'dist_to_disaster_centroid': int
                    },
                    ...  # K paths
                ]
            }
        """
        pass

    def get_name(self) -> str:
        """
        Get policy name.

        :return: Policy name
        :rtype: str
        """
        return self.__class__.__name__
