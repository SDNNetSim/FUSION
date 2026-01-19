"""
Heuristic policies for path selection.

This module provides deterministic, rule-based path selection strategies
implementing the ControlPolicy protocol. Available policies:

- FirstFeasiblePolicy: Select first feasible path (K-shortest first fit)
- ShortestFeasiblePolicy: Select shortest feasible path by distance
- LeastCongestedPolicy: Select least congested feasible path
- RandomFeasiblePolicy: Randomly select among feasible paths
- LoadBalancedPolicy: Balance path length and congestion

All policies implement the ControlPolicy protocol but do not learn
from experience (update() is a no-op).

Example:
    >>> from fusion.policies.heuristic_policy import ShortestFeasiblePolicy
    >>> policy = ShortestFeasiblePolicy()
    >>> action = policy.select_action(request, options, network_state)

Note:
    Heuristic policies are typically used as:
    - Default selection strategies
    - Baselines for RL/ML comparison
    - Fallback when ML models fail
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from fusion.domain.network_state import NetworkState
    from fusion.domain.request import Request
    from fusion.modules.rl.adapter import PathOption


class HeuristicPolicy(ABC):
    """
    Abstract base class for heuristic path selection policies.

    Heuristic policies are deterministic (except RandomFeasiblePolicy),
    rule-based strategies that select paths without learning. They serve as:

    1. Default selection strategies
    2. Baselines for RL/ML comparison
    3. Fallback policies when ML models fail

    All subclasses must implement select_action(). The update() method
    is a no-op since heuristics don't learn.

    This class implements the ControlPolicy protocol.
    """

    @abstractmethod
    def select_action(
        self,
        request: Request,
        options: list[PathOption],
        network_state: NetworkState,
    ) -> int:
        """
        Select an action (path index) based on heuristic rule.

        :param request: The incoming request (may be used for context).
        :type request: Request
        :param options: List of available PathOptions.
        :type options: list[PathOption]
        :param network_state: Current network state (read-only).
        :type network_state: NetworkState
        :return: Selected path index (0 to len(options)-1), or -1 if none valid.
        :rtype: int
        """
        ...

    def update(  # noqa: B027
        self, request: Request, action: int, reward: float
    ) -> None:
        """
        Update policy based on experience.

        Heuristic policies do not learn, so this is a no-op.
        This is intentionally not abstract - subclasses inherit this no-op.

        :param request: The request that was served (ignored).
        :type request: Request
        :param action: The action taken (ignored).
        :type action: int
        :param reward: The reward received (ignored).
        :type reward: float
        """
        pass

    def get_name(self) -> str:
        """
        Return the policy name for logging.

        :return: Policy class name.
        :rtype: str
        """
        return self.__class__.__name__

    def _get_feasible_options(
        self,
        options: list[PathOption],
    ) -> list[PathOption]:
        """
        Filter options to only feasible ones.

        :param options: List of path options.
        :type options: list[PathOption]
        :return: Filtered list containing only feasible options.
        :rtype: list[PathOption]
        """
        return [opt for opt in options if opt.is_feasible]


class FirstFeasiblePolicy(HeuristicPolicy):
    """
    Select the first feasible path in index order.

    This is the simplest heuristic. It iterates through options and
    returns the first path where is_feasible=True. Equivalent to
    "K-Shortest Path First Fit" when paths are pre-sorted by length.

    Selection Logic:
        1. Iterate through options in index order
        2. Return first option where is_feasible=True
        3. Return -1 if no feasible option exists

    Time Complexity: O(n) worst case, O(1) best case
    Space Complexity: O(1)
    """

    def select_action(
        self,
        request: Request,
        options: list[PathOption],
        network_state: NetworkState,
    ) -> int:
        """Select the first feasible path."""
        for opt in options:
            if opt.is_feasible:
                return opt.path_index
        return -1


class ShortestFeasiblePolicy(HeuristicPolicy):
    """
    Select the shortest feasible path by distance.

    Finds all feasible paths and selects the one with minimum
    weight_km (path length in kilometers).

    Selection Logic:
        1. Filter to feasible options
        2. Find option with minimum weight_km
        3. Return its path_index, or -1 if none feasible

    Tie Breaking: First occurrence when tied on weight_km.

    Time Complexity: O(n)
    Space Complexity: O(n) for feasible list
    """

    def select_action(
        self,
        request: Request,
        options: list[PathOption],
        network_state: NetworkState,
    ) -> int:
        """Select the shortest feasible path by weight_km."""
        feasible = self._get_feasible_options(options)

        if not feasible:
            return -1

        shortest = min(feasible, key=lambda opt: opt.weight_km)
        return shortest.path_index


class LeastCongestedPolicy(HeuristicPolicy):
    """
    Select the least congested feasible path.

    Prioritizes paths with lower congestion values to distribute
    load across the network and reduce fragmentation.

    Selection Logic:
        1. Filter to feasible options
        2. Find option with minimum congestion (0.0 to 1.0)
        3. Return its path_index, or -1 if none feasible

    Tie Breaking: First occurrence when tied on congestion.

    Time Complexity: O(n)
    Space Complexity: O(n) for feasible list
    """

    def select_action(
        self,
        request: Request,
        options: list[PathOption],
        network_state: NetworkState,
    ) -> int:
        """Select the least congested feasible path."""
        feasible = self._get_feasible_options(options)

        if not feasible:
            return -1

        least_congested = min(feasible, key=lambda opt: opt.congestion)
        return least_congested.path_index


class RandomFeasiblePolicy(HeuristicPolicy):
    """
    Randomly select among feasible paths.

    Uniformly samples from all feasible paths. Useful for:

    - Exploration during training
    - Baseline comparison (random performance)
    - Load distribution across multiple paths

    Uses numpy's random number generator with optional seed.

    Selection Logic:

    1. Filter to feasible options
    2. Uniformly sample one option
    3. Return its path_index, or -1 if none feasible

    Time Complexity: O(n)
    Space Complexity: O(n) for feasible list

    :ivar _rng: Numpy random generator.
    :vartype _rng: numpy.random.Generator
    :ivar _seed: Original seed for reset.
    :vartype _seed: int | None
    """

    def __init__(self, seed: int | None = None) -> None:
        """
        Initialize with optional random seed.

        :param seed: Random seed for reproducibility. If None, uses
            system entropy (non-reproducible).
        :type seed: int | None
        """
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def select_action(
        self,
        request: Request,
        options: list[PathOption],
        network_state: NetworkState,
    ) -> int:
        """Randomly select a feasible path."""
        feasible = self._get_feasible_options(options)

        if not feasible:
            return -1

        # Use integer index to avoid numpy type issues
        idx = int(self._rng.integers(0, len(feasible)))
        return feasible[idx].path_index

    def reset_rng(self, seed: int | None = None) -> None:
        """
        Reset the random number generator.

        :param seed: New seed. If None, uses the original seed.
        :type seed: int | None
        """
        if seed is None:
            seed = self._seed
        self._rng = np.random.default_rng(seed)

    def get_name(self) -> str:
        """
        Return policy name including seed.

        :return: Policy name with seed if set.
        :rtype: str
        """
        if self._seed is not None:
            return f"RandomFeasiblePolicy(seed={self._seed})"
        return "RandomFeasiblePolicy"


class LoadBalancedPolicy(HeuristicPolicy):
    """
    Select path balancing length and congestion.

    Combines path length (weight_km) and congestion into a weighted score::

        score = alpha * normalized_length + (1 - alpha) * congestion

    Where normalized_length = weight_km / max_weight_km among feasible paths.

    Alpha Parameter:

    - 0.0: Pure congestion-based (same as LeastCongestedPolicy)
    - 0.5: Equal weight to length and congestion (default)
    - 1.0: Pure length-based (same as ShortestFeasiblePolicy)

    Selection Logic:

    1. Filter to feasible options
    2. Normalize weight_km to [0, 1] range
    3. Compute weighted score for each option
    4. Return option with minimum score

    Tie Breaking: First occurrence when tied on score.

    Time Complexity: O(n)
    Space Complexity: O(n) for feasible list

    :ivar _alpha: Weight for path length (0.0 to 1.0).
    :vartype _alpha: float
    """

    def __init__(self, alpha: float = 0.5) -> None:
        """
        Initialize with load balancing weight.

        :param alpha: Weight for path length (0.0 to 1.0).
            0.0 = prioritize low congestion,
            1.0 = prioritize short length,
            0.5 = equal balance (default).
        :type alpha: float
        :raises ValueError: If alpha is not in [0, 1] range.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self._alpha = alpha

    @property
    def alpha(self) -> float:
        """
        Current alpha value.

        :return: The alpha weight parameter.
        :rtype: float
        """
        return self._alpha

    def select_action(
        self,
        request: Request,
        options: list[PathOption],
        network_state: NetworkState,
    ) -> int:
        """Select path with minimum weighted score."""
        feasible = self._get_feasible_options(options)

        if not feasible:
            return -1

        max_weight = max(opt.weight_km for opt in feasible)
        if max_weight == 0:
            max_weight = 1.0  # Avoid division by zero

        def compute_score(opt: PathOption) -> float:
            normalized_length = opt.weight_km / max_weight
            return self._alpha * normalized_length + (1 - self._alpha) * opt.congestion

        best = min(feasible, key=compute_score)
        return best.path_index

    def get_name(self) -> str:
        """
        Return policy name including alpha.

        :return: Policy name with alpha parameter.
        :rtype: str
        """
        return f"LoadBalancedPolicy(alpha={self._alpha})"
