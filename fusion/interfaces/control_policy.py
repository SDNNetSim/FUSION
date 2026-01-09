"""
ControlPolicy protocol for unified path selection.

This module defines the ControlPolicy protocol that all path selection
strategies must implement: heuristics, RL policies, and ML policies.

The protocol uses Python's structural typing (Protocol) to allow any class
with the required methods to be used as a policy without explicit inheritance.

Example:
    >>> class MyPolicy:
    ...     def select_action(self, request, options, network_state) -> int:
    ...         return 0  # Always select first option
    ...     def update(self, request, action, reward) -> None:
    ...         pass  # No learning
    ...     def get_name(self) -> str:
    ...         return "MyPolicy"
    >>>
    >>> policy = MyPolicy()
    >>> isinstance(policy, ControlPolicy)  # True
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeAlias, runtime_checkable

if TYPE_CHECKING:
    from fusion.domain.network_state import NetworkState
    from fusion.domain.request import Request
    from fusion.modules.rl.adapter import PathOption


@runtime_checkable
class ControlPolicy(Protocol):
    """
    Protocol for control policies that select actions for resource allocation.

    This protocol defines the interface for all path selection strategies
    in the FUSION simulation framework. Implementations include:

    - Heuristic policies: Rule-based selection (first-fit, shortest-path)
    - RL policies: Reinforcement learning agents (PPO, DQN, etc.)
    - ML policies: Pre-trained neural networks or classifiers
    - Composite policies: FallbackPolicy, TiebreakingPolicy

    All policies must:

    1. Respect feasibility: Only select paths where PathOption.is_feasible is True
    2. Return valid indices: Return 0 to len(options)-1, or -1 for no valid action
    3. Never mutate state: NetworkState must remain unchanged during select_action
    4. Provide a name: Return descriptive name via get_name() for logging

    Example:
        >>> policy = FirstFeasiblePolicy()
        >>> action = policy.select_action(request, options, network_state)
        >>> if action >= 0:
        ...     result = orchestrator.apply_action(action, request, options)
        ...     policy.update(request, action, result.reward)
        ...     logger.info(f"Policy {policy.get_name()} selected action {action}")
        >>> else:
        ...     # No feasible path - request blocked
        ...     handle_blocking(request)
    """

    def select_action(
        self,
        request: Request,
        options: list[PathOption],
        network_state: NetworkState,
    ) -> int:
        """
        Select an action (path index) for the given request.

        This method is the core decision-making interface. It receives the
        current request, available path options (with feasibility information),
        and read-only network state. It returns the index of the selected path.

        Args:
            request: The incoming request to serve. Contains source, destination,
                bandwidth requirements, and timing information.
            options: List of available path options, each with:
                - path_index: Index in this list (0 to len-1)
                - path: Node sequence
                - weight_km: Path length
                - is_feasible: Whether spectrum is available
                - congestion: Current congestion level (0-1)
                - slots_needed: Required spectrum slots
                - modulation: Selected modulation format
                For protected paths, also includes backup_path, backup_feasible.
            network_state: Current state of the network. This is read-only;
                policies must not modify network state. Use for additional
                context (e.g., global congestion, topology info).

        Returns:
            int: Path index (0 to len(options)-1) for the selected path,
                or -1 if no valid action exists.

        Note:
            - Policies MUST only return indices where options[i].is_feasible is True
            - For protected paths, check options[i].both_paths_feasible for full protection
            - Returning an infeasible index is undefined behavior (orchestrator may reject)
        """
        ...

    def update(self, request: Request, action: int, reward: float) -> None:
        """
        Update policy based on experience.

        Called after an action is executed and the reward is computed. This
        enables online learning for RL policies. Heuristic and pre-trained ML
        policies typically implement this as a no-op.

        Args:
            request: The request that was served
            action: The action (path index) that was taken
            reward: The reward received. Typically:
                - Positive for successful allocation
                - Negative for blocking
                - May include shaping terms (fragmentation, utilization)

        Returns:
            None

        Note:
            - Heuristic policies should implement this as `pass`
            - RL policies may update internal state, replay buffers, etc.
            - ML policies (pre-trained) typically implement as `pass`
            - This method should not raise exceptions
        """
        ...

    def get_name(self) -> str:
        """
        Return the policy name for logging and metrics.

        This method enables meaningful logging messages and metrics tracking.
        Names should be descriptive and include relevant configuration.

        Returns:
            str: Human-readable policy name

        Examples:
            - "FirstFeasiblePolicy"
            - "ShortestFeasiblePolicy"
            - "LoadBalancedPolicy(alpha=0.5)"
            - "RLPolicy(PPO)"
            - "MLControlPolicy(pytorch)"
            - "FallbackPolicy(MLControlPolicy->FirstFeasiblePolicy)"
        """
        ...


# Type alias for policy action results
PolicyAction: TypeAlias = int
"""Type alias for policy action: -1 for invalid, 0 to k-1 for valid path index."""
