"""
Offline RL dataset logger for FUSION.

This module provides functionality for logging simulation transitions in JSONL
format for offline RL training (BC, IQL). Each transition includes state,
action, reward, action mask, and metadata.
"""

import json
import logging
import random
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DatasetLogger:
    """
    Log offline RL training data to JSON Lines format.

    Logs tuples (s, a, r, s', action_mask, meta) for offline training.
    Each line is a valid JSON object representing one transition.

    :param output_path: Path to output JSONL file
    :type output_path: str
    :param engine_props: Engine configuration
    :type engine_props: dict[str, Any]

    Example:
        >>> logger = DatasetLogger('datasets/offline_data.jsonl', engine_props)
        >>> logger.log_transition(state, action, reward, next_state, mask, meta)
        >>> logger.close()
    """

    def __init__(self, output_path: str, engine_props: dict[str, Any]) -> None:
        """
        Initialize dataset logger.

        :param output_path: Output file path
        :type output_path: str
        :param engine_props: Engine configuration
        :type engine_props: dict[str, Any]
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.file_handle = open(self.output_path, "w", encoding="utf-8")
        self.engine_props = engine_props
        self.transition_count = 0

        logger.info(f"Dataset logger initialized: {self.output_path}")

    def log_transition(
        self,
        state: dict[str, Any],
        action: int,
        reward: float,
        next_state: dict[str, Any] | None,
        action_mask: list[bool],
        meta: dict[str, Any],
    ) -> None:
        """
        Log a single transition.

        :param state: Current state dict
        :type state: dict[str, Any]
        :param action: Selected action (path index)
        :type action: int
        :param reward: Reward (1.0 for accept, -1.0 for block)
        :type reward: float
        :param next_state: Next state dict (or None)
        :type next_state: dict[str, Any] | None
        :param action_mask: Feasibility mask
        :type action_mask: list[bool]
        :param meta: Metadata (decision_time_ms, bp_window_tag, etc.)
        :type meta: dict[str, Any]

        Example:
            >>> logger.log_transition(
            ...     state={'src': 0, 'dst': 5, ...},
            ...     action=2,
            ...     reward=1.0,
            ...     next_state=None,
            ...     action_mask=[False, True, True, False],
            ...     meta={'decision_time_ms': 0.35, 'bp_window_tag': 'pre'}
            ... )
        """
        transition = {
            "t": self.transition_count,
            "seed": self.engine_props.get("seed"),
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "action_mask": action_mask,
            "meta": meta,
        }

        # Write as single JSON line
        json.dump(transition, self.file_handle)
        self.file_handle.write("\n")
        self.file_handle.flush()  # Ensure immediate write

        self.transition_count += 1

        if self.transition_count % 1000 == 0:
            logger.debug(f"Logged {self.transition_count} transitions")

    def close(self) -> None:
        """
        Close the output file.

        Example:
            >>> logger.close()
        """
        if self.file_handle:
            self.file_handle.close()
            logger.info(f"Dataset logger closed: {self.transition_count} transitions logged")

    def __enter__(self) -> "DatasetLogger":
        """Context manager entry."""
        return self

    def __exit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()


def select_path_with_epsilon_mix(policy: Any, state: dict[str, Any], action_mask: list[bool], epsilon: float = 0.1) -> int:
    """
    Select path with epsilon probability of picking second-best.

    Adds behavior diversity to offline dataset to improve
    generalization of trained policies.

    :param policy: Path selection policy
    :type policy: Any (PathPolicy)
    :param state: Current state
    :type state: dict[str, Any]
    :param action_mask: Action mask
    :type action_mask: list[bool]
    :param epsilon: Probability of selecting second-best path
    :type epsilon: float
    :return: Selected path index
    :rtype: int

    Example:
        >>> from fusion.modules.rl.policies import KSPFFPolicy
        >>> policy = KSPFFPolicy()
        >>> idx = select_path_with_epsilon_mix(policy, state, mask, epsilon=0.1)
    """
    # Get feasible paths
    feasible_indices = [i for i, m in enumerate(action_mask) if m]

    if not feasible_indices:
        return -1  # All blocked

    # With probability epsilon, select second-best
    if random.random() < epsilon and len(feasible_indices) >= 2:  # nosec B311 - Simulation randomness, not cryptographic
        return feasible_indices[1]  # Second feasible path

    # Otherwise, use policy
    selected_path: int = policy.select_path(state, action_mask)
    return selected_path


def load_dataset(file_path: str) -> Any:
    """
    Load transitions from JSONL file.

    :param file_path: Path to JSONL file
    :type file_path: str
    :yield: Transition dictionaries

    Example:
        >>> for transition in load_dataset('datasets/offline_data.jsonl'):
        ...     state = transition['state']
        ...     action = transition['action']
        ...     reward = transition['reward']
        ...     # Train model...
    """
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def filter_by_window(dataset_path: str, window_tag: str) -> list[dict]:
    """
    Filter dataset by BP window tag.

    :param dataset_path: Path to dataset
    :type dataset_path: str
    :param window_tag: Window tag ('pre', 'fail', 'post')
    :type window_tag: str
    :return: Filtered transitions
    :rtype: list[dict]

    Example:
        >>> pre_failure = filter_by_window('data.jsonl', 'pre')
        >>> failure_window = filter_by_window('data.jsonl', 'fail')
    """
    filtered = []
    for transition in load_dataset(dataset_path):
        if transition["meta"]["bp_window_tag"] == window_tag:
            filtered.append(transition)
    return filtered
