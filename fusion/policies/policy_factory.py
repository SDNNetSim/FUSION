"""
PolicyFactory for instantiating control policies.

This module provides a factory to create policy instances based on
configuration, supporting:
- Heuristic policies (first_feasible, shortest, least_congested, random, load_balanced)
- ML policies (via MLControlPolicy)
- RL policies (via RLPolicy.from_file)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fusion.interfaces.control_policy import ControlPolicy
from fusion.policies.heuristic_policy import (
    FirstFeasiblePolicy,
    LeastCongestedPolicy,
    LoadBalancedPolicy,
    RandomFeasiblePolicy,
    ShortestFeasiblePolicy,
)

if TYPE_CHECKING:
    from fusion.policies.ml_policy import MLControlPolicy
    from fusion.policies.rl_policy import RLPolicy

logger = logging.getLogger(__name__)


@dataclass
class PolicyConfig:
    """
    Configuration for policy instantiation.

    :ivar policy_type: Type of policy ("heuristic", "ml", "rl").
    :vartype policy_type: str
    :ivar policy_name: Name of specific policy variant.
    :vartype policy_name: str
    :ivar model_path: Path to model file (for ml/rl types).
    :vartype model_path: str | None
    :ivar fallback_policy: Fallback policy name if primary fails.
    :vartype fallback_policy: str
    :ivar k_paths: Number of candidate paths (for rl).
    :vartype k_paths: int
    :ivar seed: Random seed (for random policies).
    :vartype seed: int | None
    :ivar alpha: Balance parameter (for load_balanced policy).
    :vartype alpha: float
    :ivar algorithm: RL algorithm name (for rl type, e.g., "PPO", "MaskablePPO").
    :vartype algorithm: str
    """

    policy_type: str = "heuristic"
    policy_name: str = "first_feasible"
    model_path: str | None = None
    fallback_policy: str = "first_feasible"
    k_paths: int = 3
    seed: int | None = None
    alpha: float = 0.5
    algorithm: str = "PPO"


# Registry of heuristic policies
HEURISTIC_POLICIES: dict[str, type[ControlPolicy]] = {
    "first_feasible": FirstFeasiblePolicy,
    "shortest": ShortestFeasiblePolicy,
    "shortest_feasible": ShortestFeasiblePolicy,
    "least_congested": LeastCongestedPolicy,
    "random": RandomFeasiblePolicy,
    "random_feasible": RandomFeasiblePolicy,
    "load_balanced": LoadBalancedPolicy,
}


class PolicyFactory:
    """
    Factory for creating control policy instances.

    This factory instantiates policies based on configuration, supporting
    multiple policy types:

    1. Heuristic policies: Built-in rule-based policies
    2. ML policies: Pre-trained PyTorch/sklearn/ONNX models
    3. RL policies: Pre-trained Stable-Baselines3 models

    Default policy is FirstFeasiblePolicy when not specified.

    Example::

        >>> config = PolicyConfig(policy_type="heuristic", policy_name="shortest")
        >>> policy = PolicyFactory.create(config)
        >>> action = policy.select_action(request, options, network_state)
    """

    @staticmethod
    def create(config: PolicyConfig | None = None) -> ControlPolicy:
        """
        Create a policy instance from configuration.

        :param config: Policy configuration. If None, creates FirstFeasiblePolicy.
        :type config: PolicyConfig | None
        :return: ControlPolicy instance.
        :rtype: ControlPolicy
        :raises ValueError: If policy type/name is unknown or model_path is missing.
        """
        if config is None:
            logger.debug("No policy config provided, using FirstFeasiblePolicy")
            return FirstFeasiblePolicy()

        policy_type = config.policy_type.lower()

        if policy_type == "heuristic":
            return PolicyFactory._create_heuristic(config)
        elif policy_type == "ml":
            return PolicyFactory._create_ml(config)
        elif policy_type == "rl":
            return PolicyFactory._create_rl(config)
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")

    @staticmethod
    def _create_heuristic(config: PolicyConfig) -> ControlPolicy:
        """
        Create heuristic policy instance.

        :param config: Policy configuration.
        :type config: PolicyConfig
        :return: Heuristic policy instance.
        :rtype: ControlPolicy
        :raises ValueError: If policy name is unknown.
        """
        policy_name = config.policy_name.lower()

        if policy_name not in HEURISTIC_POLICIES:
            available = ", ".join(HEURISTIC_POLICIES.keys())
            raise ValueError(
                f"Unknown heuristic policy: {policy_name}. "
                f"Available: {available}"
            )

        policy_class = HEURISTIC_POLICIES[policy_name]

        # Handle policies with special constructor arguments
        if policy_name in ("random", "random_feasible"):
            policy = policy_class(seed=config.seed)  # type: ignore[call-arg]
        elif policy_name == "load_balanced":
            policy = policy_class(alpha=config.alpha)  # type: ignore[call-arg]
        else:
            policy = policy_class()

        logger.info("Created heuristic policy: %s", policy.get_name())
        return policy

    @staticmethod
    def _create_ml(config: PolicyConfig) -> MLControlPolicy:
        """
        Create ML policy instance.

        :param config: Policy configuration.
        :type config: PolicyConfig
        :return: ML policy instance.
        :rtype: MLControlPolicy
        :raises ValueError: If model_path is not provided.
        :raises FileNotFoundError: If model file does not exist.
        """
        from fusion.policies.ml_policy import MLControlPolicy

        if not config.model_path:
            raise ValueError("model_path required for ML policy")

        model_path = Path(config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"ML model not found: {model_path}")

        # Create fallback policy
        fallback = PolicyFactory._create_fallback(config.fallback_policy)

        policy = MLControlPolicy(
            model_path=str(model_path),
            fallback_policy=fallback,
            k_paths=config.k_paths,
        )

        logger.info("Created ML policy from %s with fallback %s",
                    model_path.name, fallback.get_name())
        return policy

    @staticmethod
    def _create_rl(config: PolicyConfig) -> RLPolicy:
        """
        Create RL policy instance.

        :param config: Policy configuration.
        :type config: PolicyConfig
        :return: RL policy instance.
        :rtype: RLPolicy
        :raises ValueError: If model_path is not provided.
        :raises FileNotFoundError: If model file does not exist.
        """
        from fusion.policies.rl_policy import RLPolicy

        if not config.model_path:
            raise ValueError("model_path required for RL policy")

        model_path = Path(config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"RL model not found: {model_path}")

        policy = RLPolicy.from_file(
            model_path=str(model_path),
            algorithm=config.algorithm,
            k_paths=config.k_paths,
        )

        logger.info("Created RL policy from %s (algorithm=%s)",
                    model_path.name, config.algorithm)
        return policy

    @staticmethod
    def _create_fallback(policy_name: str) -> ControlPolicy:
        """
        Create a fallback heuristic policy.

        :param policy_name: Name of the fallback policy.
        :type policy_name: str
        :return: Fallback policy instance.
        :rtype: ControlPolicy
        """
        policy_name = policy_name.lower()

        if policy_name not in HEURISTIC_POLICIES:
            logger.warning(
                "Unknown fallback policy %s, using first_feasible",
                policy_name,
            )
            return FirstFeasiblePolicy()

        policy_class = HEURISTIC_POLICIES[policy_name]
        return policy_class()

    @staticmethod
    def from_dict(config_dict: dict[str, Any]) -> ControlPolicy:
        """
        Create policy from dictionary configuration.

        Convenience method for creating policies from config files.

        :param config_dict: Dictionary with policy configuration.
        :type config_dict: dict[str, Any]
        :return: ControlPolicy instance.
        :rtype: ControlPolicy
        """
        config = PolicyConfig(
            policy_type=config_dict.get("policy_type", "heuristic"),
            policy_name=config_dict.get("policy_name", "first_feasible"),
            model_path=config_dict.get("model_path"),
            fallback_policy=config_dict.get("fallback_policy", "first_feasible"),
            k_paths=config_dict.get("k_paths", 3),
            seed=config_dict.get("seed"),
            alpha=config_dict.get("alpha", 0.5),
            algorithm=config_dict.get("algorithm", "PPO"),
        )
        return PolicyFactory.create(config)

    @staticmethod
    def get_default_policy() -> ControlPolicy:
        """
        Get the default policy (FirstFeasiblePolicy).

        :return: FirstFeasiblePolicy instance.
        :rtype: ControlPolicy
        """
        return FirstFeasiblePolicy()
