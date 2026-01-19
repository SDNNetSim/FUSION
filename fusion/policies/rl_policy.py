"""
RL Policy wrapper for Stable-Baselines3 models.

This module provides the RLPolicy class that wraps pre-trained SB3 models
to implement the ControlPolicy protocol, enabling unified policy handling
in the SDNOrchestrator.

Phase: P5.1 - ControlPolicy Protocol + RLPolicy Adapter
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from fusion.utils.logging_config import get_logger

if TYPE_CHECKING:
    from stable_baselines3.common.base_class import BaseAlgorithm

    from fusion.domain.network_state import NetworkState
    from fusion.domain.request import Request
    from fusion.modules.rl.adapter import PathOption, RLSimulationAdapter

logger = get_logger(__name__)


class RLPolicy:
    """
    Wrapper enabling SB3 models to implement ControlPolicy.

    This adapter bridges the existing RL infrastructure with the
    ControlPolicy protocol. It handles:

    1. Observation building: Converts (request, options, state) to RL observation
    2. Action masking: Enforces feasibility constraints during prediction
    3. Action conversion: Converts SB3 action to path index

    The wrapped model is pre-trained and does not learn online. Use
    UnifiedSimEnv and SB3's learn() for online training.

    Attributes:
        model: Pre-trained SB3 model (PPO, DQN, A2C, etc.)
        k_paths: Number of path options (for observation space)

    Example:
        >>> from stable_baselines3 import PPO
        >>> model = PPO.load("trained_model.zip")
        >>> policy = RLPolicy(model)
        >>> action = policy.select_action(request, options, network_state)
    """

    def __init__(
        self,
        model: BaseAlgorithm,
        adapter: RLSimulationAdapter | None = None,
        k_paths: int = 5,
    ) -> None:
        """
        Initialize RLPolicy with a trained SB3 model.

        Args:
            model: Pre-trained SB3 model with predict() method
            adapter: Optional adapter for observation building. If None,
                uses internal observation construction.
            k_paths: Expected number of path options (for obs space size)
        """
        self.model = model
        self._adapter = adapter
        self.k_paths = k_paths

        # Validate model has predict method
        if not hasattr(model, "predict"):
            raise ValueError(
                f"Model {type(model).__name__} does not have predict() method"
            )

        logger.info(
            "RLPolicy initialized with %s, k_paths=%d",
            type(model).__name__,
            k_paths,
        )

    def select_action(
        self,
        request: Request,
        options: list[PathOption],
        network_state: NetworkState,
    ) -> int:
        """
        Select action using the trained SB3 model.

        Builds an observation from the inputs, generates an action mask
        from feasibility flags, and uses the model to predict an action.

        Args:
            request: The incoming request to serve
            options: Available path options with feasibility information
            network_state: Current state of the network

        Returns:
            Path index (0 to len(options)-1), or -1 if no valid action

        Note:
            The model must support action masking. For models trained with
            sb3-contrib's MaskablePPO, the action_masks parameter is used.
            For standard models, masking is applied post-prediction.
        """
        # Build observation
        obs = self._build_observation(request, options, network_state)

        # Build action mask
        action_mask = self._build_action_mask(options)

        # Check if any action is valid
        if not any(action_mask):
            logger.debug("No feasible actions available")
            return -1

        try:
            # Try to use native action masking if available
            if self._supports_action_masking():
                action, _ = self.model.predict(
                    obs,
                    deterministic=True,
                    action_masks=np.array(action_mask),
                )
            else:
                # Predict without masking, then validate
                action, _ = self.model.predict(obs, deterministic=True)
                action = int(action)

                # If predicted action is infeasible, find first feasible
                if action >= len(options) or not action_mask[action]:
                    logger.debug(
                        "Model predicted infeasible action %d, "
                        "selecting first feasible",
                        action,
                    )
                    action = self._find_first_feasible(action_mask)

            return int(action) if action >= 0 else -1

        except Exception as e:
            logger.warning("Model prediction failed: %s, returning -1", e)
            return -1

    def _build_observation(
        self,
        request: Request,
        options: list[PathOption],
        network_state: NetworkState,
    ) -> np.ndarray:
        """
        Build observation array for model prediction.

        If an adapter is available, delegates to adapter.build_observation().
        Otherwise, constructs observation matching training format.

        Returns:
            numpy array matching model's observation space
        """
        if self._adapter is not None:
            return self._adapter.build_observation(request, options, network_state)

        # Internal observation construction
        features: list[float] = []

        # Request features
        features.append(request.bandwidth_gbps / 1000.0)  # Normalized

        # Per-path features (padded to k_paths)
        for i in range(self.k_paths):
            if i < len(options):
                opt = options[i]
                features.extend(
                    [
                        opt.weight_km / 10000.0,  # Normalized length
                        opt.congestion,  # Already 0-1
                        1.0 if opt.is_feasible else 0.0,  # Feasibility
                        (opt.slots_needed or 0) / 100.0,  # Normalized slots
                    ]
                )
            else:
                # Padding for missing paths
                features.extend([0.0, 1.0, 0.0, 0.0])

        return np.array(features, dtype=np.float32)

    def _build_action_mask(self, options: list[PathOption]) -> list[bool]:
        """
        Build action mask from path options.

        Returns:
            List of booleans, True where action is valid (is_feasible)
        """
        mask = [opt.is_feasible for opt in options]

        # Pad to k_paths if needed
        while len(mask) < self.k_paths:
            mask.append(False)

        return mask[: self.k_paths]

    def _supports_action_masking(self) -> bool:
        """Check if model supports native action masking."""
        # MaskablePPO and similar algorithms support action_masks parameter
        model_name = type(self.model).__name__
        return model_name in ("MaskablePPO", "MaskableRecurrentPPO")

    def _find_first_feasible(self, mask: list[bool]) -> int:
        """Find index of first feasible action."""
        for i, is_feasible in enumerate(mask):
            if is_feasible:
                return i
        return -1

    def update(self, request: Request, action: int, reward: float) -> None:
        """
        Update policy based on experience.

        RLPolicy wraps pre-trained models that do not learn online.
        This method is a no-op to satisfy the ControlPolicy protocol.

        For online RL training, use UnifiedSimEnv with SB3's learn() method.
        """
        pass

    def get_name(self) -> str:
        """
        Return policy name for logging and metrics.

        Returns:
            String identifying this policy and underlying model
        """
        model_name = type(self.model).__name__
        return f"RLPolicy({model_name})"

    def set_adapter(self, adapter: RLSimulationAdapter) -> None:
        """
        Set the RL simulation adapter for observation building.

        Args:
            adapter: Adapter from Phase 4 RL integration
        """
        self._adapter = adapter

    @classmethod
    def from_file(
        cls,
        model_path: str,
        algorithm: str = "PPO",
        **kwargs: Any,
    ) -> "RLPolicy":
        """
        Load RLPolicy from a saved model file.

        Args:
            model_path: Path to saved model (e.g., "model.zip")
            algorithm: SB3 algorithm name ("PPO", "DQN", "A2C", "MaskablePPO", etc.)
            **kwargs: Additional arguments passed to RLPolicy.__init__

        Returns:
            RLPolicy wrapping the loaded model

        Raises:
            ValueError: If algorithm is unknown/not installed

        Example:
            >>> policy = RLPolicy.from_file("trained_ppo.zip", algorithm="PPO")
        """
        import importlib

        algorithm_class = None

        # Try standard stable_baselines3 first
        try:
            sb3_module = importlib.import_module("stable_baselines3")
            algorithm_class = getattr(sb3_module, algorithm, None)
        except ImportError:
            pass

        # If not found, try sb3_contrib for maskable algorithms
        if algorithm_class is None:
            try:
                sb3_contrib = importlib.import_module("sb3_contrib")
                algorithm_class = getattr(sb3_contrib, algorithm, None)
            except ImportError:
                pass

        if algorithm_class is None:
            raise ValueError(
                f"Unknown algorithm: {algorithm}. "
                "Ensure stable_baselines3 or sb3_contrib is installed."
            )

        model = algorithm_class.load(model_path)
        return cls(model, **kwargs)


# Type alias for backwards compatibility
RLControlPolicy = RLPolicy
