"""
Gymnasium environments for FUSION simulation.

This package provides Gymnasium-compatible environment implementations
for reinforcement learning with FUSION network simulations.

Migration Support (P4.3):
    This module provides a factory function for gradual migration from
    the legacy GeneralSimEnv to the new UnifiedSimEnv. Use create_sim_env()
    to create environments with automatic or explicit environment selection.

    Environment Variables:
        USE_UNIFIED_ENV: Set to "1", "true", or "yes" to use UnifiedSimEnv
        RL_ENV_TYPE: Set to "legacy" or "unified" for explicit selection

    Example:
        # Default (legacy)
        env = create_sim_env(config)

        # Explicit unified
        env = create_sim_env(config, env_type="unified")

        # Via environment variable
        os.environ["USE_UNIFIED_ENV"] = "1"
        env = create_sim_env(config)
"""

from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING, Any

import gymnasium as gym

from fusion.modules.rl.gymnasium_envs.constants import (
    ARRIVAL_DICT_KEYS,
    DEFAULT_ARRIVAL_COUNT,
    DEFAULT_ITERATION,
    DEFAULT_SAVE_SIMULATION,
    DEFAULT_SIMULATION_KEY,
    SUPPORTED_SPECTRAL_BANDS,
)
from fusion.modules.rl.gymnasium_envs.general_sim_env import SimEnv

if TYPE_CHECKING:
    from fusion.domain.config import SimulationConfig


class EnvType:
    """Environment type constants for factory function."""

    LEGACY = "legacy"
    UNIFIED = "unified"


def create_sim_env(
    config: dict[str, Any] | SimulationConfig,
    env_type: str | None = None,
    wrap_action_mask: bool = True,
    **kwargs: Any,
) -> gym.Env:
    """Create RL simulation environment.

    This factory function creates either the legacy GeneralSimEnv or
    the new UnifiedSimEnv based on the env_type parameter or environment
    variables.

    Args:
        config: Simulation configuration (dict for legacy, SimulationConfig
            or dict for unified)
        env_type: Environment type to create:
            - "legacy": Use GeneralSimEnv (default)
            - "unified": Use UnifiedSimEnv
            - None: Check env vars, default to legacy
        wrap_action_mask: If True and using unified env, wrap with
            ActionMaskWrapper for SB3 MaskablePPO compatibility
        **kwargs: Additional arguments passed to environment constructor

    Returns:
        Gymnasium environment instance

    Environment Variables:
        USE_UNIFIED_ENV: Set to "1", "true", or "yes" to use UnifiedSimEnv
        RL_ENV_TYPE: Explicit environment type ("legacy" or "unified")

    Priority (highest to lowest):
        1. Explicit env_type parameter
        2. RL_ENV_TYPE environment variable
        3. USE_UNIFIED_ENV environment variable
        4. Default to legacy

    Examples:
        # Legacy (default)
        env = create_sim_env(sim_dict)

        # Unified via parameter
        env = create_sim_env(config, env_type="unified")

        # Unified via environment variable
        os.environ["USE_UNIFIED_ENV"] = "1"
        env = create_sim_env(config)

        # Unified without action mask wrapper
        env = create_sim_env(config, env_type="unified", wrap_action_mask=False)
    """
    resolved_type = _resolve_env_type(env_type)

    if resolved_type == EnvType.UNIFIED:
        return _create_unified_env(config, wrap_action_mask=wrap_action_mask, **kwargs)
    else:
        return _create_legacy_env(config, **kwargs)


def _resolve_env_type(env_type: str | None) -> str:
    """Resolve environment type from parameter or environment variables.

    Priority:
        1. Explicit env_type parameter
        2. RL_ENV_TYPE environment variable
        3. USE_UNIFIED_ENV environment variable
        4. Default to legacy

    Args:
        env_type: Explicit environment type or None

    Returns:
        Resolved environment type (EnvType.LEGACY or EnvType.UNIFIED)
    """
    if env_type is not None:
        return env_type.lower()

    # Check RL_ENV_TYPE
    rl_env_type = os.environ.get("RL_ENV_TYPE", "").lower()
    if rl_env_type in (EnvType.LEGACY, EnvType.UNIFIED):
        return rl_env_type

    # Check USE_UNIFIED_ENV
    use_unified = os.environ.get("USE_UNIFIED_ENV", "").lower()
    if use_unified in ("1", "true", "yes"):
        return EnvType.UNIFIED

    # Default to legacy
    return EnvType.LEGACY


def _create_unified_env(
    config: dict[str, Any] | SimulationConfig,
    wrap_action_mask: bool = True,
    **kwargs: Any,
) -> gym.Env:
    """Create UnifiedSimEnv instance wired to real simulation.

    Args:
        config: Simulation configuration (dict or SimulationConfig)
        wrap_action_mask: Whether to wrap with ActionMaskWrapper
        **kwargs: Additional arguments for UnifiedSimEnv

    Returns:
        UnifiedSimEnv instance wired to real orchestrator and adapter,
        optionally wrapped with ActionMaskWrapper
    """
    from fusion.core.pipeline_factory import PipelineFactory
    from fusion.core.simulation import SimulationEngine
    from fusion.domain.config import SimulationConfig as DomainSimulationConfig
    from fusion.modules.rl.adapter import RLConfig, RLSimulationAdapter
    from fusion.modules.rl.environments import ActionMaskWrapper, UnifiedSimEnv

    # Filter out legacy SimEnv-specific kwargs that UnifiedSimEnv doesn't support
    legacy_only_kwargs = {"custom_callback", "sim_dict", "render_mode"}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k not in legacy_only_kwargs}

    # Handle nested s1 format (legacy RL dict format)
    if isinstance(config, dict):
        inner_config = config.get("s1", config)
    else:
        inner_config = config

    # Add required defaults BEFORE creating any components
    # These are expected by SimStats, SpectrumAdapter, etc.
    if isinstance(inner_config, dict):
        # Ensure band_list is set (required for spectrum operations)
        if "band_list" not in inner_config:
            inner_config["band_list"] = SUPPORTED_SPECTRAL_BANDS

        # Set erlang from erlang_start if not present
        if "erlang" not in inner_config and "erlang_start" in inner_config:
            inner_config["erlang"] = float(inner_config["erlang_start"])

        # Calculate arrival_rate if not present
        if "arrival_rate" not in inner_config:
            cores_per_link = inner_config.get("cores_per_link", 1)
            erlang = inner_config.get("erlang", inner_config.get("erlang_start", 300))
            holding_time = inner_config.get("holding_time", 1)
            inner_config["arrival_rate"] = (cores_per_link * float(erlang)) / float(holding_time)

        # Enable orchestrator mode for V4 stack integration
        inner_config["use_orchestrator"] = True

    # Create SimulationConfig from dict if needed
    if isinstance(inner_config, dict):
        try:
            sim_config = DomainSimulationConfig.from_engine_props(inner_config)
        except (KeyError, ValueError, TypeError) as e:
            # If we can't create a full SimulationConfig, fall back to standalone mode
            warnings.warn(
                f"Could not create SimulationConfig from dict ({e}), "
                "using standalone mode. For full simulation, provide complete config.",
                RuntimeWarning,
                stacklevel=3,
            )
            # Fall back to standalone mode
            rl_config = RLConfig(
                k_paths=inner_config.get("k_paths", 3),
                rl_success_reward=inner_config.get("reward", inner_config.get("rl_success_reward", 1.0)),
                rl_block_penalty=inner_config.get("penalty", inner_config.get("rl_block_penalty", -1.0)),
                num_nodes=inner_config.get("num_nodes", 14),
                total_slots=inner_config.get("spectral_slots", inner_config.get("total_slots", 320)),
            )
            env = UnifiedSimEnv(config=rl_config, **filtered_kwargs)
            if wrap_action_mask:
                env = ActionMaskWrapper(env)
            return env
    else:
        sim_config = inner_config

    # Create SimulationEngine for stats tracking
    # The engine requires the full engine_props dict with topology_info, mod_per_bw, etc.
    # These should be populated by create_input() before calling create_sim_env()
    engine: SimulationEngine | None = None
    if isinstance(inner_config, dict):
        try:
            engine = SimulationEngine(engine_props=inner_config)
            # Initialize the engine's topology and orchestrator
            # This creates engine._orchestrator and engine._network_state
            engine.create_topology()
        except (KeyError, ValueError, TypeError) as e:
            warnings.warn(
                f"Could not create SimulationEngine ({e}), "
                "stats tracking will be disabled.",
                RuntimeWarning,
                stacklevel=3,
            )

    # Use the engine's orchestrator (if available) to ensure stats tracking works
    # The engine creates its own orchestrator when use_orchestrator=True
    if engine is not None and hasattr(engine, "_orchestrator") and engine._orchestrator is not None:
        orchestrator = engine._orchestrator
    else:
        # Fallback: create orchestrator separately (stats tracking won't work)
        orchestrator = PipelineFactory.create_orchestrator(sim_config)

    # Create RLConfig from SimulationConfig
    rl_config = RLConfig(
        k_paths=sim_config.k_paths,
        rl_success_reward=getattr(sim_config, "rl_success_reward", 1.0),
        rl_block_penalty=getattr(sim_config, "rl_block_penalty", -1.0),
        num_nodes=getattr(sim_config, "num_nodes", 14),
        total_slots=sum(sim_config.band_slots.values()) if sim_config.band_slots else 320,
    )

    # Create RLSimulationAdapter using the engine's orchestrator
    adapter = RLSimulationAdapter(orchestrator=orchestrator, config=rl_config)

    # Create RL agent infrastructure for non-DRL algorithms (bandits, Q-learning)
    path_agent = None
    rl_props = None
    path_algorithm = inner_config.get("path_algorithm", "") if isinstance(inner_config, dict) else ""
    is_training = inner_config.get("is_training", False) if isinstance(inner_config, dict) else False

    if path_algorithm and "bandit" in path_algorithm or path_algorithm == "q_learning":
        # Create RLProps and PathAgent like legacy SimEnv does
        from fusion.modules.rl.algorithms.algorithm_props import RLProps
        from fusion.modules.rl.agents.path_agent import PathAgent

        rl_props = RLProps()
        rl_props.k_paths = inner_config.get("k_paths", 3) if isinstance(inner_config, dict) else 3
        rl_props.num_nodes = rl_config.num_nodes
        rl_props.cores_per_link = inner_config.get("cores_per_link", 1) if isinstance(inner_config, dict) else 1
        rl_props.spectral_slots = inner_config.get("c_band", 320) if isinstance(inner_config, dict) else 320

        path_agent = PathAgent(
            path_algorithm=path_algorithm,
            rl_props=rl_props,
            rl_help_obj=None,  # Not needed for bandit path selection
        )

        # Set engine_props and setup the algorithm
        if engine is not None:
            path_agent.engine_props = engine.engine_props
            if is_training:
                path_agent.setup_env(is_path=True)

    # Create UnifiedSimEnv with engine for stats tracking
    env = UnifiedSimEnv(
        config=rl_config,
        engine=engine,
        orchestrator=orchestrator,
        adapter=adapter,
        path_agent=path_agent,
        rl_props=rl_props,
        path_algorithm=path_algorithm,
        is_training=is_training,
        **filtered_kwargs,
    )

    if wrap_action_mask:
        env = ActionMaskWrapper(env)

    return env


def _create_legacy_env(
    config: dict[str, Any] | SimulationConfig,
    **kwargs: Any,
) -> gym.Env:
    """Create legacy GeneralSimEnv instance.

    Args:
        config: Simulation configuration
        **kwargs: Additional arguments for SimEnv

    Returns:
        SimEnv (GeneralSimEnv) instance
    """
    # Convert SimulationConfig to dict if needed
    if hasattr(config, "to_dict"):
        sim_dict = config.to_dict()
    elif hasattr(config, "to_legacy_dict"):
        sim_dict = config.to_legacy_dict()
    elif isinstance(config, dict):
        sim_dict = config
    else:
        raise TypeError(f"Unsupported config type: {type(config)}")

    # Wrap in "s1" key if needed (legacy format requirement)
    if "s1" not in sim_dict:
        sim_dict = {"s1": sim_dict}

    return SimEnv(sim_dict=sim_dict, **kwargs)


__all__ = [
    # Factory function and type
    "create_sim_env",
    "EnvType",
    # Main environment class (backward compatibility)
    "SimEnv",
    # Constants
    "DEFAULT_SIMULATION_KEY",
    "DEFAULT_SAVE_SIMULATION",
    "SUPPORTED_SPECTRAL_BANDS",
    "ARRIVAL_DICT_KEYS",
    "DEFAULT_ITERATION",
    "DEFAULT_ARRIVAL_COUNT",
]
