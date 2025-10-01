"""
RL Utils Module.

Provides utility functions and classes for reinforcement learning components
in the FUSION network simulation framework, including simulation environment
helpers, data loading utilities, and configuration management.
"""

from fusion.modules.rl.utils.callbacks import (
    EpisodicRewardCallback,
    GetModelParams,
    LearnRateEntCallback,
)
from fusion.modules.rl.utils.deep_rl import (
    get_action_space,
    get_algorithm_instance,
    get_obs_space,
)
from fusion.modules.rl.utils.errors import (
    CacheError,
    ConfigurationError,
    DataLoadingError,
    FeatureExtractorError,
    HyperparameterError,
    ModelSetupError,
    RLUtilsError,
    SimulationDataError,
)
from fusion.modules.rl.utils.general_utils import (
    CoreUtilHelpers,
    determine_model_type,
    save_arr,
)
# Removed create_environment import to avoid circular dependency with gymnasium_envs
# Import directly: from fusion.modules.rl.utils.gym_envs import create_environment
from fusion.modules.rl.utils.hyperparams import HyperparamConfig, get_optuna_hyperparams
from fusion.modules.rl.utils.observation_space import (
    FragmentationTracker,
    get_observation_space,
)
from fusion.modules.rl.utils.setup import (
    SetupHelper,
    setup_a2c,
    setup_dqn,
    setup_feature_extractor,
    setup_ppo,
    setup_qr_dqn,
    setup_rl_sim,
)
from fusion.modules.rl.utils.sim_env import SimEnvObs, SimEnvUtils
from fusion.modules.rl.utils.topology import (
    convert_networkx_topo,
    load_topology_from_graph,
)

__all__ = [
    # Callbacks
    "EpisodicRewardCallback",
    "GetModelParams",
    "LearnRateEntCallback",
    # Deep RL utilities
    "get_action_space",
    "get_algorithm_instance",
    "get_obs_space",
    # Error classes
    "CacheError",
    "ConfigurationError",
    "DataLoadingError",
    "FeatureExtractorError",
    "HyperparameterError",
    "ModelSetupError",
    "RLUtilsError",
    "SimulationDataError",
    # General utilities
    "CoreUtilHelpers",
    "determine_model_type",
    "save_arr",
    # Environment creation - import directly from gym_envs to avoid circular import
    # "create_environment",
    # Hyperparameters
    "HyperparamConfig",
    "get_optuna_hyperparams",
    # Observation space
    "FragmentationTracker",
    "get_observation_space",
    # Setup utilities
    "SetupHelper",
    "setup_a2c",
    "setup_dqn",
    "setup_feature_extractor",
    "setup_ppo",
    "setup_qr_dqn",
    "setup_rl_sim",
    # Simulation environment
    "SimEnvObs",
    "SimEnvUtils",
    # Topology utilities
    "convert_networkx_topo",
    "load_topology_from_graph",
]
