"""Model management utilities for reinforcement learning.

This module provides functions for creating, loading, and saving RL models,
including integration with cached feature extractors and hyperparameter
configuration.
"""

import os
from typing import Any

import torch
import torch.nn as nn

from fusion.modules.rl.args.registry_args import get_algorithm_registry
from fusion.modules.rl.errors import (
    AlgorithmNotFoundError,
    ModelLoadError,
    RLConfigurationError,
)
from fusion.modules.rl.feat_extrs.constants import CACHE_DIR
from fusion.modules.rl.feat_extrs.path_gnn_cached import CachedPathGNN
from fusion.modules.rl.utils.general_utils import determine_model_type
from fusion.sim.utils.io import parse_yaml_file
from fusion.utils.logging_config import get_logger

logger = get_logger(__name__)


def _parse_policy_kwargs(string: str) -> dict[str, Any]:
    """
    Parse policy kwargs from string representation to dictionary.

    Converts strings like "dict(ortho_init=True, activation_fn=nn.ReLU,
    net_arch=dict(pi=[64]))" into actual Python dictionaries. Only 'dict'
    and 'nn' references are allowed for security.

    :param string: String representation of policy kwargs
    :type string: str
    :return: Parsed policy kwargs dictionary
    :rtype: Dict[str, Any]
    :raises RLConfigurationError: If parsing fails due to invalid syntax
    """
    safe_globals = {"__builtins__": None, "dict": dict, "nn": nn}
    try:
        # eval is safe here with restricted globals - only dict and nn allowed
        result = eval(string, safe_globals, {})  # nosec B307 # pylint: disable=eval-used
        if not isinstance(result, dict):
            raise RLConfigurationError(f"Expected dict, got {type(result).__name__}")
        return result
    except (SyntaxError, NameError, TypeError) as exc:
        raise RLConfigurationError(
            f"Failed to parse policy_kwargs string: {string!r}. "
            "Ensure it contains valid Python syntax with only "
            "'dict' and 'nn' references."
        ) from exc


def get_model(sim_dict: dict[str, Any], device: str, env: Any, yaml_dict: dict[str, Any] | None) -> tuple[Any, dict[str, Any]]:
    """
    Build and return a Stable-Baselines3 model with configuration.

    Automatically integrates CachedPathGNN feature extractor if a cache file
    exists for the specified network. Loads hyperparameters from YAML
    configuration files.

    :param sim_dict: Simulation configuration dictionary
    :type sim_dict: Dict[str, Any]
    :param device: Device for model computation (e.g., 'cpu', 'cuda')
    :type device: str
    :param env: The reinforcement learning environment
    :type env: Any
    :param yaml_dict: Optional pre-loaded YAML configuration
    :type yaml_dict: Dict[str, Any] | None
    :return: Tuple of (model, parameters)
    :rtype: Tuple[Any, Dict[str, Any]]
    """
    model_type = determine_model_type(sim_dict)
    algorithm = sim_dict[model_type]

    # TODO(v6.1): Standardize hyperparameter file discovery across CLI and modules.
    #   Current approach hardcodes path to fusion/configs/hyperparams/{algorithm}_{network}.yml.
    #   Should use a centralized config lookup or registry pattern instead.
    if yaml_dict is None:
        logger.debug("Current working directory: %s", os.getcwd())

        yml = os.path.join("fusion", "configs", "hyperparams", f"{algorithm}_{sim_dict['network']}.yml")
        yaml_dict = parse_yaml_file(yml)
        env_name = next(iter(yaml_dict))
        parameters = yaml_dict[env_name]
    else:
        parameters = yaml_dict
    cache_file_path = CACHE_DIR / f"{sim_dict['network']}.pt"
    if os.path.exists(cache_file_path):
        # Loading trusted model checkpoint from local cache
        cached = torch.load(cache_file_path)  # nosec B614
        policy_kwargs_raw = parameters.get("policy_kwargs", {})
        if isinstance(policy_kwargs_raw, str):
            policy_kwargs = _parse_policy_kwargs(policy_kwargs_raw)
        else:
            policy_kwargs = policy_kwargs_raw
        parameters["policy_kwargs"] = policy_kwargs
        policy_kwargs.update(
            features_extractor_class=CachedPathGNN,
            features_extractor_kwargs={"cached_embedding": cached},
        )
        parameters["policy_kwargs"] = policy_kwargs
        logger.info("Using CachedPathGNN from %s", cache_file_path)

    algorithm_registry = get_algorithm_registry()
    setup_func = algorithm_registry[algorithm]["setup"]
    if setup_func is None:
        raise RLConfigurationError(f"Setup function not implemented for algorithm '{algorithm}'")
    model = setup_func(env=env, device=device)
    return model, parameters


def get_trained_model(env: Any, sim_dict: dict[str, Any]) -> Any:
    """
    Load a pre-trained reinforcement learning model from disk.

    Loads a previously trained model based on the algorithm and agent type
    specified in the simulation dictionary. The model file must exist at
    the expected path.

    :param env: The reinforcement learning environment
    :type env: Any
    :param sim_dict: Simulation configuration dictionary containing model info
    :type sim_dict: Dict[str, Any]
    :return: The loaded RL model
    :rtype: Any
    :raises RLConfigurationError: If algorithm info format is invalid
    :raises AlgorithmNotFoundError: If algorithm is not registered
    :raises ModelLoadError: If model file doesn't exist or loading fails
    """
    model_type = determine_model_type(sim_dict=sim_dict)
    algorithm_info = sim_dict.get(model_type)

    if algorithm_info is None:
        raise RLConfigurationError(f"No algorithm info found for model type '{model_type}'")

    if not isinstance(algorithm_info, str):
        raise RLConfigurationError(f"Algorithm info must be a string, got {type(algorithm_info).__name__}")

    if "_" not in algorithm_info:
        raise RLConfigurationError(f"Algorithm info '{algorithm_info}' must include both algorithm and agent type (e.g., 'ppo_path').")
    algorithm, agent_type = algorithm_info.split("_", 1)

    algorithm_registry = get_algorithm_registry()
    if algorithm not in algorithm_registry:
        raise AlgorithmNotFoundError(f"Algorithm '{algorithm}' is not registered. Available algorithms: {list(algorithm_registry.keys())}")

    model_key = f"{agent_type}_model"
    model_path = os.path.join("logs", sim_dict[model_key], f"{algorithm_info}_model.zip")

    if not os.path.exists(model_path):
        raise ModelLoadError(f"Model file not found at '{model_path}'. Please ensure the model has been trained and saved.")

    try:
        load_func = algorithm_registry[algorithm]["load"]
        if load_func is None:
            raise ModelLoadError(f"Model loading not implemented for algorithm '{algorithm}'")
        model = load_func(model_path, env=env)
    except Exception as exc:
        raise ModelLoadError(f"Failed to load model from '{model_path}': {exc}") from exc

    return model


def save_model(sim_dict: dict[str, Any], env: Any, model: Any) -> None:
    """
    Save a trained model to the appropriate directory structure.

    Saves the model to a hierarchical directory structure based on the
    algorithm, network, date, and simulation start time. Creates the
    necessary directories if they don't exist.

    :param sim_dict: Simulation configuration dictionary
    :type sim_dict: Dict[str, Any]
    :param env: The reinforcement learning environment
    :type env: Any
    :param model: The trained model to save
    :type model: Any
    :raises RLConfigurationError: If algorithm info format is invalid
    """
    model_type = determine_model_type(sim_dict=sim_dict)

    algorithm = sim_dict.get(model_type)
    if algorithm is None:
        raise RLConfigurationError(f"No algorithm found for model type '{model_type}'")
    if not isinstance(algorithm, str):
        raise RLConfigurationError(f"Algorithm must be a string, got {type(algorithm).__name__}")
    if "_" not in algorithm:
        raise RLConfigurationError(f"Algorithm info '{algorithm}' must include both algorithm and agent type (e.g., 'ppo_path').")

    save_fp = os.path.join(
        "logs",
        algorithm,
        env.modified_props["network"],
        env.modified_props["date"],
        env.modified_props["sim_start"],
        f"{algorithm}_model.zip",
    )
    model.save(save_fp)
