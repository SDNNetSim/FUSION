import os

import torch
import torch.nn as nn  # pylint: disable=consider-using-from-import

from fusion.modules.rl.args.registry_args import ALGORITHM_REGISTRY
from fusion.modules.rl.errors import (
    AlgorithmNotFoundError,
    ModelLoadError,
    RLConfigurationError,
)
from fusion.modules.rl.feat_extrs.constants import CACHE_DIR
from fusion.modules.rl.feat_extrs.path_gnn_cached import CachedPathGNN
from fusion.modules.rl.utils.general_utils import determine_model_type
from fusion.sim.utils import parse_yaml_file
from fusion.utils.logging_config import get_logger

logger = get_logger(__name__)


def _parse_policy_kwargs(string: str) -> dict:
    """
    Turn strings like
        "dict( ortho_init=True, activation_fn=nn.ReLU, net_arch=dict(pi=[64]) )"
    into an actual Python dict.  Only `dict` and `nn` are allowed names.
    """
    safe_globals = {"__builtins__": None, "dict": dict, "nn": nn}
    try:
        return eval(string, safe_globals, {})  # pylint: disable=eval-used
    except (SyntaxError, NameError, TypeError) as exc:
        raise RLConfigurationError(
            f"Failed to parse policy_kwargs string: {string!r}. "
            f"Ensure it contains valid Python syntax with only 'dict' and 'nn' references."
        ) from exc


def get_model(sim_dict: dict, device: str, env: object, yaml_dict: dict):
    """
    Build/return the SB3 model.
    Adds CachedPathGNN automatically if a cache file exists.
    """
    model_type = determine_model_type(sim_dict)
    algorithm = sim_dict[model_type]

    # TODO: Ensure this is consistent acoss the board (other cli, files, etc.)
    #   We might want to find a better way to do this
    if yaml_dict is None:
        logger.debug("Current working directory: %s", os.getcwd())

        yml = os.path.join(
            "fusion", "configs", "hyperparams", f"{algorithm}_{sim_dict['network']}.yml"
        )
        yaml_dict = parse_yaml_file(yml)
        env_name = next(iter(yaml_dict))
        parameters = yaml_dict[env_name]
    else:
        parameters = yaml_dict
    cache_file_path = CACHE_DIR / f"{sim_dict['network']}.pt"
    if os.path.exists(cache_file_path):
        cached = torch.load(cache_file_path)
        policy_kwargs_raw = parameters.get("policy_kwargs", {})
        if isinstance(policy_kwargs_raw, str):
            policy_kwargs = _parse_policy_kwargs(policy_kwargs_raw)
        else:
            policy_kwargs = policy_kwargs_raw
        parameters["policy_kwargs"] = policy_kwargs
        policy_kwargs.update(
            features_extractor_class=CachedPathGNN,
            features_extractor_kwargs=dict(cached_embedding=cached),
        )
        parameters["policy_kwargs"] = policy_kwargs
        logger.info("Using CachedPathGNN from %s", cache_file_path)

    model = ALGORITHM_REGISTRY[algorithm]["setup"](env=env, device=device)
    return model, parameters


def get_trained_model(env: object, sim_dict: dict):
    """
    Loads a pre-trained reinforcement learning model from disk or initializes a new one.

    :param env: The reinforcement learning environment.
    :param sim_dict: A dictionary containing simulation-specific parameters, including the model type and path.
    :return: The loaded or newly initialized RL model.
    """
    model_type = determine_model_type(sim_dict=sim_dict)
    algorithm_info = sim_dict.get(model_type)

    if "_" not in algorithm_info:
        raise RLConfigurationError(
            f"Algorithm info '{algorithm_info}' must include both algorithm and agent type (e.g., 'ppo_path')."
        )
    algorithm, agent_type = algorithm_info.split("_", 1)

    if algorithm not in ALGORITHM_REGISTRY:
        raise AlgorithmNotFoundError(
            f"Algorithm '{algorithm}' is not registered. "
            f"Available algorithms: {list(ALGORITHM_REGISTRY.keys())}"
        )

    model_key = f"{agent_type}_model"
    model_path = os.path.join(
        "logs", sim_dict[model_key], f"{algorithm_info}_model.zip"
    )

    if not os.path.exists(model_path):
        raise ModelLoadError(
            f"Model file not found at '{model_path}'. "
            f"Please ensure the model has been trained and saved."
        )

    try:
        model = ALGORITHM_REGISTRY[algorithm]["load"](model_path, env=env)
    except Exception as exc:
        raise ModelLoadError(
            f"Failed to load model from '{model_path}': {exc}"
        ) from exc

    return model


def save_model(sim_dict: dict, env: object, model):
    """
    Saves the trained model to the appropriate location based on the algorithm and agent type.

    :param sim_dict: Simulation configuration dictionary.
    :param env: The reinforcement learning environment.
    :param model: The trained model to be saved.
    """
    model_type = determine_model_type(sim_dict=sim_dict)
    if "_" not in model_type:
        raise RLConfigurationError(
            f"Algorithm info '{model_type}' must include both algorithm and agent type (e.g., 'ppo_path')."
        )

    algorithm = sim_dict.get(model_type)
    save_fp = os.path.join(
        "logs",
        algorithm,
        env.modified_props["network"],
        env.modified_props["date"],
        env.modified_props["sim_start"],
        f"{algorithm}_model.zip",
    )
    model.save(save_fp)
