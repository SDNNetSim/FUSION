import os

import torch

from reinforcement_learning.utils.general_utils import determine_model_type
from reinforcement_learning.args.registry_args import ALGORITHM_REGISTRY

from reinforcement_learning.feat_extrs.constants import CACHE_DIR
from reinforcement_learning.feat_extrs.path_gnn_cached import CachedPathGNN
from helper_scripts.sim_helpers import parse_yaml_file

import torch.nn as nn


def _parse_policy_kwargs(s: str) -> dict:
    """
    Turn strings like
        "dict( ortho_init=True, activation_fn=nn.ReLU, net_arch=dict(pi=[64]) )"
    into an actual Python dict.  Only `dict` and `nn` are allowed names.
    """
    safe_globals = {"__builtins__": None, "dict": dict, "nn": nn}
    try:
        return eval(s, safe_globals, {})  # pylint: disable=eval-used
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Bad policy_kwargs string: {s!r}") from exc


def get_model(sim_dict: dict, device: str, env: object, yaml_dict: dict):
    """
    Build/return the SB3 model.
    Adds CachedPathGNN automatically if a cache file exists.
    """
    model_type = determine_model_type(sim_dict)
    algorithm = sim_dict[model_type]

    if yaml_dict is None:
        yml = os.path.join("sb3_scripts", "yml",
                           f"{algorithm}_{sim_dict['network']}.yml")
        yaml_dict = parse_yaml_file(yml)
        env_name = next(iter(yaml_dict))
        param = yaml_dict[env_name]
    else:
        param = yaml_dict
    cache_fp = CACHE_DIR / f"{sim_dict['network']}.pt"
    if os.path.exists(cache_fp):
        cached = torch.load(cache_fp)
        pk_raw = param.get("policy_kwargs", {})
        if isinstance(pk_raw, str):
            pk = _parse_policy_kwargs(pk_raw)
        else:
            pk = pk_raw
        param["policy_kwargs"] = pk
        pk.update(
            features_extractor_class=CachedPathGNN,
            features_extractor_kwargs=dict(cached_embedding=cached)
        )
        param["policy_kwargs"] = pk
        print("✅ Using CachedPathGNN from", cache_fp)

    model = ALGORITHM_REGISTRY[algorithm]["setup"](env=env, device=device)
    return model, param


def get_trained_model(env: object, sim_dict: dict):
    """
    Loads a pre-trained reinforcement learning model from disk or initializes a new one.

    :param env: The reinforcement learning environment.
    :param sim_dict: A dictionary containing simulation-specific parameters, including the model type and path.
    :return: The loaded or newly initialized RL model.
    """
    model_type = determine_model_type(sim_dict=sim_dict)
    algorithm_info = sim_dict.get(model_type)

    if '_' not in algorithm_info:
        raise ValueError(
            f"Algorithm info '{algorithm_info}' must include both algorithm and agent type (e.g., 'ppo_path').")
    algorithm, agent_type = algorithm_info.split('_', 1)

    if algorithm not in ALGORITHM_REGISTRY:
        raise NotImplementedError(f"Algorithm '{algorithm}' is not supported for loading.")

    model_key = f"{agent_type}_model"
    model_path = os.path.join('logs', sim_dict[model_key], f"{algorithm_info}_model.zip")
    model = ALGORITHM_REGISTRY[algorithm]['load'](model_path, env=env)

    return model


def save_model(sim_dict: dict, env: object, model):
    """
    Saves the trained model to the appropriate location based on the algorithm and agent type.

    :param sim_dict: Simulation configuration dictionary.
    :param env: The reinforcement learning environment.
    :param model: The trained model to be saved.
    """
    model_type = determine_model_type(sim_dict=sim_dict)
    if '_' not in model_type:
        raise ValueError(
            f"Algorithm info '{model_type}' must include both algorithm and agent type (e.g., 'ppo_path').")

    algorithm = sim_dict.get(model_type)
    save_fp = os.path.join(
        'logs',
        algorithm,
        env.modified_props['network'],
        env.modified_props['date'],
        env.modified_props['sim_start'],
        f"{algorithm}_model.zip"
    )
    model.save(save_fp)
