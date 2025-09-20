# NOTE: The 'device' parameter is maintained for API compatibility but may be deprecated
# in future versions as device selection is handled through YAML configuration
# pylint: disable=unused-argument

import ast
import copy
import os

import torch
from sb3_contrib import QRDQN
from stable_baselines3 import A2C, DQN, PPO

from fusion.cli.config_setup import ConfigManager
from fusion.cli.main_parser import get_train_args
from fusion.core.routing import Routing
from fusion.core.simulation import SimulationEngine
from fusion.modules.rl.args.general_args import (VALID_CORE_ALGORITHMS,
                                                 VALID_PATH_ALGORITHMS)
from fusion.modules.rl.feat_extrs.constants import CACHE_DIR
from fusion.modules.rl.feat_extrs.graphormer import GraphTransformerExtractor
from fusion.modules.rl.feat_extrs.path_gnn_cached import (CachedPathGNN,
                                                          PathGNNEncoder)
from fusion.modules.rl.utils.errors import (ConfigurationError,
                                            FeatureExtractorError,
                                            ModelSetupError)
from fusion.sim.input_setup import create_input, save_input
from fusion.sim.utils import get_start_time, parse_yaml_file


def setup_feature_extractor(env: object):
    """
    Sets up a custom feature extractor.
    """
    engine_props = env.engine_obj.engine_props
    feat_extr = engine_props["feature_extractor"]
    feat_kwargs = {
        "emb_dim": engine_props["emb_dim"],
        "layers": engine_props["layers"],
    }

    if feat_extr == "path_gnn":
        # where the cache will live
        network = env.engine_obj.engine_props["network"]
        cache_fp = CACHE_DIR / f"{engine_props['network']}.pt"

        if os.path.exists(cache_fp):  # ✔ cache already there
            cached = torch.load(cache_fp)
            extr_class = CachedPathGNN
            feat_kwargs = {"cached_embedding": cached}
            print(f"✅  Using cached GNN embedding from {cache_fp}")
        else:  # ✘ no cache → make one now
            print(f"⏳  Caching GNN embedding for {network} …")
            obs = env.reset()[0]
            enc = PathGNNEncoder(
                env.observation_space,
                emb_dim=env.engine_obj.engine_props["emb_dim"],
                gnn_type=env.engine_obj.engine_props["gnn_type"],
                layers=env.engine_obj.engine_props["layers"],
            ).to(env.engine_obj.engine_props["device"])
            enc.eval()
            with torch.no_grad():
                emb = enc(  # pylint: disable=not-callable
                    obs["x"].to(enc.device),
                    obs["edge_index"].long().to(enc.device),
                    obs["path_masks"].to(enc.device),
                ).cpu()
            os.makedirs("gnn_cached", exist_ok=True)
            torch.save(emb, cache_fp)
            print(f"✅  Saved cache to {cache_fp}")

            extr_class = CachedPathGNN
            feat_kwargs = {"cached_embedding": emb}
    elif feat_extr == "graphormer":
        extr_class = GraphTransformerExtractor
        feat_kwargs["heads"] = engine_props["heads"]
    else:
        raise FeatureExtractorError(
            f"Unsupported feature extractor type: '{feat_extr}'. "
            f"Supported types are: 'path_gnn', 'graphormer'. "
            f"Please check your configuration and ensure a valid feature extractor is specified."
        )

    return extr_class, feat_kwargs


def get_drl_dicts(env, yaml_path):
    """
    Gets dictionaries related to DRL algorithms.
    """
    yaml_dict = parse_yaml_file(yaml_path)
    env_name = list(yaml_dict.keys())[0]

    # Safely parse policy_kwargs using ast.literal_eval instead of eval()
    policy_kwargs_str = yaml_dict[env_name]["policy_kwargs"]
    try:
        kwargs_dict = ast.literal_eval(policy_kwargs_str)
    except (ValueError, SyntaxError) as e:
        raise ConfigurationError(
            f"Failed to parse policy_kwargs from YAML file '{yaml_path}'. "
            f"Expected a valid Python literal expression, got: {policy_kwargs_str}. "
            f"Error: {e}"
        ) from e

    if "graph" in env.engine_obj.engine_props["obs_space"]:
        (
            kwargs_dict["features_extractor_class"],
            kwargs_dict["features_extractor_kwargs"],
        ) = setup_feature_extractor(env=env)

    return yaml_dict, kwargs_dict, env_name


def setup_rl_sim(config_path: str = None):
    """
    Loads config and args using the new CLI-based ConfigManager.
    """
    args = get_train_args()
    if config_path:
        args.config_path = config_path  # override if manually passed
    config = ConfigManager.from_args(args)
    return config.get()  # returns sim_dict['s1']


def _get_common_model_parameters(
    yaml_dict: dict, env_name: str, kwargs_dict: dict, env: object
):
    """
    Extract common parameters used across different model types.

    :param yaml_dict: Configuration dictionary from YAML file
    :param env_name: Environment name key
    :param kwargs_dict: Policy kwargs dictionary
    :param env: Custom environment
    :return: Dictionary of common parameters
    :rtype: dict
    """
    return {
        "policy": yaml_dict[env_name]["policy"],
        "env": env,
        "learning_rate": yaml_dict[env_name]["learning_rate"],
        "gamma": yaml_dict[env_name]["gamma"],
        "policy_kwargs": kwargs_dict,
        "verbose": yaml_dict[env_name].get("verbose"),
        "device": yaml_dict[env_name].get("device", "cpu"),
        "_init_setup_model": yaml_dict[env_name].get("_init_setup_model"),
    }


def setup_ppo(env: object, device: str):
    """
    Setups up the StableBaselines3 PPO model.

    :param env: Custom environment created.
    :param device: Device to use, cpu or gpu.
    :return: A PPO model.
    :rtype: object
    """
    network = env.engine_obj.engine_props["network"]
    yaml_path = os.path.join("sb3_scripts", "yml", f"ppo_{network}.yml")
    yaml_dict, kwargs_dict, env_name = get_drl_dicts(env=env, yaml_path=yaml_path)

    # Get common parameters
    common_params = _get_common_model_parameters(yaml_dict, env_name, kwargs_dict, env)

    # Add PPO-specific parameters
    ppo_params = {
        "n_steps": yaml_dict[env_name]["n_steps"],
        "batch_size": yaml_dict[env_name]["batch_size"],
        "n_epochs": yaml_dict[env_name]["n_epochs"],
        "gae_lambda": yaml_dict[env_name]["gae_lambda"],
        "clip_range": yaml_dict[env_name]["clip_range"],
        "clip_range_vf": yaml_dict[env_name].get("clip_range_vf"),
        "normalize_advantage": yaml_dict[env_name].get("normalize_advantage"),
        "ent_coef": yaml_dict[env_name]["ent_coef"],
        "vf_coef": yaml_dict[env_name]["vf_coef"],
        "max_grad_norm": yaml_dict[env_name]["max_grad_norm"],
        "use_sde": yaml_dict[env_name].get("use_sde", False),
        "sde_sample_freq": yaml_dict[env_name].get("sde_sample_freq"),
        "rollout_buffer_class": yaml_dict[env_name].get("rollout_buffer_class"),
        "rollout_buffer_kwargs": yaml_dict[env_name].get("rollout_buffer_kwargs"),
        "target_kl": yaml_dict[env_name].get("target_kl"),
        "stats_window_size": yaml_dict[env_name].get("stats_window_size"),
        "tensorboard_log": yaml_dict[env_name].get("tensorboard_log"),
    }

    # Combine parameters and create model
    all_params = {**common_params, **ppo_params}
    return PPO(**all_params)


def setup_a2c(env: object, device: str):
    """
    Setups up the StableBaselines3 A2C model.

    :param env: Custom environment created.
    :param device: Device to use, cpu or gpu.
    :return: An A2C model.
    :rtype: object
    """
    network = env.engine_obj.engine_props["network"]
    yaml_path = os.path.join("sb3_scripts", "yml", f"a2c_{network}.yml")
    yaml_dict, kwargs_dict, env_name = get_drl_dicts(env=env, yaml_path=yaml_path)

    # Get common parameters
    common_params = _get_common_model_parameters(yaml_dict, env_name, kwargs_dict, env)

    # Add A2C-specific parameters
    a2c_params = {
        "n_steps": yaml_dict[env_name]["n_steps"],
        "gae_lambda": yaml_dict[env_name]["gae_lambda"],
        "ent_coef": yaml_dict[env_name]["ent_coef"],
        "vf_coef": yaml_dict[env_name]["vf_coef"],
        "max_grad_norm": yaml_dict[env_name]["max_grad_norm"],
        "rms_prop_eps": yaml_dict[env_name].get("rms_prop_eps"),
        "use_rms_prop": yaml_dict[env_name].get("use_rms_prop"),
        "use_sde": yaml_dict[env_name]["use_sde"],
        "sde_sample_freq": yaml_dict[env_name]["sde_sample_freq"],
        "rollout_buffer_class": yaml_dict[env_name].get("rollout_buffer_class"),
        "rollout_buffer_kwargs": yaml_dict[env_name].get("rollout_buffer_kwargs"),
        "stats_window_size": yaml_dict[env_name]["stats_window_size"],
        "tensorboard_log": yaml_dict[env_name]["tensorboard_log"],
    }

    # Override default _init_setup_model for A2C
    common_params["_init_setup_model"] = yaml_dict[env_name].get(
        "_init_setup_model", True
    )

    # Combine parameters and create model
    all_params = {**common_params, **a2c_params}
    return A2C(**all_params)


def setup_dqn(env: object, device: str):
    """
    Sets up the StableBaselines3 DQN model.

    :param env: Custom environment created.
    :param device: Device to use, cpu or gpu.
    :return: A DQN model.
    :rtype: object
    """
    network = env.engine_obj.engine_props["network"]
    # NOTE: YAML path handling should be standardized across all DRL algorithms
    # Currently different algorithms use different path structures for configuration files
    yaml_path = os.path.join("fusion", "configs", "hyperparams", f"dqn_{network}.yml")
    yaml_dict, kwargs_dict, env_name = get_drl_dicts(env=env, yaml_path=yaml_path)

    # Get common parameters (but reorder env parameter for DQN)
    common_params = _get_common_model_parameters(yaml_dict, env_name, kwargs_dict, env)

    # Add DQN-specific parameters
    dqn_params = {
        "buffer_size": yaml_dict[env_name]["buffer_size"],
        "learning_starts": yaml_dict[env_name]["learning_starts"],
        "batch_size": yaml_dict[env_name]["batch_size"],
        "tau": yaml_dict[env_name].get("tau"),
        "train_freq": yaml_dict[env_name]["train_freq"],
        "gradient_steps": yaml_dict[env_name]["gradient_steps"],
        "target_update_interval": yaml_dict[env_name]["target_update_interval"],
        "exploration_initial_eps": yaml_dict[env_name].get(
            "exploration_initial_eps", 1.0
        ),
        "exploration_fraction": yaml_dict[env_name]["exploration_fraction"],
        "exploration_final_eps": yaml_dict[env_name]["exploration_final_eps"],
        "max_grad_norm": yaml_dict[env_name].get("max_grad_norm"),
        "replay_buffer_class": yaml_dict[env_name].get("replay_buffer_class", None),
        "replay_buffer_kwargs": yaml_dict[env_name].get("replay_buffer_kwargs", None),
        "optimize_memory_usage": yaml_dict[env_name].get(
            "optimize_memory_usage", False
        ),
    }

    # Override defaults for DQN
    common_params["verbose"] = yaml_dict[env_name].get("verbose", 1)
    common_params["_init_setup_model"] = yaml_dict[env_name].get(
        "_init_setup_model", True
    )

    # Combine parameters and create model
    all_params = {**common_params, **dqn_params}
    return DQN(**all_params)


def setup_qr_dqn(env: object, device: str):
    """
    Sets up the SB3-Contrib QRDQN model (distributional DQN with dueling support).

    :param env: Custom environment created.
    :param device: Device to use, cpu or gpu.
    :return: A QRDQN model.
    :rtype: object
    """
    network = env.engine_obj.engine_props["network"]
    yaml_path = os.path.join("sb3_scripts", "yml", f"qr_dqn_{network}.yml")
    yaml_dict, kwargs_dict, env_name = get_drl_dicts(env=env, yaml_path=yaml_path)

    model = QRDQN(
        env=env,
        device=yaml_dict[env_name].get("device", "cpu"),
        policy=yaml_dict[env_name]["policy"],
        learning_rate=yaml_dict[env_name]["learning_rate"],
        buffer_size=yaml_dict[env_name]["buffer_size"],
        learning_starts=yaml_dict[env_name]["learning_starts"],
        batch_size=yaml_dict[env_name]["batch_size"],
        tau=yaml_dict[env_name].get("tau"),
        gamma=yaml_dict[env_name]["gamma"],
        train_freq=yaml_dict[env_name]["train_freq"],
        gradient_steps=yaml_dict[env_name]["gradient_steps"],
        target_update_interval=yaml_dict[env_name]["target_update_interval"],
        exploration_initial_eps=yaml_dict[env_name].get("exploration_initial_eps", 1.0),
        exploration_fraction=yaml_dict[env_name]["exploration_fraction"],
        exploration_final_eps=yaml_dict[env_name]["exploration_final_eps"],
        max_grad_norm=yaml_dict[env_name].get("max_grad_norm"),
        replay_buffer_class=yaml_dict[env_name].get("replay_buffer_class", None),
        replay_buffer_kwargs=yaml_dict[env_name].get("replay_buffer_kwargs", None),
        optimize_memory_usage=yaml_dict[env_name].get("optimize_memory_usage", False),
        policy_kwargs=kwargs_dict,
        verbose=yaml_dict[env_name].get("verbose", 1),
        _init_setup_model=yaml_dict[env_name].get("_init_setup_model", True),
    )

    return model


def print_info(sim_dict: dict):
    """
    Prints relevant RL simulation information.

    :param sim_dict: Simulation dictionary (engine props).
    """
    if sim_dict["path_algorithm"] in VALID_PATH_ALGORITHMS:
        print(
            f"Beginning training process for the PATH AGENT using the "
            f"{sim_dict['path_algorithm'].title()} algorithm."
        )
    elif sim_dict["core_algorithm"] in VALID_CORE_ALGORITHMS:
        print(
            f"Beginning training process for the CORE AGENT using the "
            f"{sim_dict['core_algorithm'].title()} algorithm."
        )
    elif sim_dict["spectrum_algorithm"]:
        raise ModelSetupError(
            "Spectrum algorithm setup is not yet implemented. "
            "This feature requires additional development for spectrum-based RL agents."
        )
    else:
        raise ModelSetupError(
            f"Invalid algorithm configuration or non-RL algorithms detected. "
            f"Expected RL algorithms (q_learning, dqn, ppo, a2c), but got: "
            f"path_algorithm={sim_dict.get('path_algorithm')}, "
            f"core_algorithm={sim_dict.get('core_algorithm')}, "
            f"spectrum_algorithm={sim_dict.get('spectrum_algorithm')}. "
            f"Please verify your configuration contains valid RL algorithm settings."
        )


class SetupHelper:
    """
    A helper class to handle setup-related tasks for the SimEnv environment.
    """

    def __init__(self, sim_env: object):
        """
        Constructor for RLSetupHelper.

        :param sim_env: Reference to the parent SimEnv instance, to update relevant attributes directly.
        """
        self.sim_env = sim_env

    def create_input(self):
        """
        Creates input for RL agents based on the simulation configuration.
        """
        base_fp = os.path.join("data")
        self.sim_env.sim_dict["thread_num"] = "s1"

        get_start_time(sim_dict={"s1": self.sim_env.sim_dict})
        file_name = "sim_input_s1.json"

        self.sim_env.engine_obj = SimulationEngine(engine_props=self.sim_env.sim_dict)
        self.sim_env.route_obj = Routing(
            engine_props=self.sim_env.engine_obj.engine_props,
            sdn_props=self.sim_env.rl_props.mock_sdn_dict,
        )

        self.sim_env.sim_props = create_input(
            base_fp=base_fp, engine_props=self.sim_env.sim_dict
        )
        self.sim_env.modified_props = copy.deepcopy(self.sim_env.sim_props)

        save_input(
            base_fp=base_fp,
            properties=self.sim_env.modified_props,
            file_name=file_name,
            data_dict=self.sim_env.modified_props,
        )

    def init_envs(self):
        """
        Sets up environments for RL agents based on the simulation configuration.
        """
        # Environment initialization logic (from the original _init_envs method)
        if (
            self.sim_env.sim_dict["path_algorithm"] in VALID_PATH_ALGORITHMS
            and self.sim_env.sim_dict["is_training"]
        ):
            self.sim_env.path_agent.engine_props = self.sim_env.engine_obj.engine_props
            self.sim_env.path_agent.setup_env(is_path=True)
        elif (
            self.sim_env.sim_dict["core_algorithm"] in VALID_CORE_ALGORITHMS
            and self.sim_env.sim_dict["is_training"]
        ):
            self.sim_env.core_agent.engine_props = self.sim_env.engine_obj.engine_props
            self.sim_env.core_agent.setup_env(is_path=False)

    # NOTE: Future enhancement should support selective loading of specific AI agents
    # based on configuration to optimize memory usage and initialization time
    def load_models(self):
        """
        Loads pretrained models for RL agents and configures agent properties.

        :raises ModelSetupError: This functionality is not yet implemented
        """
        raise ModelSetupError(
            "Model loading functionality is not yet implemented. "
            "This feature requires additional development to support "
            "loading and configuring pretrained RL models."
        )
