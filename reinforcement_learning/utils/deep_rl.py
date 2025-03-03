from reinforcement_learning.args.general_args import VALID_PATH_ALGORITHMS
from reinforcement_learning.utils.general_utils import determine_model_type
from reinforcement_learning.args.registry_args import ALGORITHM_REGISTRY


# TODO: (drl_path_agents) Only works for 's1'
def get_algorithm_instance(sim_dict: dict, rl_props: object, engine_obj: object):
    """
    Retrieve an instance of the algorithm class associated with the model type.

    :param sim_dict: A dictionary containing the simulation configuration, including the algorithm type.
    :param rl_props: An object containing properties for the RL algorithm.
    :param engine_obj: An object containing properties for the simulation engine.
    """
    model_type = determine_model_type(sim_dict=sim_dict)

    if '_' not in model_type:
        raise ValueError("Algorithm info must include both algorithm and agent type (e.g., 'ppo_path').")

    if 's1' in sim_dict:
        algorithm = sim_dict['s1'].get(model_type)
    else:
        algorithm = sim_dict.get(model_type)

    # Non-DRL case, skip
    if algorithm in VALID_PATH_ALGORITHMS and algorithm not in ALGORITHM_REGISTRY:
        engine_obj.engine_props['is_drl_agent'] = False
        return None

    engine_obj.engine_props['is_drl_agent'] = True
    if algorithm not in ALGORITHM_REGISTRY:
        raise NotImplementedError(f"Algorithm '{algorithm}' is not registered.")

    algorithm_class = ALGORITHM_REGISTRY[algorithm]['class']
    return algorithm_class(rl_props=rl_props, engine_obj=engine_obj)


def get_obs_space(sim_dict: dict, rl_props: object, engine_obj: object):
    """
    Get the observation space for the provided algorithm type.

    :param sim_dict: A dictionary containing the simulation configuration, including the algorithm type.
    :param rl_props: An object containing properties for the RL algorithm.
    :param engine_obj: An object containing properties for the simulation engine.

    :return: Observation space as defined by the algorithm.
    """
    algorithm_instance = get_algorithm_instance(sim_dict=sim_dict, rl_props=rl_props, engine_obj=engine_obj)
    if algorithm_instance is None:
        return None

    return algorithm_instance.get_obs_space()


def get_action_space(sim_dict: dict, rl_props: object, engine_obj: object):
    """
    Get the action space for the provided algorithm type.

    :param sim_dict: A dictionary containing the simulation configuration, including the algorithm type.
    :param rl_props: An object containing properties for the RL algorithm.
    :param engine_obj: An object containing properties for the simulation engine.

    :return: Action space as defined by the algorithm.
    """
    algorithm_instance = get_algorithm_instance(sim_dict=sim_dict, rl_props=rl_props, engine_obj=engine_obj)
    if algorithm_instance is None:
        return None

    return algorithm_instance.get_action_space()
