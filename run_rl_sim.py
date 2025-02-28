from reinforcement_learning.args.registry_args import ALGORITHM_REGISTRY
from reinforcement_learning.workflow_runner import run_optuna_study, run
from reinforcement_learning.utils.gym_envs import create_environment


# TODO: (drl_path_agents) Put 'utils' file ending (imports) in the standards and guidelines
# TODO: (drl_path_agents) No support for core or spectrum assignment
# TODO: (drl_path_agents) Does not support multi-band
# TODO: (drl_path_agents) Q-Learning does not save iteratively
# TODO: (drl_path_agents) Create run mods directory and raise an error if it doesn't exist

def run_rl_sim():
    """
    The main function that controls reinforcement learning simulations, including hyperparameter optimization.
    """
    env, sim_dict, callback_obj = create_environment()

    if not sim_dict['optimize'] and not sim_dict['optimize_hyperparameters']:
        run(env=env, sim_dict=sim_dict, callback_obj=callback_obj)
    else:
        # For DRL only
        if sim_dict['path_algorithm'] in ALGORITHM_REGISTRY:
            run(env=env, sim_dict=sim_dict, callback_obj=callback_obj)
        else:
            run_optuna_study(sim_dict=sim_dict)


if __name__ == '__main__':
    run_rl_sim()
