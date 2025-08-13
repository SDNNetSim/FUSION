from pathlib import Path
from fusion.modules.rl.utils.gym_envs import create_environment
from fusion.modules.rl import workflow_runner

def train_rl_agent(config):
    """
    Launches RL training using legacy workflow, via new CLI + config system.
    """
    config_path = Path(config.get_args().config_path)

    env, sim_dict, callback_list = create_environment(config_path=config_path)

    # Legacy runner expects flat sim_dict
    flat_dict = sim_dict.get("s1", sim_dict)
    flat_dict["callback"] = callback_list

    workflow_runner.run(env=env, sim_dict=flat_dict, callback_list=callback_list)

    print("✅ RL training pipeline ran successfully using legacy logic.")
