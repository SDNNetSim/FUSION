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


def run_training_pipeline(args):
    """
    Pipeline function for running RL training from CLI.

    Args:
        args: Parsed command line arguments
    """
    # Create config object with args
    class ConfigWrapper:  # pylint: disable=too-few-public-methods
        """Wrapper class to adapt args for legacy train_rl_agent function."""
        def __init__(self, args):
            self.args = args
        def get_args(self):
            """Return the arguments."""
            return self.args

    config = ConfigWrapper(args)
    train_rl_agent(config)
