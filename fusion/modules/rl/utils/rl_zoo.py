from fusion.modules.rl.utils.errors import ModelSetupError


# NOTE: RL Zoo functionality has been deprecated and replaced by direct algorithm calls
# This module is maintained for backward compatibility but should not be used in new code
def run_rl_zoo():
    """
    Executes RL Zoo commands for training or running simulations using the specified algorithm.

    :raises ModelSetupError: This functionality has been deprecated
    """
    raise ModelSetupError(
        "RL Zoo functionality has been deprecated and is no longer supported. "
        "Please use direct algorithm setup functions from the setup module instead. "
        "See setup.py for available model setup functions (setup_ppo, setup_dqn, etc.)."
    )
