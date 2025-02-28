from rl_scripts.utils.setup_helpers import setup_ppo
from rl_scripts.algorithms.ppo import PPO

ALGORITHM_REGISTRY = {
    'ppo': {
        'setup': setup_ppo,
        'load': None,
        'class': PPO,
    },
}
