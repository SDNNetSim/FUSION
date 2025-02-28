from reinforcement_learning.utils.setup import setup_ppo
from reinforcement_learning.algorithms.ppo import PPO

ALGORITHM_REGISTRY = {
    'ppo': {
        'setup': setup_ppo,
        'load': None,
        'class': PPO,
    },
}
