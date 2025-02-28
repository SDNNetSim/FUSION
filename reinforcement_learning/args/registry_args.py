from reinforcement_learning.utils.setup import setup_ppo
from reinforcement_learning.algorithms.ppo import PPO
from reinforcement_learning.utils.setup import  setup_A2C
from reinforcement_learning.algorithms.ppo import A2C


ALGORITHM_REGISTRY = {
    'ppo': {
        'setup': setup_ppo,
        'load': None,
        'class': PPO,
    },
    'a2c': {
        'setup': setup_A2C,
        'load': None,
        'class': A2C,
    }
}
