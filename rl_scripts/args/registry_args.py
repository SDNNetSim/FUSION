from rl_scripts.helpers.setup_helpers import setup_ppo
from rl_scripts.algorithms.ppo import PPO
from rl_scripts.helpers.setup_helpers import  setup_A2C
from rl_scripts.algorithms.a2c import A2C


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
