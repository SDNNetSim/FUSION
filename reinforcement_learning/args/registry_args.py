from reinforcement_learning.utils.setup import setup_ppo
from reinforcement_learning.utils.setup import setup_a2c
from reinforcement_learning.utils.setup import setup_dqn
from reinforcement_learning.utils.setup import setup_qr_dqn
from reinforcement_learning.algorithms.ppo import PPO
from reinforcement_learning.algorithms.a2c import A2C
from reinforcement_learning.algorithms.dqn import DQN
from reinforcement_learning.algorithms.qr_dqn import QrDQN


ALGORITHM_REGISTRY = {
    'ppo': {
        'setup': setup_ppo,
        'load': None,
        'class': PPO,
    },
    'a2c': {
        'setup': setup_a2c,
        'load': None,
        'class': A2C,
    },
    'dqn': {
        'setup': setup_dqn,
        'load': None,
        'class': DQN,
    },
    'qr_dqn': {
        'setup': setup_qr_dqn,
        'load': None,
        'class': QrDQN,
    },
}
