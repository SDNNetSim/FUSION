from fusion.modules.rl.utils.setup import setup_ppo
from fusion.modules.rl.utils.setup import setup_a2c
from fusion.modules.rl.utils.setup import setup_dqn
from fusion.modules.rl.utils.setup import setup_qr_dqn
from fusion.modules.rl.algorithms.ppo import PPO
from fusion.modules.rl.algorithms.a2c import A2C
from fusion.modules.rl.algorithms.dqn import DQN
from fusion.modules.rl.algorithms.qr_dqn import QrDQN

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
