"""
Reinforcement Learning Module.

Provides reinforcement learning capabilities for network optimization in FUSION,
including traditional RL (Q-learning, bandits), deep RL via Stable-Baselines3
(PPO, A2C, DQN), and offline RL policies (BC, IQL).

Key Components:
- agents: RL agent implementations (PathAgent, CoreAgent, SpectrumAgent)
- algorithms: Algorithm implementations (QLearning, PPO, DQN, etc.)
- environments: Gymnasium-compatible environments (UnifiedSimEnv)
- policies: Path selection policies (KSPFFPolicy, BCPolicy, IQLPolicy)
- adapter: V4 simulation integration layer (RLSimulationAdapter)
"""
