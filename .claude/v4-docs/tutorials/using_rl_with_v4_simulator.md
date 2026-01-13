# Tutorial: Using RL with the V4 Simulator

## Overview

This tutorial shows how to use reinforcement learning (RL) with FUSION's V4 architecture. The key advantage of V4 is that RL uses the SAME pipelines as non-RL simulation, eliminating duplicated logic and ensuring consistent behavior.

## Prerequisites

- FUSION installed with RL dependencies: `pip install fusion[rl]`
- Familiarity with Gymnasium environments
- Basic understanding of Stable-Baselines3 (optional)

---

## Quick Start

### 1. Create a Simulation Configuration

```python
from fusion.domain.config import SimulationConfig

# From parameters
config = SimulationConfig(
    network_name="NSFNET",
    cores_per_link=1,
    band_list=("c",),
    band_slots={"c": 320},
    guard_slots=1,
    num_requests=1000,
    erlang=100.0,
    holding_time=1.0,
    route_method="k_shortest_path",
    k_paths=3,
    allocation_method="first_fit",
    grooming_enabled=False,
    slicing_enabled=False,
    max_slices=1,
    snr_enabled=False,
    snr_type=None,
    snr_recheck=False,
    can_partially_serve=False,
    modulation_formats={},
    mod_per_bw={},
    snr_thresholds={},
)

# Or from legacy engine_props dict
config = SimulationConfig.from_engine_props(engine_props)
```

### 2. Create the RL Environment

```python
from fusion.rl.environments import UnifiedSimEnv

env = UnifiedSimEnv(config)

# Standard Gymnasium interface
obs, info = env.reset(seed=42)
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")
print(f"Action mask: {info['action_mask']}")
```

### 3. Run a Simple Episode

```python
obs, info = env.reset(seed=42)
total_reward = 0
steps = 0

while True:
    # Get valid actions from mask
    action_mask = info.get("action_mask")
    valid_actions = [i for i, m in enumerate(action_mask) if m > 0]

    # Random valid action
    if valid_actions:
        action = np.random.choice(valid_actions)
    else:
        action = 0  # Fallback

    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    steps += 1

    if terminated:
        break

print(f"Episode finished: {steps} steps, total reward: {total_reward:.2f}")
```

---

## Understanding the Environment

### Observation Space

The observation is a dictionary with these keys:

| Key | Shape | Description |
|-----|-------|-------------|
| `source` | `(num_nodes,)` | One-hot encoding of source node |
| `destination` | `(num_nodes,)` | One-hot encoding of destination node |
| `bandwidth` | `(1,)` | Normalized bandwidth (0-1) |
| `path_lengths` | `(k_paths,)` | Normalized path lengths |
| `feasibility` | `(k_paths,)` | 1.0 if path is feasible, 0.0 otherwise |
| `congestion` | `(k_paths,)` | Path congestion (0-1) |
| `num_hops` | `(k_paths,)` | Number of hops per path |
| `network_utilization` | `(1,)` | Overall network utilization |

### Action Space

- **Type**: `Discrete(k_paths)`
- **Meaning**: Index of path to use for allocation
- **Action Mask**: Available in `info["action_mask"]`

### Rewards

| Event | Reward | Notes |
|-------|--------|-------|
| Successful allocation | +1.0 | Base reward |
| Blocked (no path) | -1.0 | Penalty |
| Grooming bonus | +0.1 | If reusing existing lightpath |
| Slicing penalty | -0.05 | If request was sliced |

---

## Working with Action Masks

### Why Action Masks?

Not all paths are feasible for every request. The action mask tells you which actions are valid:

```python
obs, info = env.reset(seed=42)
mask = info["action_mask"]

# mask[i] = 1.0 means action i is valid
# mask[i] = 0.0 means action i would fail
print(f"Valid actions: {[i for i, m in enumerate(mask) if m > 0]}")
```

### Using Action Masks in Your Policy

```python
def masked_policy(obs, info, model=None):
    """Policy that respects action masks."""
    mask = info.get("action_mask", np.ones(3))
    valid_actions = np.where(mask > 0)[0]

    if len(valid_actions) == 0:
        return 0  # Fallback

    if model is not None:
        # Get action probabilities from model
        logits = model.predict(obs)
        # Mask invalid actions
        masked_logits = np.where(mask > 0, logits, -np.inf)
        return np.argmax(masked_logits)
    else:
        # Random valid action
        return np.random.choice(valid_actions)
```

---

## Integration with Stable-Baselines3

### With MaskablePPO (Recommended)

```python
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from fusion.rl.environments import UnifiedSimEnv

# Create environment
config = SimulationConfig.from_engine_props(engine_props)
env = UnifiedSimEnv(config)

# Wrap for action masking
def get_action_mask(env):
    return env._get_action_mask()

env = ActionMasker(env, get_action_mask)

# Create and train model
model = MaskablePPO(
    "MultiInputPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
)

model.learn(total_timesteps=100_000)
model.save("fusion_ppo_masked")
```

### Loading and Using a Trained Model

```python
from sb3_contrib import MaskablePPO

# Load model
model = MaskablePPO.load("fusion_ppo_masked")

# Evaluate
env = UnifiedSimEnv(config)
obs, info = env.reset(seed=42)

total_reward = 0
while True:
    action, _ = model.predict(obs, action_masks=info["action_mask"])
    obs, reward, terminated, _, info = env.step(action)
    total_reward += reward
    if terminated:
        break

print(f"Total reward: {total_reward:.2f}")
```

---

## Configuring Episodes, Seeds, and Evaluation

### Controlling Episode Length

Episode length is determined by `num_requests` in config:

```python
# Short episodes for faster iteration
config_short = SimulationConfig.from_engine_props({
    **engine_props,
    "num_requests": 100,
})

# Full-length episodes for evaluation
config_full = SimulationConfig.from_engine_props({
    **engine_props,
    "num_requests": 10000,
})
```

### Reproducible Experiments

```python
# Same seed = same traffic, same results
env = UnifiedSimEnv(config)

# Run 1
env.reset(seed=42)
results1 = run_episode(env, policy)

# Run 2 - identical
env.reset(seed=42)
results2 = run_episode(env, policy)

assert results1 == results2
```

### Evaluation Runs

```python
def evaluate_policy(env, policy, num_episodes=10, seed_base=1000):
    """Evaluate policy over multiple episodes."""
    results = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed_base + ep)

        episode_reward = 0
        blocked = 0
        total = 0

        while True:
            action = policy(obs, info)
            obs, reward, terminated, _, info = env.step(action)

            episode_reward += reward
            total += 1
            if reward < 0:
                blocked += 1

            if terminated:
                break

        results.append({
            "episode": ep,
            "total_reward": episode_reward,
            "blocking_rate": blocked / total if total > 0 else 0,
            "num_requests": total,
        })

    return results

# Compare policies
random_results = evaluate_policy(env, random_policy)
trained_results = evaluate_policy(env, trained_policy)

print(f"Random blocking: {np.mean([r['blocking_rate'] for r in random_results]):.2%}")
print(f"Trained blocking: {np.mean([r['blocking_rate'] for r in trained_results]):.2%}")
```

---

## Logging and Interpreting Results

### Accessing Statistics

```python
obs, info = env.reset(seed=42)

# Run episode
while True:
    obs, reward, terminated, _, info = env.step(action)
    if terminated:
        break

# Get final statistics
final_stats = info.get("final_stats", {})
print(f"Blocking probability: {final_stats.get('blocking_probability', 0):.4f}")
print(f"Groomed requests: {final_stats.get('groomed_requests', 0)}")
print(f"Sliced requests: {final_stats.get('sliced_requests', 0)}")
```

### Comparing with Non-RL Simulation

```python
from fusion.core.simulation import SimulationEngine

# Same config, same seed
config = SimulationConfig.from_engine_props(engine_props)
seed = 42

# Run non-RL simulation
sim_engine = SimulationEngine(config)
sim_engine.reset(seed=seed)
sim_results = sim_engine.run_simulation()
print(f"Simulation blocking: {sim_results.blocking_probability:.4f}")

# Run RL with heuristic policy (same as KSP)
env = UnifiedSimEnv(config)
obs, info = env.reset(seed=seed)

rl_blocked = 0
rl_total = 0
while True:
    action = 0  # Always shortest path (like KSP)
    obs, reward, terminated, _, info = env.step(action)
    rl_total += 1
    if reward < 0:
        rl_blocked += 1
    if terminated:
        break

rl_blocking = rl_blocked / rl_total
print(f"RL blocking: {rl_blocking:.4f}")

# Should be nearly identical
assert abs(sim_results.blocking_probability - rl_blocking) < 0.01
```

### TensorBoard Logging

```python
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

# Wrap environment for logging
env = Monitor(UnifiedSimEnv(config))

# Create evaluation callback
eval_env = Monitor(UnifiedSimEnv(config))
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs/best_model",
    log_path="./logs/eval",
    eval_freq=10000,
    deterministic=True,
)

# Train with logging
model = MaskablePPO("MultiInputPolicy", env, verbose=1, tensorboard_log="./logs/tb")
model.learn(total_timesteps=100_000, callback=eval_callback)

# View in TensorBoard: tensorboard --logdir=./logs/tb
```

---

## Custom Reward Functions

### Modifying Rewards

The `RLSimulationAdapter` computes rewards from `AllocationResult`. You can customize:

```python
from fusion.rl.adapter import RLSimulationAdapter

class CustomAdapter(RLSimulationAdapter):
    def compute_reward(self, result, request):
        """Custom reward function."""
        if not result.success:
            return -1.0

        reward = 1.0

        # Bonus for high-bandwidth requests
        if request.bandwidth_gbps >= 200:
            reward += 0.5

        # Penalty for long paths
        if result.lightpaths_created:
            # Access path length from domain objects
            pass

        # Bonus for resource efficiency
        if result.is_groomed:
            reward += 0.2

        return reward
```

### Using Custom Adapter

```python
from fusion.core.pipeline_factory import PipelineFactory

config = SimulationConfig.from_engine_props(engine_props)
orchestrator = PipelineFactory.create_orchestrator(config)

custom_adapter = CustomAdapter(config, orchestrator)

# Use in environment
env = UnifiedSimEnv(config)
env.adapter = custom_adapter  # Replace adapter
```

---

## Custom Observations

### Adding Features

```python
from fusion.rl.adapter import RLSimulationAdapter
import numpy as np

class ExtendedAdapter(RLSimulationAdapter):
    def build_observation(self, request, options, network_state):
        # Get base observation
        obs = super().build_observation(request, options, network_state)

        # Add custom features
        obs["holding_time"] = np.array(
            [request.holding_time / 10.0], dtype=np.float32
        )

        obs["max_congestion"] = np.array(
            [max(o.congestion for o in options) if options else 0.0],
            dtype=np.float32,
        )

        return obs
```

### Updating Observation Space

```python
from gymnasium import spaces

class CustomEnv(UnifiedSimEnv):
    def _setup_spaces(self):
        super()._setup_spaces()

        # Extend observation space
        self.observation_space.spaces["holding_time"] = spaces.Box(
            0, 1, shape=(1,), dtype=np.float32
        )
        self.observation_space.spaces["max_congestion"] = spaces.Box(
            0, 1, shape=(1,), dtype=np.float32
        )
```

---

## Multiple Training Configurations

### Different Network Topologies

```python
topologies = ["NSFNET", "USNET", "COST239"]

for topo in topologies:
    config = SimulationConfig.from_engine_props({
        **base_props,
        "network": topo,
    })
    env = UnifiedSimEnv(config)

    model = MaskablePPO("MultiInputPolicy", env)
    model.learn(total_timesteps=50_000)
    model.save(f"models/ppo_{topo}")
```

### Different Traffic Loads

```python
erlangs = [50, 100, 150, 200]

for erlang in erlangs:
    config = SimulationConfig.from_engine_props({
        **base_props,
        "erlang": erlang,
    })
    env = UnifiedSimEnv(config)

    # Train or evaluate
    results = evaluate_policy(env, policy)
    print(f"Erlang {erlang}: blocking = {np.mean([r['blocking_rate'] for r in results]):.2%}")
```

---

## Troubleshooting

### Issue: Different Results from Legacy RL

**Symptom**: Blocking rate differs between `GeneralSimEnv` and `UnifiedSimEnv`.

**Cause**: V4 uses real pipelines which may include SNR checking that was skipped in legacy.

**Solution**: Ensure config matches:
```python
config = SimulationConfig.from_engine_props({
    **engine_props,
    "snr_type": None,  # Disable SNR to match legacy
})
```

### Issue: All Actions Masked

**Symptom**: Action mask is all zeros.

**Cause**: No feasible paths for current request.

**Solution**: Handle gracefully:
```python
mask = info.get("action_mask", np.ones(k_paths))
if np.sum(mask) == 0:
    # No valid actions - this will result in blocking
    action = 0
else:
    valid = np.where(mask > 0)[0]
    action = np.random.choice(valid)
```

### Issue: Non-Deterministic Results

**Symptom**: Same seed produces different results.

**Cause**: RNG not properly seeded.

**Solution**: Always call reset with seed:
```python
obs, info = env.reset(seed=42)  # Not env.reset()
```

---

## Next Steps

- Read `architecture/rl_integration.md` for detailed architecture
- See `decisions/0009-rl-env-design.md` for design rationale
- Check `testing/phase_4_testing.md` for test examples
- Explore `fusion/rl/` source code for implementation details

---

## Related Documentation

- `architecture/rl_integration.md` - RL architecture
- `architecture/orchestration.md` - SDNOrchestrator design
- `architecture/pipelines.md` - Pipeline implementations
- `migration/phase_4_rl_integration.md` - Migration details
