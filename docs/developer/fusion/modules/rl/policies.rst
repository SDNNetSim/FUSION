.. _rl-policies:

=================
RL Policies Module
=================

.. note::

   **Status: BETA**

   This module is currently in **BETA** and is actively being developed.
   The API may evolve in future releases.

   **About BETA Labels:**

   We mark features as BETA to clearly communicate maturity levels to users.
   BETA indicates features that are functional and tested, but may undergo
   API changes or refinement based on user feedback. This transparency helps
   users make informed decisions about which features to adopt in their work.
   BETA does not imply the feature is unreliable - it means we're still
   gathering experience to finalize the design.

.. admonition:: At a Glance
   :class: tip

   :Purpose: Path selection policies for RL and heuristic baselines
   :Location: ``fusion/modules/rl/policies/``
   :Key Classes: ``PathPolicy``, ``BCPolicy``, ``IQLPolicy``, ``KSPFFPolicy``
   :Integration: Orchestrator via ``RLSimulationAdapter`` and ``OfflinePolicyAdapter``

Overview
========

The policies module provides path selection strategies for network resource
allocation. It serves two key purposes:

1. **Heuristic Baselines**: Standard algorithms (KSP-FF, 1+1) for benchmarking
2. **Offline RL Policies**: Pre-trained neural networks for intelligent path selection

**Why Policies?**

In optical network simulation, each request requires selecting one of K candidate
paths. The choice significantly impacts blocking probability, spectrum utilization,
and network survivability. Policies encapsulate this decision logic behind a
consistent interface, enabling:

- Fair comparison between RL and heuristic approaches
- Easy swapping of strategies without code changes
- Integration with the SDN controller (orchestrator) for production use

Orchestrator Integration
========================

Policies integrate with the FUSION simulation stack through the ``RLSimulationAdapter``.
The adapter ensures policies use the **same pipeline instances** as the orchestrator,
maintaining simulation consistency.

.. code-block:: text

   +------------------+     +----------------------+     +----------------+
   | SDNOrchestrator  |<--->| RLSimulationAdapter  |<--->| PathPolicy     |
   |------------------|     |----------------------|     |----------------|
   | - routing        |     | - shares pipelines   |     | - select_path()|
   | - spectrum       |     | - builds state       |     | - action mask  |
   | - network_state  |     | - applies actions    |     |                |
   +------------------+     +----------------------+     +----------------+

**Key Integration Points:**

- Adapter shares pipeline references with orchestrator (same objects, not copies)
- State is built from orchestrator's current network state
- Actions route through orchestrator's allocation pipelines

Using Offline Policies
======================

For offline RL policies (BC, IQL), use the ``OfflinePolicyAdapter``:

.. code-block:: python

   from fusion.modules.rl.policies import BCPolicy
   from fusion.modules.rl.adapter import RLSimulationAdapter, OfflinePolicyAdapter
   from fusion.modules.rl.environments import UnifiedSimEnv

   # Load pre-trained offline policy
   bc_policy = BCPolicy("models/bc_model.pt", device="cpu")

   # Create environment and adapter
   env = UnifiedSimEnv(config=rl_config)
   rl_adapter = env.adapter

   # Wrap policy for environment integration
   offline_adapter = OfflinePolicyAdapter(
       policy=bc_policy,
       rl_adapter=rl_adapter,
   )

   # Run evaluation
   obs, info = env.reset(seed=42)
   while True:
       action_mask = info["action_mask"]
       action = offline_adapter.select_action(obs, action_mask)
       obs, reward, terminated, truncated, info = env.step(action)
       if terminated:
           break

PathPolicy Interface
====================

All policies implement the ``PathPolicy`` abstract base class:

.. code-block:: python

   from abc import ABC, abstractmethod
   from typing import Any

   class PathPolicy(ABC):
       @abstractmethod
       def select_path(
           self,
           state: dict[str, Any],
           action_mask: list[bool]
       ) -> int:
           """
           Select a path index from K candidates.

           :param state: State dictionary with request and path features
           :param action_mask: Feasibility mask (True = path is feasible)
           :return: Selected path index (0 to K-1), or -1 if all masked
           """
           pass

**State Format:**

The state dictionary contains request and path information:

.. code-block:: python

   state = {
       'src': int,                    # Source node
       'dst': int,                    # Destination node
       'slots_needed': int,           # Required spectrum slots
       'est_remaining_time': float,   # Estimated holding time
       'is_disaster': int,            # 0 or 1 (failure scenario)
       'paths': [                     # K candidate paths
           {
               'path_hops': int,
               'min_residual_slots': int,
               'frag_indicator': float,
               'failure_mask': int,
               'dist_to_disaster_centroid': int
           },
           # ... K paths
       ]
   }

**Action Mask:**

The action mask indicates which paths are feasible:

- ``True``: Path can accommodate the request
- ``False``: Path is infeasible (failed link, insufficient spectrum)

When all paths are masked, policies return ``-1`` to indicate the request
should be blocked.

Available Policies
==================

Heuristic Baselines
-------------------

KSPFFPolicy
^^^^^^^^^^^

K-Shortest Path First-Fit - the standard baseline in optical network literature.
Always selects the first feasible path from shortest to longest.

.. code-block:: python

   from fusion.modules.rl.policies import KSPFFPolicy

   policy = KSPFFPolicy()

   # Always returns first feasible path
   action_mask = [False, True, True]  # Path 0 infeasible
   selected = policy.select_path(state, action_mask)
   print(selected)  # 1 (first feasible)

OnePlusOnePolicy
^^^^^^^^^^^^^^^^

1+1 protection policy for survivable networks. Uses pre-computed disjoint
paths, selecting primary if feasible, otherwise backup.

.. code-block:: python

   from fusion.modules.rl.policies import OnePlusOnePolicy

   policy = OnePlusOnePolicy()

   # Primary failed, use backup
   action_mask = [False, True]  # Primary infeasible
   selected = policy.select_path(state, action_mask)
   print(selected)  # 1 (backup path)

Offline RL Policies
-------------------

BCPolicy (Behavior Cloning)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Imitates heuristic behavior using supervised learning on offline datasets.
Trained to mimic KSP-FF or 1+1 decisions.

.. code-block:: python

   from fusion.modules.rl.policies import BCPolicy

   # Load pre-trained model
   policy = BCPolicy("models/bc_model.pt", device="cuda")

   # Select path using learned policy
   selected = policy.select_path(state, action_mask)

**Model Architecture:**

Default BC model is a 3-layer MLP:

- Input: Flattened state features
- Hidden: 128 -> 64 neurons with ReLU
- Output: K-way logits (one per path)

IQLPolicy (Implicit Q-Learning)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Conservative offline RL policy that avoids out-of-distribution actions.
IQL learns value functions without explicit policy optimization.

.. code-block:: python

   from fusion.modules.rl.policies import IQLPolicy

   # Load pre-trained model
   policy = IQLPolicy("models/iql_model.pt", device="cuda")

   # Select path using learned Q-values
   selected = policy.select_path(state, action_mask)

**Why IQL?**

- Avoids overestimation of OOD actions (common in offline RL)
- No need for behavior policy density estimation
- Works well with suboptimal demonstration data

Attention-Based Policies
------------------------

PointerPolicy
^^^^^^^^^^^^^

Attention-based policy using pointer networks for path selection.
Designed for scenarios where path relationships matter.

.. code-block:: python

   from fusion.modules.rl.policies import PointerPolicy
   from stable_baselines3 import PPO

   # Use with SB3
   model = PPO(
       PointerPolicy,
       env,
       policy_kwargs={
           "features_extractor_kwargs": {"dimension": 64}
       }
   )

**When to Use:**

- Path features have complex interdependencies
- Standard MLP policies underperform
- K is small (attention scales O(K^2))

Action Masking
==============

The module provides utilities for computing and applying action masks:

compute_action_mask
-------------------

Computes feasibility mask based on network conditions:

.. code-block:: python

   from fusion.modules.rl.policies import compute_action_mask

   mask = compute_action_mask(
       k_paths=[[0,1,2], [0,3,2], [0,4,5,2]],
       k_path_features=features,
       slots_needed=4
   )
   # Returns: [False, True, True] if path 0 infeasible

**Masking Conditions:**

- ``failure_mask == 1``: Path uses failed link
- ``min_residual_slots < slots_needed``: Insufficient spectrum

apply_fallback_policy
---------------------

Applies fallback when all actions are masked:

.. code-block:: python

   from fusion.modules.rl.policies import apply_fallback_policy, KSPFFPolicy

   fallback = KSPFFPolicy()
   idx = apply_fallback_policy(state, fallback, action_mask)

   if idx == -1:
       print("Request blocked (no feasible path)")

Training Offline Policies
=========================

Offline policies (BC, IQL) are trained on datasets collected from heuristic
runs. The training pipeline is separate from the policy module.

**Data Collection:**

.. code-block:: python

   # Collect data using heuristic policy
   policy = KSPFFPolicy()
   dataset = []

   for episode in range(num_episodes):
       obs, info = env.reset()
       while True:
           action = policy.select_path(state, info["action_mask"])
           next_obs, reward, done, _, info = env.step(action)
           dataset.append((obs, action, reward, next_obs, done))
           if done:
               break

**Training BC:**

.. code-block:: python

   # Train BC model (simplified)
   model = BCModel(input_dim, k_paths)
   optimizer = torch.optim.Adam(model.parameters())

   for obs, action, _, _, _ in dataset:
       logits = model(obs)
       loss = F.cross_entropy(logits, action)
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()

   torch.save(model, "models/bc_model.pt")

Configuration Reference
=======================

Policy Selection in Config
--------------------------

Select policies via configuration:

.. code-block:: ini

   [rl_settings]
   # Heuristic baselines
   path_algorithm = ksp_ff
   # path_algorithm = one_plus_one

   # Offline RL (requires model path)
   # path_algorithm = bc
   # bc_model_path = models/bc_model.pt

   # path_algorithm = iql
   # iql_model_path = models/iql_model.pt

Model File Format
-----------------

Offline policies expect PyTorch model files:

- **Full model**: ``torch.save(model, path)``
- **State dict**: ``torch.save(model.state_dict(), path)`` (requires architecture inference)

File Reference
==============

.. code-block:: text

   fusion/modules/rl/policies/
   |-- __init__.py              # Public exports
   |-- README.md                # Module documentation
   |-- base.py                  # PathPolicy abstract base class
   |-- ksp_ff_policy.py         # KSP-FF baseline
   |-- one_plus_one_policy.py   # 1+1 protection baseline
   |-- bc_policy.py             # Behavior Cloning policy
   |-- iql_policy.py            # Implicit Q-Learning policy
   |-- pointer_policy.py        # Attention-based pointer network
   `-- action_masking.py        # Action mask utilities

**Public API:**

.. code-block:: python

   from fusion.modules.rl.policies import (
       # Base class
       PathPolicy,

       # Heuristic baselines
       KSPFFPolicy,
       OnePlusOnePolicy,

       # Offline RL policies
       BCPolicy,
       IQLPolicy,

       # Attention-based
       PointerHead,
       PointerPolicy,

       # Utilities
       compute_action_mask,
       apply_fallback_policy,
   )

Related Documentation
=====================

- :ref:`rl-adapter` - RLSimulationAdapter and OfflinePolicyAdapter
- :ref:`rl-environments` - UnifiedSimEnv for policy evaluation
- :ref:`rl-algorithms` - Online RL algorithms (alternative to offline policies)
- :ref:`rl-module` - Parent RL module documentation

.. seealso::

   - `Implicit Q-Learning Paper <https://arxiv.org/abs/2110.06169>`_
   - `Behavior Cloning Tutorial <https://imitation.readthedocs.io/>`_
   - `Pointer Networks Paper <https://arxiv.org/abs/1506.03134>`_
