.. _policies-module:

===============
Policies Module
===============

.. tip::

   **Lost in the architecture?** This page explains where policies fit.
   Start with :ref:`the-big-picture` below before diving into details.

Overview
========

.. admonition:: At a Glance
   :class: tip

   :Purpose: Decision-making for path selection ("which path should I use?")
   :Location: ``fusion/policies/``
   :Key Files: ``heuristic_policy.py``, ``ml_policy.py``, ``rl_policy.py``, ``policy_factory.py``
   :Used By: ``SDNOrchestrator`` (new architecture)
   :Protocol: Implements ``ControlPolicy`` from ``fusion/interfaces/control_policy.py``

**What this module does:**

- Chooses which path to use when serving a network request
- Provides pluggable strategies: heuristics, ML models, RL models
- Decouples "how to decide" from "how to execute"

**What this module does NOT do:**

- Find paths (that's ``fusion/modules/routing/`` or ``fusion/pipelines/``)
- Assign spectrum (that's ``fusion/modules/spectrum/``)
- Run the simulation (that's ``fusion/core/``)

.. _the-big-picture:

The Big Picture: Where Policies Fit
===================================

.. important::

   **The core confusion:** FUSION has many components with similar-sounding names.
   This section explains what each does and how they connect.

The Request Lifecycle
---------------------

When a network request arrives, here's what happens:

.. code-block:: text

   +==========================================================================+
   |                    REQUEST LIFECYCLE (Orchestrator)                       |
   +==========================================================================+
   |                                                                           |
   |   1. REQUEST ARRIVES                                                      |
   |          |                                                                |
   |          v                                                                |
   |   2. FIND CANDIDATE PATHS                                                 |
   |      +------------------+                                                 |
   |      | RoutingPipeline  |  "Here are 5 possible paths from A to Z"        |
   |      +--------+---------+                                                 |
   |               |                                                           |
   |               v                                                           |
   |   3. CHECK FEASIBILITY                                                    |
   |      +------------------+                                                 |
   |      | SpectrumPipeline |  "Paths 1, 3, 5 have available spectrum"        |
   |      | SNRPipeline      |  "Paths 1, 5 meet SNR requirements"             |
   |      +--------+---------+                                                 |
   |               |                                                           |
   |               v                                                           |
   |   4. SELECT PATH  <-- THIS IS WHERE POLICIES COME IN                      |
   |      +------------------+                                                 |
   |      | ControlPolicy    |  "Use path 5 (it's the least congested)"        |
   |      +--------+---------+                                                 |
   |               |                                                           |
   |               v                                                           |
   |   5. ALLOCATE RESOURCES                                                   |
   |      +------------------+                                                 |
   |      | SpectrumPipeline |  "Reserved slots 10-15 on path 5"               |
   |      +--------+---------+                                                 |
   |               |                                                           |
   |               v                                                           |
   |   6. LIGHTPATH CREATED                                                    |
   |                                                                           |
   +==========================================================================+

**Policies answer ONE question:** Given multiple feasible paths, which one should we use?

The Component Map
-----------------

Here's how all the confusing components relate:

.. code-block:: text

   +==========================================================================+
   |                         FUSION COMPONENT MAP                              |
   +==========================================================================+
   |                                                                           |
   |   DECISION LAYER ("What to do")                                           |
   |   +------------------------------------------------------------------+    |
   |   |  fusion/policies/           <-- YOU ARE HERE                     |    |
   |   |  - Chooses which path to use                                     |    |
   |   |  - Heuristics, ML models, RL models                              |    |
   |   +------------------------------------------------------------------+    |
   |                              |                                            |
   |                              | "Use path 3"                               |
   |                              v                                            |
   |   ORCHESTRATION LAYER ("How to coordinate")                               |
   |   +------------------------------------------------------------------+    |
   |   |  fusion/core/orchestrator.py                                     |    |
   |   |  - Coordinates the request lifecycle                             |    |
   |   |  - Calls pipelines in order                                      |    |
   |   |  - Asks policy for decisions                                     |    |
   |   +------------------------------------------------------------------+    |
   |                              |                                            |
   |                              v                                            |
   |   PIPELINE LAYER ("How to do multi-step operations")                      |
   |   +------------------------------------------------------------------+    |
   |   |  fusion/pipelines/                                               |    |
   |   |  - RoutingPipeline: find paths with protection                   |    |
   |   |  - SlicingPipeline: split large requests                         |    |
   |   |  - ProtectionPipeline: 1+1 backup allocation                     |    |
   |   +------------------------------------------------------------------+    |
   |                              |                                            |
   |                              v                                            |
   |   ALGORITHM LAYER ("How to do single operations")                         |
   |   +------------------------------------------------------------------+    |
   |   |  fusion/modules/routing/    - K-shortest path, congestion-aware  |    |
   |   |  fusion/modules/spectrum/   - First-fit, best-fit assignment     |    |
   |   |  fusion/modules/snr/        - GSNR calculation                   |    |
   |   +------------------------------------------------------------------+    |
   |                              |                                            |
   |                              v                                            |
   |   DATA LAYER ("What we're working with")                                  |
   |   +------------------------------------------------------------------+    |
   |   |  fusion/domain/             - Request, NetworkState, Lightpath   |    |
   |   |  fusion/interfaces/         - Protocols (contracts)              |    |
   |   +------------------------------------------------------------------+    |
   |                                                                           |
   +==========================================================================+

Why Do Policies Exist?
----------------------

**Without policies:** The path selection logic is hardcoded in the simulator.
Want to try a different strategy? Edit the simulator code.

**With policies:** Path selection is pluggable. The simulator asks "which path?"
and the policy answers. Want to try ML? Just swap the policy.

.. code-block:: python

   # Without policies (hardcoded in simulator)
   def serve_request(request, paths):
       for path in paths:
           if path.is_feasible:
               return allocate(path)  # Always picks first feasible

   # With policies (pluggable)
   def serve_request(request, paths, policy):
       action = policy.select_action(request, paths, network_state)
       return allocate(paths[action])  # Policy decides

This separation enables:

1. **Experimentation**: Compare heuristics vs ML vs RL without changing simulator
2. **Research**: Train RL agents, then deploy trained policies
3. **Flexibility**: Different policies for different scenarios

Legacy vs. Orchestrator Architecture
====================================

.. warning::

   FUSION has TWO architectures. Understanding which one you're working with
   is critical to understanding where policies fit.

.. code-block:: text

   +==========================================================================+
   |                    LEGACY vs ORCHESTRATOR                                 |
   +==========================================================================+
   |                                                                           |
   |   LEGACY (SDNController)              ORCHESTRATOR (SDNOrchestrator)      |
   |   ========================            ==============================      |
   |                                                                           |
   |   - Decision logic embedded           - Decision logic in policies        |
   |   - Uses modules directly             - Uses pipelines + adapters         |
   |   - No ControlPolicy protocol         - Uses ControlPolicy protocol       |
   |   - Hardcoded heuristics              - Pluggable policies                |
   |                                                                           |
   |   fusion/core/sdn_controller.py       fusion/core/orchestrator.py         |
   |            |                                      |                       |
   |            v                                      v                       |
   |   fusion/modules/routing/             fusion/pipelines/ (routing)         |
   |   fusion/modules/spectrum/            fusion/pipelines/ (slicing)         |
   |            |                                      |                       |
   |            |                                      v                       |
   |            |                          fusion/policies/  <-- NEW           |
   |            |                                      |                       |
   |            v                                      v                       |
   |   [Hardcoded: first feasible]         [Pluggable: any policy]             |
   |                                                                           |
   +==========================================================================+

**Key point:** Policies are part of the NEW orchestrator architecture. If you're
working with the legacy SDNController, policies are not used.

How Policies Work Internally
============================

The ControlPolicy Protocol
--------------------------

All policies implement this protocol (from ``fusion/interfaces/control_policy.py``):

.. code-block:: python

   class ControlPolicy(Protocol):
       def select_action(
           self,
           request: Request,
           options: list[PathOption],
           network_state: NetworkState,
       ) -> int:
           """Return index of selected path, or -1 if none."""
           ...

       def update(
           self,
           request: Request,
           action: int,
           reward: float,
       ) -> None:
           """Update policy from experience (no-op for heuristics/deployment)."""
           ...

Policy Types
------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Type
     - When to Use
     - Examples
   * - **Heuristic**
     - Baselines, simple deployments, fallbacks
     - FirstFeasible, ShortestFeasible, LeastCongested
   * - **ML**
     - Deploy pre-trained supervised models
     - PyTorch, sklearn, ONNX models
   * - **RL**
     - Deploy pre-trained RL agents
     - SB3 PPO, MaskablePPO, DQN

Decision Flow
-------------

.. code-block:: text

   +===========================================================================+
   |                      POLICY DECISION FLOW                                  |
   +===========================================================================+
   |                                                                            |
   |   INPUT                                                                    |
   |   +------------------------+                                               |
   |   | Request                |  bandwidth=100Gbps, src=A, dst=Z              |
   |   | PathOptions (list)     |  [Path0, Path1, Path2, Path3, Path4]          |
   |   | NetworkState           |  current topology and spectrum state          |
   |   +------------------------+                                               |
   |              |                                                             |
   |              v                                                             |
   |   FEASIBILITY CHECK (already done by pipelines)                            |
   |   +------------------------+                                               |
   |   | PathOption.is_feasible |  [True, False, True, False, True]             |
   |   +------------------------+                                               |
   |              |                                                             |
   |              v                                                             |
   |   POLICY DECISION                                                          |
   |   +------------------------+                                               |
   |   | Heuristic: "Pick shortest feasible" -> Path 2                          |
   |   | ML Model:  "Score each, pick highest" -> Path 4                        |
   |   | RL Model:  "Predict action from obs" -> Path 0                         |
   |   +------------------------+                                               |
   |              |                                                             |
   |              v                                                             |
   |   OUTPUT                                                                   |
   |   +------------------------+                                               |
   |   | action = 2             |  (index of selected path)                     |
   |   +------------------------+                                               |
   |                                                                            |
   +===========================================================================+

Components
==========

heuristic_policy.py
-------------------

:Purpose: Rule-based deterministic policies
:Classes: ``FirstFeasiblePolicy``, ``ShortestFeasiblePolicy``, ``LeastCongestedPolicy``, ``RandomFeasiblePolicy``, ``LoadBalancedPolicy``

.. code-block:: python

   from fusion.policies import ShortestFeasiblePolicy

   policy = ShortestFeasiblePolicy()
   action = policy.select_action(request, options, network_state)
   # Returns index of shortest feasible path

ml_policy.py
------------

:Purpose: Deploy pre-trained ML models (PyTorch, sklearn, ONNX)
:Classes: ``MLControlPolicy``, ``FeatureBuilder``, model wrappers

.. code-block:: python

   from fusion.policies import MLControlPolicy

   policy = MLControlPolicy(
       model_path="model.pt",
       fallback_type="first_feasible",  # Fallback if model fails
   )
   action = policy.select_action(request, options, network_state)

**Key feature:** Automatic fallback to heuristic on errors.

rl_policy.py
------------

:Purpose: Deploy pre-trained Stable-Baselines3 models
:Classes: ``RLPolicy``, ``RLControlPolicy``

.. code-block:: python

   from fusion.policies import RLPolicy

   policy = RLPolicy.from_file(
       model_path="trained_ppo.zip",
       algorithm="MaskablePPO",
   )
   action = policy.select_action(request, options, network_state)

**Key feature:** Supports action masking for feasibility constraints.

policy_factory.py
-----------------

:Purpose: Create policies from configuration
:Classes: ``PolicyFactory``, ``PolicyConfig``

.. code-block:: python

   from fusion.policies import PolicyFactory, PolicyConfig

   # From config object
   config = PolicyConfig(policy_type="heuristic", policy_name="shortest")
   policy = PolicyFactory.create(config)

   # From dictionary (e.g., config file)
   policy = PolicyFactory.from_dict({"policy_type": "rl", "model_path": "model.zip"})

Frequently Asked Questions
==========================

**Q: What's the difference between policies and pipelines?**

- **Policies** = decision-making ("which path?")
- **Pipelines** = execution ("find paths", "assign spectrum", "allocate protection")

**Q: What's the difference between policies and modules/routing?**

- **Policies** = high-level decision ("use path 3")
- **modules/routing** = low-level algorithm ("here are the 5 shortest paths")

**Q: Why do ML/RL policies have update() if they don't learn?**

The ``update()`` method satisfies the ``ControlPolicy`` protocol. For deployment
policies, it's a no-op. For online RL training, use ``UnifiedSimEnv`` with SB3's
``learn()`` method instead.

**Q: When should I use policies vs. just using modules directly?**

- Use **policies** when you want pluggable path selection in the orchestrator
- Use **modules directly** for custom simulations or legacy SDNController

**Q: Can I add a new policy type?**

Yes! Implement the ``ControlPolicy`` protocol:

.. code-block:: python

   from fusion.interfaces.control_policy import ControlPolicy

   class MyCustomPolicy:
       def select_action(self, request, options, network_state) -> int:
           # Your logic here
           return selected_index

       def update(self, request, action, reward) -> None:
           pass  # No-op for deployment

       def get_name(self) -> str:
           return "MyCustomPolicy"

Testing
=======

.. code-block:: bash

   # Run all policy tests
   pytest fusion/policies/tests/ -v

   # Run with coverage
   pytest --cov=fusion.policies fusion/policies/tests/

Related Documentation
=====================

- :ref:`pipelines-module` - Multi-step provisioning operations
- :ref:`modules-directory` - Algorithm implementations
- :ref:`core-module` - SDNOrchestrator that uses policies
- :ref:`interfaces-module` - ControlPolicy protocol definition
