.. _failures-module:

===============
Failures Module
===============

.. warning::

   **Beta Status**: This module is in beta. It has not been fully validated
   in production simulations. See ``TODO.md`` for the development roadmap.

.. warning::

   **Integration is Incomplete**: The orchestrator path has partial failure support.
   Read the "Current Integration Status" section carefully before using failures
   in your experiments.

Quick Summary: What Works and What Doesn't
==========================================

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Capability
     - Legacy Path
     - Orchestrator Path
   * - Inject failures (link/node/SRLG/geo)
     - YES
     - YES
   * - Activate failures at scheduled time
     - YES
     - YES
   * - Repair failures at scheduled time
     - YES
     - YES
   * - Handle impact on allocated requests
     - YES
     - YES
   * - **Avoid failed paths during NEW routing**
     - **YES**
     - **NO** (gap!)
   * - RL policies receive failure info
     - N/A
     - YES (via DisasterState)

**Bottom line**: If you need new allocations to avoid failed links, use the legacy
path (``use_orchestrator=False``) until orchestrator integration is complete.

Overview
========

.. admonition:: At a Glance
   :class: tip

   :Purpose: Network failure injection for survivability testing
   :Location: ``fusion/modules/failures/``
   :Key Files: ``failure_manager.py``, ``failure_types.py``
   :Status: **Beta** - Partial orchestrator integration

The failures module injects network failures (failed links) into simulations to test
survivability and recovery mechanisms.

What This Module Does
---------------------

1. **Injects failures**: Schedule when links fail and when they're repaired
2. **Tracks active failures**: Know which links are currently down
3. **Checks path feasibility**: Determine if a path avoids failed links
4. **Handles failure impact**: When failures hit allocated requests, switch to backup or drop

Failure Types Supported
-----------------------

.. code-block:: text

   F1: Link Failure      - Single link fails
   F2: Node Failure      - Node + all adjacent links fail
   F3: SRLG Failure      - Multiple links sharing risk fail together
   F4: Geographic        - All links within hop-radius of center fail

Current Integration Status
==========================

This is the confusing part. Here's exactly what happens in each path:

Visual: How Failures Flow Through the System
--------------------------------------------

.. code-block:: text

   +===========================================================================+
   |                         SIMULATION ENGINE                                  |
   |   (fusion/core/simulation.py)                                             |
   |                                                                           |
   |   Owns FailureManager - works for BOTH paths                              |
   +===========================================================================+
            |
            |  At simulation start:
            |  _initialize_failure_manager() creates FailureManager
            |
            v
   +------------------+
   | FailureManager   |  <-- Owned by SimulationEngine, shared reference
   | - scheduled      |      to SDNController (legacy) but NOT to
   |   failures       |      SDNOrchestrator
   | - active failures|
   | - repair schedule|
   +--------+---------+
            |
            |
   =========|=====================================================================
            |     DURING SIMULATION (main event loop)
   =========|=====================================================================
            |
            |  For EACH time step, SimulationEngine calls:
            |
            v
   +------------------+     +------------------+     +------------------+
   | activate_        |     | _handle_failure_ |     | repair_          |
   | failures(time)   |---->| impact()         |---->| failures(time)   |
   |                  |     |                  |     |                  |
   | Moves scheduled  |     | For allocated    |     | Removes links    |
   | failures to      |     | requests hit by  |     | from active      |
   | active set       |     | new failures:    |     | failures         |
   +------------------+     | - Switch to      |     +------------------+
                            |   backup path    |
         WORKS FOR          | - Or drop request|          WORKS FOR
         BOTH PATHS         +------------------+          BOTH PATHS
                                    |
                             WORKS FOR
                             BOTH PATHS

   =========|=====================================================================
            |     DURING ROUTING (when new request arrives)
   =========|=====================================================================
            |
            +------------------+------------------+
            |                                     |
            v                                     v
   +------------------+                  +------------------+
   | LEGACY PATH      |                  | ORCHESTRATOR     |
   | use_orchestrator |                  | use_orchestrator |
   | = False          |                  | = True           |
   +------------------+                  +------------------+
            |                                     |
            v                                     v
   +------------------+                  +------------------+
   | SDNController    |                  | SDNOrchestrator  |
   | HAS reference to |                  | NO reference to  |
   | failure_manager  |                  | failure_manager  |
   +--------+---------+                  +--------+---------+
            |                                     |
            v                                     v
   +------------------+                  +------------------+
   | Before routing:  |                  | Routing happens  |
   | checks           |                  | WITHOUT checking |
   | is_path_feasible |                  | path feasibility |
   |                  |                  |                  |
   | AVOIDS FAILED    |                  | MAY ALLOCATE     |
   | PATHS            |                  | THROUGH FAILED   |
   +------------------+                  | LINKS!           |
            |                            +------------------+
            |                                     |
            v                                     v
        SAFE                              GAP - NOT SAFE
                                          (will be fixed in
                                           _handle_failure_impact
                                           but allocation already
                                           happened)

What This Means For Your Experiments
------------------------------------

**If using legacy path** (``use_orchestrator=False``):

- Failures work as expected
- New requests avoid failed paths
- Protected requests switch to backup when primary fails

**If using orchestrator path** (``use_orchestrator=True``):

- Failures are injected and tracked (this works)
- Already-allocated requests are handled when failures hit (this works)
- **BUT**: New requests may be allocated through failed links (BUG/GAP)
- The allocation will then immediately be impacted by ``_handle_failure_impact()``
- This is inefficient and may cause unexpected behavior

**If using RL with orchestrator**:

- RL policies CAN receive failure information via ``DisasterState``
- The RL adapter computes ``failure_mask`` features
- RL policies trained on survivability CAN make failure-aware decisions
- This is a workaround, not a fix for the core gap

The Two Failure-Related Concepts
================================

There are TWO different things that sound similar but are different:

.. code-block:: text

   +----------------------------------+----------------------------------+
   |         FailureManager           |        ProtectionPipeline        |
   |    (fusion/modules/failures/)    |     (fusion/pipelines/)          |
   +----------------------------------+----------------------------------+
   |                                  |                                  |
   |  SIMULATES failures happening    |  PREPARES for failures           |
   |                                  |                                  |
   |  "At t=100, link (0,1) fails"    |  "Allocate backup path now       |
   |  "At t=200, it's repaired"       |   in case primary fails later"   |
   |                                  |                                  |
   |  Answers: "Is this path          |  Answers: "What disjoint backup  |
   |  currently blocked?"             |  path should we provision?"      |
   |                                  |                                  |
   |  Used by: SimulationEngine,      |  Used by: SDNOrchestrator        |
   |  SDNController                   |  (orchestrator path only)        |
   |                                  |                                  |
   +----------------------------------+----------------------------------+
   |                                  |                                  |
   |  REACTIVE                        |  PROACTIVE                       |
   |  (respond to failures)           |  (prepare before failures)       |
   |                                  |                                  |
   +----------------------------------+----------------------------------+

**They should work together** but currently don't fully integrate:

- ProtectionPipeline provisions backup paths
- FailureManager should trigger switchover when failures hit
- The ``_handle_failure_impact()`` method does this, but only AFTER allocation

Future Intent
=============

The intended architecture (not yet implemented):

.. code-block:: text

   FUTURE STATE (v6.x):

   +------------------+
   | SDNOrchestrator  |
   |                  |
   | routing ---------|---> RoutingPipeline checks FailureManager
   | spectrum         |     before returning paths
   | protection       |
   | failure_manager -|---> Reference to FailureManager (NEW)
   +------------------+

   Options being considered:

   1. Pass FailureManager to orchestrator
      - Orchestrator checks is_path_feasible() during routing
      - Similar to how SDNController works

   2. Add failure info to NetworkState
      - NetworkState.failed_links property
      - Routing pipelines read from NetworkState

   3. Create FailuresPipeline
      - New pipeline stage that filters infeasible paths
      - Fits the pipeline architecture pattern

**No decision has been made yet.** This is tracked in the module's ``TODO.md``.

Module Components
=================

failure_manager.py
------------------

The main class for managing failures:

.. code-block:: python

   from fusion.modules.failures import FailureManager

   manager = FailureManager(engine_props, topology)

   # Schedule a failure
   event = manager.inject_failure(
       'link',
       t_fail=100.0,      # Fails at t=100
       t_repair=200.0,    # Repaired at t=200
       link_id=(0, 1)
   )

   # Later, at t=100:
   activated = manager.activate_failures(100.0)  # Returns [(0, 1)]

   # Check if path avoids failures:
   if manager.is_path_feasible([0, 1, 2]):  # False - uses failed link
       allocate(path)

   # At t=200:
   repaired = manager.repair_failures(200.0)  # Returns [(0, 1)]

failure_types.py
----------------

Implementation of F1-F4 failure types:

.. code-block:: python

   from fusion.modules.failures import fail_link, fail_node, fail_srlg, fail_geo

   # F1: Single link
   event = fail_link(topology, link_id=(0, 1), t_fail=10, t_repair=20)

   # F2: Node (all adjacent links)
   event = fail_node(topology, node_id=5, t_fail=10, t_repair=20)

   # F3: SRLG (multiple links)
   event = fail_srlg(topology, srlg_links=[(0,1), (2,3)], t_fail=10, t_repair=20)

   # F4: Geographic (hop radius)
   event = fail_geo(topology, center_node=5, hop_radius=2, t_fail=10, t_repair=20)

registry.py
-----------

Registry pattern for extensibility:

.. code-block:: python

   from fusion.modules.failures import register_failure_type, get_failure_handler

   # Get built-in handler
   handler = get_failure_handler('link')

   # Register custom handler
   def my_failure(topology, t_fail, t_repair, **kwargs):
       return {"failure_type": "custom", "failed_links": [...], ...}

   register_failure_type('custom', my_failure)

Development Guide
=================

Adding a New Failure Type
-------------------------

1. Add function to ``failure_types.py``
2. Register in ``registry.py``
3. Export in ``__init__.py``
4. Add tests

Running Survivability Experiments
---------------------------------

**For now, use legacy path:**

.. code-block:: python

   engine_props = {
       "use_orchestrator": False,  # Use legacy for reliable failure handling
       "failure_type": "geo",
       "failure_center_node": 5,
       "failure_hop_radius": 2,
       # ... other props
   }

**If you must use orchestrator with RL:**

.. code-block:: python

   from fusion.modules.rl.adapter import DisasterState, RLSimulationAdapter

   # Create disaster state for RL
   disaster_state = DisasterState(
       active=True,
       centroid=(x, y),
       radius=100.0,
       failed_links=frozenset([(0, 1), (2, 3)]),
   )

   # Pass to RL adapter for feature computation
   state = adapter.compute_state(request, network_state, disaster_state)

Testing
=======

.. code-block:: bash

   # Run failure module tests
   pytest fusion/modules/failures/tests/ -v

   # Run with coverage
   pytest fusion/modules/failures/tests/ -v --cov=fusion.modules.failures

Related Documentation
=====================

- :ref:`modules-directory` - Overview of all modules
- :ref:`core-module` - SimulationEngine and SDNController
- ``fusion/pipelines/protection_pipeline.py`` - 1+1 protection (different from this)
- ``fusion/modules/rl/adapter/rl_adapter.py`` - DisasterState for RL

Troubleshooting
===============

**"My orchestrator simulation allocates through failed links"**

This is the known gap. Use ``use_orchestrator=False`` for now.

**"Failures aren't being activated"**

Check that you called ``activate_failures(time)`` at the failure time.
The SimulationEngine does this automatically in the main loop.

**"Protected requests aren't switching to backup"**

Check ``_handle_failure_impact()`` in ``simulation.py``. This handles
switchover but requires protection to have been set up via ProtectionPipeline.
