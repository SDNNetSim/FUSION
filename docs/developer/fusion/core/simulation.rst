.. _core-simulation:

================
Simulation Guide
================

This document explains how the ``SimulationEngine`` works, from initialization
to completion.

Overview
========

The ``SimulationEngine`` is the main entry point for running FUSION simulations.
It coordinates:

- Network topology creation
- Request generation (Poisson arrival process)
- Request processing (via SDNController or SDNOrchestrator)
- Metrics collection (via SimStats)
- Results persistence (via StatsPersistence)

.. code-block:: text

   SimulationEngine
        |
        +-- create_topology()     # Initialize network
        |
        +-- run()                 # Main simulation loop
        |    |
        |    +-- generate_requests()
        |    |
        |    +-- for each iteration:
        |    |    |
        |    |    +-- init_iter_stats()
        |    |    |
        |    |    +-- for each request:
        |    |    |    |
        |    |    |    +-- SDNController.allocate()  # Legacy
        |    |    |    |   OR
        |    |    |    +-- SDNOrchestrator.handle_arrival()  # v6.0+
        |    |    |    |
        |    |    |    +-- SimStats.iter_update()
        |    |    |
        |    |    +-- calculate_blocking_statistics()
        |    |    |
        |    |    +-- check confidence interval
        |    |
        |    +-- save_stats()
        |
        +-- cleanup()

SimulationEngine Class
======================

:Location: ``fusion/core/simulation.py``
:Size: 2,209 lines (needs refactoring)

Initialization
--------------

.. code-block:: python

   from fusion.core import SimulationEngine

   engine_props = {
       # Simulation control
       'max_iters': 10,
       'num_requests': 1000,
       'erlang': 300,
       'holding_time': 3600,

       # Network configuration
       'network': 'NSFNet',
       'cores_per_link': 7,
       'c_band': 320,

       # Algorithm selection
       'route_method': 'k_shortest_path',
       'allocation_method': 'first_fit',
       'k_paths': 3,

       # Architecture selection
       'use_orchestrator': True,  # or False for legacy
   }

   engine = SimulationEngine(engine_props)

Key Properties
--------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Property
     - Description
   * - ``engine_props``
     - Configuration dictionary
   * - ``topology``
     - NetworkX graph of the network
   * - ``sdn_controller``
     - SDNController instance (legacy path)
   * - ``orchestrator``
     - SDNOrchestrator instance (v6.0+ path)
   * - ``stats``
     - SimStats instance for metrics
   * - ``network_spectrum_dict``
     - Current spectrum allocation state

Topology Creation
=================

The ``create_topology()`` method initializes the network:

.. code-block:: python

   engine.create_topology()

**What happens:**

1. Load network graph from file (e.g., ``NSFNet.json``)
2. Initialize spectrum dictionary for each link
3. Set up lightpath status dictionary
4. Create modulation format mappings
5. Initialize failure manager (if survivability enabled)

**Network spectrum dictionary structure:**

.. code-block:: python

   # network_spectrum_dict[(src, dst)][core][slot] = request_id or 0
   network_spectrum_dict = {
       ("0", "1"): {
           0: np.zeros(320, dtype=int),  # Core 0
           1: np.zeros(320, dtype=int),  # Core 1
           # ... for each core
       },
       ("0", "2"): {...},
       # ... for each link
   }

Request Generation
==================

Requests are generated using a Poisson arrival process:

.. code-block:: python

   from fusion.core.request import generate_simulation_requests

   requests = generate_simulation_requests(
       num_requests=1000,
       erlang=300,
       holding_time=3600,
       topology=topology,
       bandwidth_distribution={"100": 0.5, "200": 0.3, "400": 0.2},
       seed=42,
   )

**Request structure:**

.. code-block:: python

   request = {
       'request_id': 1,
       'source': '0',
       'destination': '5',
       'bandwidth': 100,
       'arrive': 0.5,      # Arrival time
       'depart': 3600.5,   # Departure time
       'request_type': 'arrival',  # or 'departure'
   }

**Arrival process:**

- Inter-arrival times: Exponential distribution with rate = erlang / holding_time
- Holding times: Exponential distribution with mean = holding_time
- Node pairs: Uniform random selection
- Bandwidth: According to bandwidth_distribution

Main Simulation Loop
====================

The ``run()`` method executes the main simulation:

.. code-block:: python

   completed_iterations = engine.run()

**Flow:**

.. code-block:: text

   for iteration in range(max_iters):
       |
       +-- Initialize iteration statistics
       |
       +-- Reset network state (spectrum, lightpaths)
       |
       +-- for request in sorted_requests:
       |    |
       |    +-- if request_type == 'arrival':
       |    |    |
       |    |    +-- [Legacy] sdn_controller.allocate()
       |    |    |   OR
       |    |    +-- [Orchestrator] orchestrator.handle_arrival()
       |    |    |
       |    |    +-- stats.iter_update()
       |    |
       |    +-- elif request_type == 'departure':
       |         |
       |         +-- Release spectrum
       |         +-- Remove lightpath
       |
       +-- Calculate blocking statistics
       |
       +-- Check confidence interval
       |    |
       |    +-- If converged: break
       |
       +-- Save iteration results

Feature Flag Resolution
-----------------------

The simulation chooses between legacy and orchestrator based on:

1. Environment variable: ``FUSION_USE_ORCHESTRATOR``
2. Config parameter: ``use_orchestrator``
3. Default: ``False`` (legacy)

.. code-block:: python

   def _resolve_use_orchestrator(self) -> bool:
       """Resolve which architecture to use."""
       # 1. Check environment variable (highest priority)
       env_value = os.environ.get('FUSION_USE_ORCHESTRATOR')
       if env_value is not None:
           return env_value.lower() in ('1', 'true', 'yes')

       # 2. Check config parameter
       return self.engine_props.get('use_orchestrator', False)

Request Processing
==================

Legacy Path (SDNController)
---------------------------

.. code-block:: python

   # In simulation loop
   if not use_orchestrator:
       # Set up request in sdn_props
       self.sdn_props.source = request['source']
       self.sdn_props.destination = request['destination']
       self.sdn_props.bandwidth = request['bandwidth']

       # Process request
       self.sdn_controller.allocate()

       # Check result
       if self.sdn_props.was_routed:
           # Update network state
           self._update_network_after_allocation()
       else:
           # Track blocking
           self._handle_blocked_request()

Orchestrator Path (SDNOrchestrator)
-----------------------------------

.. code-block:: python

   # In simulation loop
   if use_orchestrator:
       # Create Request object
       request_obj = Request(
           request_id=request['request_id'],
           source=request['source'],
           destination=request['destination'],
           bandwidth_gbps=request['bandwidth'],
           arrive_time=request['arrive'],
           depart_time=request['depart'],
       )

       # Create network state snapshot
       network_state = NetworkState(
           topology=self.topology,
           network_spectrum_dict=self.network_spectrum_dict,
           lightpath_status_dict=self.lightpath_status_dict,
       )

       # Process request
       result = self.orchestrator.handle_arrival(request_obj, network_state)

       # Handle result
       if result.success:
           self._apply_allocation(result, request_obj)
       else:
           self._handle_blocked_request(result.block_reason)

Metrics Collection
==================

After each request, metrics are updated:

.. code-block:: python

   # Update statistics
   self.stats.iter_update(
       request_data=request,
       sdn_data=self.sdn_props,  # or result for orchestrator
       network_spectrum_dict=self.network_spectrum_dict,
   )

At the end of each iteration:

.. code-block:: python

   # Calculate iteration statistics
   self.stats.calculate_blocking_statistics()
   self.stats.finalize_iteration_statistics()

   # Check convergence
   if self.stats.calculate_confidence_interval():
       logger.info("Confidence interval reached, stopping early")
       break

Results Persistence
===================

Results are saved at the end of simulation:

.. code-block:: python

   # Create persistence handler
   persistence = StatsPersistence(self.engine_props, self.sim_info)

   # Prepare statistics
   blocking_stats = {
       'block_mean': self.stats.blocking_mean,
       'block_variance': self.stats.blocking_variance,
       'block_ci': self.stats.confidence_interval,
       'iteration': current_iteration,
   }

   # Save to file
   persistence.save_stats(
       stats_dict=self.stats.get_stats_dict(),
       stats_props=self.stats.stats_props,
       blocking_stats=blocking_stats,
   )

Multiprocessing Support
=======================

.. note::

   Multiprocessing support is currently being worked on in v6.0 and is not
   fully functional yet. The information below describes the intended design.

FUSION supports parallel simulation across multiple processes:

.. code-block:: python

   from fusion.cli.run_sim import run_simulation

   # Configuration with Erlang range
   engine_props = {
       'erlang_start': 300,
       'erlang_stop': 600,
       'erlang_step': 100,
       'thread_erlangs': True,  # Enable parallel execution
       # ... other config
   }

   # Run parallel simulations
   run_simulation(engine_props)

**How it works:**

1. Main process creates worker processes
2. Each process can run multiple Erlang values (not one per process)
3. Each process saves results independently to separate files
4. Results are not automatically aggregated - each process outputs its own files

Failure Manager Integration
===========================

.. note::

   Failure manager integration is currently being worked on in v6.0 and is not
   fully functional yet. The information below describes the intended design.

For survivability experiments, the simulation integrates with FailureManager:

.. code-block:: python

   # Enable failures in config
   engine_props['failure_enabled'] = True
   engine_props['failure_type'] = 'link'  # link, node, srlg, geo
   engine_props['t_fail_arrival_index'] = 500  # Fail after 500 arrivals
   engine_props['t_repair_after_arrivals'] = 100  # Repair after 100 more

**Failure handling flow:**

.. code-block:: text

   Request arrival
        |
        +-- Check if failure should trigger
        |    |
        |    +-- If t_fail reached: inject_failure()
        |
        +-- Process request (with failed links removed)
        |
        +-- Check if repair should trigger
             |
             +-- If t_repair reached: repair_failure()

Common Configuration Options
============================

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Option
     - Default
     - Description
   * - ``max_iters``
     - 10
     - Maximum simulation iterations
   * - ``num_requests``
     - 1000
     - Requests per iteration
   * - ``erlang``
     - 300
     - Traffic load in Erlangs
   * - ``holding_time``
     - 3600
     - Average request duration (seconds)
   * - ``network``
     - "NSFNet"
     - Network topology name
   * - ``cores_per_link``
     - 7
     - Fiber cores per link
   * - ``c_band``
     - 320
     - C-band spectrum slots
   * - ``k_paths``
     - 3
     - Number of candidate paths
   * - ``print_step``
     - 0
     - Progress reporting interval
   * - ``save_snapshots``
     - False
     - Save periodic state snapshots
   * - ``ci_target``
     - 5.0
     - Confidence interval target (%)

Debugging Tips
==============

**Enable verbose output:**

.. code-block:: python

   engine_props['print_step'] = 10  # Print every 10 requests

**Save snapshots for analysis:**

.. code-block:: python

   engine_props['save_snapshots'] = True
   engine_props['snapshot_interval'] = 100  # Every 100 requests

**Use deterministic seed:**

.. code-block:: python

   engine_props['seed'] = 42  # Reproducible results

**Run single iteration:**

.. code-block:: python

   engine_props['max_iters'] = 1
   engine_props['num_requests'] = 100  # Small for quick testing

See Also
========

- :doc:`architecture` - Legacy vs. orchestrator
- :doc:`orchestrator` - Pipeline flow details
- :doc:`metrics` - Statistics collection
- :ref:`configs-module` - Configuration system
