.. _data-input-output:

==============
input / output
==============

Overview
========

.. admonition:: At a Glance
   :class: tip

   :Purpose: Stores simulation configuration inputs and result outputs
   :Location: ``data/input/`` and ``data/output/``
   :Format: JSON files

The ``input/`` and ``output/`` directories store simulation data organized by
network, date, and time. This structure supports reproducibility (you can see
exactly what configuration produced which results) and enables parallel/cluster
execution.

Directory Structure
===================

Both directories follow a similar hierarchical structure:

.. code-block:: text

   input/
   └── <Network>/
       └── <MMDD>/
           └── <HH_MM_SS_microseconds>/
               ├── bw_info_s1.json
               ├── sim_input_s1.json
               ├── bw_info_s2.json      # If multi-processing
               └── sim_input_s2.json

   output/
   └── <Network>/
       └── <MMDD>/
           └── <HH_MM_SS_microseconds>/
               └── s1/                   # Process directory
                   ├── 300.0_erlang.json
                   ├── 600.0_erlang.json
                   └── ...

Path Components
---------------

- **Network**: The network topology name (e.g., ``NSFNet``, ``USbackbone60``)
- **MMDD**: Date in month-day format (e.g., ``0113`` for January 13) - FUSION was born in the United States :)
- **HH_MM_SS_microseconds**: Time down to microseconds (e.g., ``11_56_57_213820``)
- **s#**: Process number for multi-processing support

.. note::

   **Why microseconds?** FUSION supports the SLURM job scheduler for cluster
   computing. When submitting many parallel jobs, they can start within
   microseconds of each other. The microsecond precision ensures unique
   directory paths for each job.

Process Numbers (s#)
--------------------

The ``s#`` suffix (e.g., ``s1``, ``s2``) indicates the process number. FUSION
supports multi-processing to run simulations in parallel. Each process gets its
own numbered identifier to avoid file conflicts.

See the configuration documentation for details on enabling multi-processing.

Input Files
===========

bw_info_s#.json
---------------

Stores the modulation format configuration used for the simulation. This is
essentially the resolved modulation formats from your ``mod_assumption`` setting.

.. code-block:: text

   {
     "100": {
       "QPSK": {
         "max_length": 5540,
         "slots_needed": 4
       },
       "16-QAM": {
         "max_length": 2375,
         "slots_needed": 2
       },
       "64-QAM": {
         "max_length": 916,
         "slots_needed": 2
       }
     },
     "400": { ... }
   }

sim_input_s#.json
-----------------

Stores the complete simulation configuration - essentially your ``.ini`` file
resolved to actual values. This allows you to know exactly what parameters were
used for any simulation run.

Key fields include:

- **Traffic parameters**: ``erlang_start``, ``erlang_stop``, ``erlang_step``,
  ``holding_time``, ``num_requests``
- **Network configuration**: ``network``, ``cores_per_link``, ``c_band``
- **Algorithm settings**: ``route_method``, ``allocation_method``, ``k_paths``
- **Topology info**: Complete node and link information with fiber properties
- **Timestamps**: ``date``, ``sim_start``, ``thread_num``

Output Files
============

<Erlang>_erlang.json
--------------------

Each Erlang value produces its own output file (e.g., ``300.0_erlang.json``).
These files contain all simulation results for that traffic load.

Erlang Calculation
~~~~~~~~~~~~~~~~~~

Erlang is a measure of traffic intensity, calculated as:

.. code-block:: text

   arrival_rate = (cores_per_link * erlang) / holding_time

The simulation iterates through Erlang values from ``erlang_start`` to
``erlang_stop`` in increments of ``erlang_step``.

Configuration example:

.. code-block:: ini

   [general_settings]
   holding_time = 0.2
   erlang_start = 300
   erlang_stop = 1200
   erlang_step = 300

This produces output files: ``300.0_erlang.json``, ``600.0_erlang.json``,
``900.0_erlang.json``, ``1200.0_erlang.json``.

Output File Contents
~~~~~~~~~~~~~~~~~~~~

Each Erlang output file contains:

**Iteration Statistics** (``iter_stats``):

Per-iteration results including:

- ``cores_dict``: Usage count per core
- ``weights_dict``: Path weight statistics (mean, std, min, max) per bandwidth
  and modulation format
- ``hops_dict``: Hop count statistics per bandwidth and modulation
- ``block_reasons_dict``: Breakdown of blocking reasons
- ``mod_format_dict``: Modulation format usage counts
- ``path_index_list``: Which k-th path was selected for each request
- ``sim_block_list``: Blocking probability per iteration
- ``transponders_list``: Average transponders used

**Aggregate Statistics**:

- ``block_mean``: Mean blocking probability across iterations
- ``block_variance``: Variance in blocking probability
- ``block_ci``: Confidence interval for blocking probability
- ``link_usage``: Per-link spectrum utilization

**Snapshots** (if enabled):

Periodic snapshots of network state during simulation:

- ``request_number``: Request count at snapshot
- ``blocked_count``: Blocked requests at snapshot
- ``spectrum_utilization``: Current spectrum utilization

Example Output Structure
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   {
     "iter_stats": {                              // Per-iteration results
       "0": {                                     // Iteration 0
         "cores_dict": {"0": 468, "1": 109, "2": 0},  // Requests per core
         "weights_dict": {                        // Path weight stats by BW/mod
           "100": {                               // 100 Gbps
             "QPSK": {"mean": 3119.3, "std": 451.4, "min": 2400, "max": 3800}
           }
         },
         "sim_block_list": [0.024],               // Blocking prob this iteration
         "block_reasons_dict": {"congestion": 0.8, "distance": 0.2}  // Why blocked
       },
       "1": { ... }                               // Iteration 1, etc.
     },
     "block_mean": 0.025,                         // Mean blocking across iterations
     "block_variance": 0.0001,                    // Variance in blocking
     "block_ci": 0.002,                           // Confidence interval
     "link_usage": { ... }                        // Per-link utilization
   }

Connection to Simulation
========================

Input files are created by ``fusion/sim/input_setup.py``:

- ``save_input()`` writes both ``bw_info_s#.json`` and ``sim_input_s#.json``
- Called at the start of simulation before the first Erlang iteration

Output files are created by ``fusion/core/persistence.py``:

- ``SimulationPersistence.save()`` writes the Erlang output files
- Called after each Erlang value completes all iterations

The timestamp and directory structure is generated by
``fusion/sim/utils/simulation.py`` which ensures unique paths even for
concurrent cluster jobs.
