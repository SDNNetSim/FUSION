====================
Batch Simulations
====================

Run multiple simulations in parallel for parameter sweeps and comparisons.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
========

Efficiently run multiple simulation scenarios in parallel using FUSION's
batch execution framework.

Quick Start
===========

.. code-block:: python

   from fusion.sim.batch_runner import BatchRunner

   # Define scenarios
   scenarios = [
       {'topology': 'NSFNet', 'c_band': 160},
       {'topology': 'NSFNet', 'c_band': 320},
       {'topology': 'NSFNet', 'c_band': 640},
   ]

   # Run batch
   runner = BatchRunner(scenarios)
   results = runner.run_all(parallel=True, num_workers=4)

Parameter Sweep
===============

.. code-block:: python

   from fusion.sim.batch_runner import parameter_sweep

   # Sweep traffic loads and routing algorithms
   results = parameter_sweep(
       base_config='config.ini',
       parameters={
           'erlang_start': [100, 300, 500],
           'route_method': ['k_shortest_path', 'fragmentation_aware_ksp']
       }
   )

See Also
========

* :doc:`basic_simulation` - Single simulation guide
* :doc:`performance_optimization` - Speed up execution
