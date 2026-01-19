===========================
Performance Optimization
===========================

Speed up simulations and optimize memory usage.

.. contents:: Table of Contents
   :local:
   :depth: 2

Quick Optimizations
===================

.. code-block:: ini

   [general_settings]
   # Reduce requests for faster iterations
   num_requests = 200

   # Use fewer traffic load points
   erlang_step = 500

   # Reduce iteration count
   max_iters = 2

   # Enable parallel execution
   thread_erlangs = True

Python Multiprocessing
======================

.. code-block:: python

   from fusion.sim.batch_runner import BatchRunner

   scenarios = [...]  # Your scenarios
   runner = BatchRunner(scenarios)

   # Run with 8 parallel workers
   results = runner.run_all(parallel=True, num_workers=8)

Memory Optimization
===================

.. code-block:: ini

   [general_settings]
   # Disable snapshot saving
   save_snapshots = False

   # Reduce save frequency
   save_step = 100

See Also
========

* :doc:`batch_simulations` - Parallel execution
