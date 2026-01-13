===============
Troubleshooting
===============

Common issues and solutions.

.. contents:: Table of Contents
   :local:
   :depth: 2

Installation Issues
===================

ModuleNotFoundError: No module named 'fusion'
----------------------------------------------

**Cause**: FUSION not installed or not in Python path.

**Solution**:

.. code-block:: bash

   cd /path/to/FUSION
   pip install -e .

Numpy Architecture Mismatch
----------------------------

**Error**: ``mach-o file, but is an incompatible architecture (have 'arm64', need 'x86_64')``

**Solution**:

.. code-block:: bash

   pip uninstall numpy
   pip install numpy --no-cache-dir

PyTorch Geometric Installation Fails
-------------------------------------

**Solution**: Install dependencies first:

.. code-block:: bash

   pip install torch
   pip install torch-geometric

Configuration Errors
====================

"Invalid configuration" error
------------------------------

**Cause**: Missing required fields or invalid values.

**Solution**: Use a template as starting point:

.. code-block:: bash

   cp fusion/configs/templates/minimal.ini my_config.ini

"Network not found" error
--------------------------

**Cause**: Invalid topology name or file path.

**Solution**: Use built-in topology or valid JSON file:

.. code-block:: ini

   network = NSFNet  # Built-in
   # or
   network = /full/path/to/topology.json  # Custom

Simulation Issues
=================

All Requests Blocked (100% blocking)
-------------------------------------

**Causes**:

1. Insufficient spectrum
2. Traffic load too high
3. Routing algorithm finds no paths

**Solutions**:

.. code-block:: ini

   # Increase spectrum
   c_band = 640  # Double from 320

   # Reduce traffic
   erlang_start = 100  # Lower starting load

   # Check routing
   k_paths = 5  # More path candidates

No Blocking (0% blocking)
--------------------------

**Cause**: Network over-provisioned.

**Solution**:

.. code-block:: ini

   # Reduce spectrum
   c_band = 160

   # Increase traffic
   erlang_start = 800

Simulation Runs Forever
------------------------

**Cause**: Too many requests or iterations.

**Solution**:

.. code-block:: ini

   num_requests = 100  # Reduce from 500
   max_iters = 2       # Reduce from 4

RL Training Issues
==================

Training Not Converging
-----------------------

**Solutions**:

1. Adjust learning rate:

   .. code-block:: ini

      alpha_start = 0.0001  # Lower learning rate

2. Increase training time:

   .. code-block:: ini

      total_timesteps = 500000  # More training

3. Simplify observation space:

   .. code-block:: ini

      obs_space = obs_1  # Simpler observations

CUDA Out of Memory
------------------

**Solutions**:

.. code-block:: ini

   # Use CPU
   device = cpu

   # Or reduce batch size (if using DQN/PPO)
   batch_size = 32

Data & Output Issues
====================

Results File Not Found
-----------------------

**Cause**: Simulation failed or wrong output directory.

**Solution**: Check simulation logs and output directory:

.. code-block:: bash

   ls -la output/

JSON Decode Error
-----------------

**Cause**: Corrupted results file.

**Solution**: Re-run simulation or check file manually:

.. code-block:: bash

   cat output/simulation_results.json | python -m json.tool

Performance Issues
==================

Simulation Too Slow
-------------------

**Solutions**:

1. Enable parallel execution:

   .. code-block:: ini

      thread_erlangs = True

2. Reduce problem size:

   .. code-block:: ini

      num_requests = 200
      erlang_step = 500

3. Use simpler algorithms:

   .. code-block:: ini

      route_method = k_shortest_path  # Not ML/RL

High Memory Usage
-----------------

**Solutions**:

.. code-block:: ini

   # Disable snapshots
   save_snapshots = False

   # Reduce save frequency
   save_step = 100

Custom Topology Issues
======================

"Graph not connected" error
----------------------------

**Cause**: Disconnected topology components.

**Solution**: Validate topology:

.. code-block:: python

   import networkx as nx
   import json

   with open('topology.json') as f:
       data = json.load(f)

   G = nx.Graph()
   for node in data["nodes"]:
       G.add_node(node)
   for link in data["links"].values():
       G.add_edge(link["source"], link["destination"])

   if not nx.is_connected(G):
       print("Disconnected! Components:", list(nx.connected_components(G)))

Invalid JSON Topology
---------------------

**Solution**: Validate JSON syntax:

.. code-block:: bash

   python -m json.tool topology.json

Getting Help
============

If your issue isn't listed here:

1. Check :doc:`faq` for common questions
2. Review :doc:`../examples/index` for usage examples
3. Search `GitHub Issues <https://github.com/SDNNetSim/FUSION/issues>`_
4. Ask on `GitHub Discussions <https://github.com/SDNNetSim/FUSION/discussions>`_
5. Open a new issue with:
   - FUSION version
   - Python version
   - Error message
   - Minimal reproduction steps
