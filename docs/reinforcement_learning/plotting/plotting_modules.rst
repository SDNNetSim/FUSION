Plotting
==============
The plotting module provides tools for visualizing simulation results — from blocking probability
and resource usage to modulation and learning dynamics. To generate plots, users interface with
``plot_runner.py``, which loads plot instructions from a YAML configuration file.

Usage
-----

To generate plots, run:

.. code-block:: bash

   python -m reinforcement_learning.plotting.plot_runner

This will parse the provided ``plot_config.yml`` file and dispatch the appropriate plotting routines.

YAML Configuration
------------------

The ``plot_config.yml`` file specifies what plots to generate and what data to include. A minimal example:

.. code-block:: yaml

   network: "dt_network"
   dates:
     - "0606"
     - "0611"
   plots:
     - blocking
   algorithms:
     - ppo
   runs:
     drl: true
     non_drl: true
   observation_spaces:
     - obs_7

Each field controls a different aspect of plot behavior:

- ``network``: Specifies which network's results to process.
- ``dates``: Specifies which experiment folders (by date) to search for logs.
- ``plots``: A list of plot types to generate (must match keys in the registry).
- ``algorithms``: List of algorithms to include in the comparison.
- ``runs``: Flags for including DRL and non-DRL baselines.
- ``observation_spaces``: Filters which obs space variants to include in plots.

Behind the scenes, `plot_runner.py` coordinates data loading, preprocessing, and plotting via a central registry.
This makes it easy to add new plots or modify the pipeline in a modular fashion.

.. toctree::

    blocking
    bw_block
    link_data
    loaders
    memory_usage
    mod_usage
    path_index
    plot_runner
    processors
    registry
    resource_stats
    rewards
    sim_times
    state_values
