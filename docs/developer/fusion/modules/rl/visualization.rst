.. _rl-visualization:

==============================
RL Visualization Plugin
==============================

.. note::

   **Status: BETA**

   This module is currently in **BETA** and is actively being developed.
   The API may evolve in future releases.

.. admonition:: At a Glance
   :class: tip

   :Purpose: Plugin extension providing RL-specific visualization capabilities
   :Location: ``fusion/modules/rl/visualization/``
   :Key Class: ``RLVisualizationPlugin``
   :Relationship: Extends ``fusion/visualization/`` core system via plugin interface

Understanding the Architecture
==============================

**This module is a plugin**, not a standalone visualization system. To understand
how it works, you need to understand FUSION's visualization architecture.

The Core Visualization System
-----------------------------

FUSION has a centralized visualization system at ``fusion/visualization/`` built
using Domain-Driven Design (DDD) principles:

.. code-block:: text

   fusion/visualization/                    <-- Core system
   |-- domain/                              <-- Domain entities, value objects
   |   |-- entities/metric.py              <-- MetricDefinition
   |   `-- strategies/processing_strategies.py
   |-- infrastructure/
   |   |-- renderers/base_renderer.py      <-- BaseRenderer interface
   |   `-- processors/                     <-- Data processors
   |-- plugins/
   |   |-- base_plugin.py                  <-- Plugin interface (BasePlugin)
   |   `-- plugin_registry.py              <-- Discovery and loading
   `-- application/
       `-- use_cases/generate_plot.py      <-- Main entry point

This core system handles:

- Data loading and version adaptation
- Metric aggregation and statistics
- Plot rendering infrastructure
- Caching and performance optimization

The RL Plugin Extension
-----------------------

This module extends the core system by implementing the ``BasePlugin`` interface:

.. code-block:: text

   fusion/modules/rl/visualization/         <-- This module (plugin)
   |-- rl_plugin.py                        <-- Implements BasePlugin
   |-- rl_metrics.py                       <-- RL-specific MetricDefinitions
   |-- rl_plots.py                         <-- RL-specific BaseRenderer subclasses
   `-- rl_processors.py                    <-- RL-specific processing strategies

**Key Point**: This module does not work independently. It registers its components
with the core visualization system, which then handles the actual plot generation.

How Plugin Loading Works
------------------------

.. code-block:: text

   1. Core system starts
          |
          v
   2. Plugin registry discovers plugins
          |
          v
   3. registry.load_plugin("rl") called
          |
          v
   4. RLVisualizationPlugin instantiated
          |
          v
   5. Plugin registers:
      - Metrics (episode_reward, q_values, etc.)
      - Plot types (reward_learning_curve, etc.)
      - Processors (RewardProcessingStrategy, etc.)
          |
          v
   6. User calls generate_plot(plot_type="reward_learning_curve")
          |
          v
   7. Core system uses registered RL components to render plot

Usage
=====

Loading the Plugin
------------------

The RL visualization plugin must be loaded before use:

.. code-block:: python

   from fusion.visualization.plugins import get_global_registry

   # Discover and load plugins
   registry = get_global_registry()
   registry.discover_plugins()
   registry.load_plugin("rl")

   # Now RL plot types are available
   print(registry.get_available_plot_types())
   # ['blocking_probability', 'reward_learning_curve', 'q_value_heatmap', ...]

Generating RL Plots
-------------------

Once loaded, use the standard visualization API:

.. code-block:: python

   from fusion.visualization.application.use_cases.generate_plot import generate_plot

   # Generate a reward learning curve
   result = generate_plot(
       config_path="my_rl_experiment.yml",
       plot_type="reward_learning_curve",
       output_path="plots/learning_curve.png",
   )

   # Generate a convergence analysis
   result = generate_plot(
       config_path="my_rl_experiment.yml",
       plot_type="convergence_plot",
       output_path="plots/convergence.png",
   )

Available Plot Types
====================

The RL plugin registers four specialized plot types:

reward_learning_curve
---------------------

Learning curve showing episode rewards over training with smoothing and
confidence intervals.

.. code-block:: python

   # Default configuration
   {
       "window_size": 100,        # Smoothing window
       "confidence_level": 0.95,  # CI level
       "show_ci": True,           # Show confidence bands
   }

**Output**: Line plot with mean reward and shaded confidence intervals.

q_value_heatmap
---------------

Heatmap visualization of Q-values across states and actions.

.. code-block:: python

   # Default configuration
   {
       "colormap": "viridis",
       "annotate": False,
   }

**Output**: Heatmap showing Q-value magnitude by state/action.

convergence_plot
----------------

Training convergence analysis showing when metrics stabilize.

.. code-block:: python

   # Default configuration
   {
       "window_size": 100,
       "threshold": 0.01,          # Relative change threshold
       "show_convergence_point": True,
   }

**Output**: Two-panel plot with metric progression and convergence statistics.

rl_dashboard
------------

Comprehensive multi-metric dashboard for RL training analysis.

.. code-block:: python

   # Default configuration
   {
       "layout": "3x2",
       "window_size": 100,
   }

**Output**: 6-panel dashboard with rewards, losses, entropy, and Q-values.

Registered Metrics
==================

The plugin registers these RL-specific metrics:

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Metric Name
     - Data Type
     - Description
   * - ``episode_reward``
     - FLOAT
     - Total reward accumulated in an episode
   * - ``episode_reward_mean``
     - FLOAT
     - Moving average of episode rewards
   * - ``td_error``
     - ARRAY
     - Temporal difference prediction errors
   * - ``q_values``
     - ARRAY
     - Action-value function estimates
   * - ``policy_entropy``
     - FLOAT
     - Entropy of the policy distribution
   * - ``policy_loss``
     - FLOAT
     - Policy gradient loss
   * - ``value_loss``
     - FLOAT
     - Value function loss
   * - ``epsilon``
     - FLOAT
     - Epsilon-greedy exploration rate
   * - ``learning_rate``
     - FLOAT
     - Current learning rate

Processing Strategies
=====================

The plugin provides three specialized processing strategies:

RewardProcessingStrategy
------------------------

Processes episode rewards with smoothing and statistical aggregation.

**Features:**

- Moving average smoothing via ``scipy.ndimage.uniform_filter1d``
- Confidence interval calculation
- Multi-seed aggregation

.. code-block:: python

   from fusion.modules.rl.visualization.rl_processors import RewardProcessingStrategy

   processor = RewardProcessingStrategy(
       window_size=100,
       confidence_level=0.95,
   )

QValueProcessingStrategy
------------------------

Processes Q-value data for heatmap visualization.

**Features:**

- Action-wise aggregation
- Normalization for visualization
- State grouping

ConvergenceDetectionStrategy
----------------------------

Detects when training metrics have converged.

**Algorithm:**

1. Compute mean in sliding window
2. Compare adjacent windows
3. Flag convergence when relative change < threshold

.. code-block:: python

   from fusion.modules.rl.visualization.rl_processors import ConvergenceDetectionStrategy

   detector = ConvergenceDetectionStrategy(
       window_size=100,
       threshold=0.01,  # 1% relative change
   )

Custom Renderers
================

The plugin provides four renderer classes extending ``BaseRenderer``:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Renderer
     - Description
   * - ``RewardLearningCurveRenderer``
     - Line plots with confidence bands for reward curves
   * - ``QValueHeatmapRenderer``
     - Seaborn heatmaps for Q-value visualization
   * - ``ConvergencePlotRenderer``
     - Dual-panel convergence analysis plots
   * - ``MultiMetricDashboardRenderer``
     - 6-panel comprehensive training dashboard

All renderers support PNG, PDF, SVG, and JPG output formats.

File Reference
==============

.. code-block:: text

   fusion/modules/rl/visualization/
   |-- __init__.py          # Exports RLVisualizationPlugin
   |-- rl_plugin.py         # Main plugin class
   |-- rl_metrics.py        # Metric definitions
   |-- rl_plots.py          # Plot renderers
   `-- rl_processors.py     # Processing strategies

**Public API:**

.. code-block:: python

   from fusion.modules.rl.visualization import RLVisualizationPlugin

   # Or access components directly
   from fusion.modules.rl.visualization.rl_metrics import get_rl_metrics
   from fusion.modules.rl.visualization.rl_plots import (
       RewardLearningCurveRenderer,
       QValueHeatmapRenderer,
       ConvergencePlotRenderer,
       MultiMetricDashboardRenderer,
   )
   from fusion.modules.rl.visualization.rl_processors import (
       RewardProcessingStrategy,
       QValueProcessingStrategy,
       ConvergenceDetectionStrategy,
   )

Dependencies
============

This plugin requires additional packages beyond the core visualization system:

- **scipy**: For smoothing filters (``uniform_filter1d``)
- **seaborn**: For heatmap visualization

These are checked at plugin load time via ``_check_dependencies()``.

Related Documentation
=====================

- :ref:`rl-module` - Parent RL module documentation
- :ref:`rl-utils` - Custom callbacks for training metrics collection

.. seealso::

   - Core visualization: ``fusion/visualization/README.md``
   - Plugin interface: ``fusion/visualization/plugins/base_plugin.py``
   - `Matplotlib Documentation <https://matplotlib.org/>`_
   - `Seaborn Documentation <https://seaborn.pydata.org/>`_
