==============
Visualization
==============

This guide covers plotting and visualizing simulation results, training metrics, and network analysis in FUSION.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
========

FUSION provides comprehensive visualization tools for:

- **Simulation Results**: Blocking probability, spectrum utilization, bandwidth blocking
- **Network Analysis**: Topology visualization, link utilization heatmaps
- **ML/RL Training**: Learning curves, feature importance, policy behavior
- **Comparative Analysis**: Multi-scenario comparisons, algorithm benchmarking

All plots are publication-ready with customizable styling, export to multiple formats, and interactive options.

Quick Start
===========

Basic Plotting
--------------

Plot simulation results from JSON/CSV output:

.. code-block:: python

   from fusion.visualization.plot_stats import PlotStats

   # Load results
   sims_info = {
       'networks_matrix': [['NSFNet']],
       'dates_matrix': [['2025-01-01']],
       'times_matrix': [['14-30-00']],
       'file_paths': ['data/results/nsfnet_results.json']
   }

   # Create plotter
   plotter = PlotStats(sims_info_dict=sims_info)

   # Generate blocking probability plot
   plotter.plot_blocking_probability(save_path='plots/blocking_prob.png')

Command-Line Plotting
----------------------

Generate plots directly from CLI:

.. code-block:: bash

   fusion-plot --results data/results/simulation.json \
               --plot_type blocking \
               --save plots/blocking_probability.png

Simulation Results Visualization
=================================

Blocking Probability
--------------------

Plot blocking probability vs traffic load (Erlang):

.. code-block:: python

   from fusion.visualization.plot_stats import PlotStats

   plotter = PlotStats(sims_info_dict)

   # Standard BP plot
   plotter.plot_blocking_probability(
       title='Blocking Probability vs Load',
       y_label='Blocking Probability',
       x_label='Offered Load (Erlangs)',
       save_path='plots/bp.png'
   )

Features:
   - Logarithmic y-axis for wide probability ranges
   - Multiple algorithms/scenarios on same plot
   - Confidence intervals from multiple iterations
   - Customizable markers and line styles

Bandwidth Blocking
------------------

Visualize blocked bandwidth (Gbps):

.. code-block:: python

   plotter.plot_bandwidth_blocking(
       save_path='plots/bandwidth_blocking.png'
   )

Useful for understanding which traffic types (high vs low bandwidth) are blocked.

Spectrum Utilization
--------------------

Track spectrum efficiency over time or load:

.. code-block:: python

   plotter.plot_spectrum_utilization(
       metric='average',  # or 'peak', 'variance'
       save_path='plots/spectrum_util.png'
   )

Metrics include:
   - Average utilization across all links
   - Peak utilization (most congested link)
   - Utilization variance (load balancing metric)

Iteration-Level Statistics
---------------------------

Plot metrics per iteration to show convergence:

.. code-block:: python

   plotter.plot_iter_stats(
       y_vals_list=['blocking_prob', 'bandwidth_blocking'],
       erlang=400,
       save_path='plots/iter_convergence.png'
   )

Helps verify simulation has reached steady state.

Network Visualization
=====================

Topology Plots
--------------

Visualize network topology with current state:

.. code-block:: python

   from fusion.visualization.network_viz import plot_topology

   plot_topology(
       topology=topology_graph,
       node_colors=utilization_per_node,
       edge_widths=link_utilization,
       save_path='plots/network_topology.png'
   )

**Color Coding**:
   - Nodes: Utilization levels (green → yellow → red)
   - Edges: Link congestion intensity

**Interactive Mode**:
   Use plotly for interactive exploration

   .. code-block:: python

      plot_topology(
          topology=topology_graph,
          interactive=True,
          save_path='plots/network_interactive.html'
      )

Spectrum Heatmaps
-----------------

Visualize spectrum allocation across links:

.. code-block:: python

   from fusion.visualization.spectrum_viz import plot_spectrum_heatmap

   plot_spectrum_heatmap(
       network_spectrum_dict=spectrum_state,
       link_ids=[(0,1), (1,2), (2,3)],
       save_path='plots/spectrum_heatmap.png'
   )

Shows:
   - Occupied vs free slots (color-coded)
   - Fragmentation patterns
   - Per-core allocation (multi-core fibers)

Link Utilization Over Time
---------------------------

Time-series view of link usage:

.. code-block:: python

   from fusion.visualization.link_viz import plot_link_utilization_timeline

   plot_link_utilization_timeline(
       snapshots=snapshot_data,
       link_id=(0, 1),
       save_path='plots/link_timeline.png'
   )

ML/RL Visualization
===================

Training Curves
---------------

Plot ML model training progress:

.. code-block:: python

   from fusion.modules.ml.visualization import plot_training_curves

   plot_training_curves(
       train_loss=train_losses,
       val_loss=val_losses,
       train_acc=train_accuracies,
       val_acc=val_accuracies,
       save_path='plots/ml_training.png'
   )

Shows:
   - Loss curves (training vs validation)
   - Accuracy/F1-score progression
   - Overfitting detection

RL Training Metrics
-------------------

Visualize reinforcement learning training:

.. code-block:: python

   from fusion.modules.rl.visualization.rl_plots import plot_training_progress

   plot_training_progress(
       log_dir='logs/rl_tensorboard/',
       metrics=['reward', 'success_rate', 'policy_loss'],
       save_path='plots/rl_training.png'
   )

Includes:
   - Episode rewards (mean ± std)
   - Success rate over time
   - Policy/value function losses
   - Entropy (exploration metric)

Feature Importance
------------------

Understand ML model decisions:

.. code-block:: python

   from fusion.modules.ml.visualization import plot_feature_importance

   plot_feature_importance(
       model=trained_model,
       feature_names=['bandwidth', 'path_length', 'congestion', 'fragmentation'],
       save_path='plots/feature_importance.png'
   )

Displays:
   - Bar chart of feature importances
   - Relative contribution of each feature
   - Helps interpret model behavior

Learning Curves
---------------

Diagnose model performance with learning curves:

.. code-block:: python

   from fusion.modules.ml.visualization import plot_learning_curve

   plot_learning_curve(
       model=model,
       X=X_train,
       y=y_train,
       cv=5,
       save_path='plots/learning_curve.png'
   )

Shows training/validation score vs training set size - useful for detecting overfitting/underfitting.

Comparative Analysis
====================

Multi-Scenario Comparison
--------------------------

Compare multiple algorithms or configurations:

.. code-block:: python

   from fusion.visualization.comparison import compare_scenarios

   compare_scenarios(
       scenario_data={
           'K-SP + FF': ksp_ff_results,
           'K-SP + BF': ksp_bf_results,
           'ML Routing': ml_results,
           'RL Agent': rl_results
       },
       metric='blocking_probability',
       save_path='plots/algorithm_comparison.png'
   )

Creates side-by-side or overlay plots for easy comparison.

Performance Gain Charts
-----------------------

Visualize improvement over baseline:

.. code-block:: python

   from fusion.visualization.comparison import plot_performance_gains

   plot_performance_gains(
       baseline=ksp_ff_results,
       methods={
           'ML': ml_results,
           'RL': rl_results
       },
       metric='blocking_probability',
       save_path='plots/performance_gains.png'
   )

Shows percentage improvement (positive = better).

Customization
=============

Plot Styling
------------

Customize appearance with `PlotProps`:

.. code-block:: python

   from fusion.visualization.properties import PlotProps

   props = PlotProps()
   props.color_list = ['#1f77b4', '#ff7f0e', '#2ca02c']
   props.style_list = ['-', '--', '-.']
   props.marker_list = ['o', 's', '^']
   props.title_size = 16
   props.label_size = 14

   plotter = PlotStats(sims_info_dict, plot_props=props)

Export Formats
--------------

FUSION supports multiple output formats:

.. code-block:: python

   # PNG (raster, default)
   plotter.plot_blocking_probability(save_path='plot.png', dpi=300)

   # PDF (vector, publication-ready)
   plotter.plot_blocking_probability(save_path='plot.pdf')

   # SVG (vector, web-friendly)
   plotter.plot_blocking_probability(save_path='plot.svg')

   # Interactive HTML (with plotly)
   plotter.plot_blocking_probability(save_path='plot.html', interactive=True)

Figure Size and DPI
-------------------

Control output resolution:

.. code-block:: python

   import matplotlib.pyplot as plt

   plt.figure(figsize=(10, 6), dpi=150)  # Width x Height in inches
   plotter.plot_blocking_probability()
   plt.savefig('high_res_plot.png', dpi=300, bbox_inches='tight')

Programmatic Visualization
===========================

Custom Plots with Matplotlib
-----------------------------

Create custom visualizations:

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np

   # Load your data
   erlangs = np.array([300, 400, 500, 600, 700])
   blocking_probs = np.array([0.001, 0.005, 0.02, 0.08, 0.15])

   # Create plot
   plt.figure(figsize=(8, 5))
   plt.semilogy(erlangs, blocking_probs, marker='o', linewidth=2)
   plt.xlabel('Offered Load (Erlangs)', fontsize=12)
   plt.ylabel('Blocking Probability', fontsize=12)
   plt.title('Custom Blocking Probability Plot', fontsize=14)
   plt.grid(True, alpha=0.3)
   plt.tight_layout()
   plt.savefig('custom_plot.png', dpi=300)

Batch Plotting
--------------

Generate multiple plots automatically:

.. code-block:: python

   from fusion.visualization.batch import BatchPlotter

   batch_plotter = BatchPlotter(results_dir='data/results/')

   # Generate all standard plots
   batch_plotter.generate_all_plots(
       output_dir='plots/',
       formats=['png', 'pdf']
   )

Creates:
   - Blocking probability plots
   - Bandwidth blocking plots
   - Utilization plots
   - Comparative plots

Visualization Utilities
=======================

Data Loading Helpers
--------------------

.. code-block:: python

   from fusion.visualization.utils import PlotHelpers

   helper = PlotHelpers(plot_props=props, net_names_list=['NSFNet'])
   helper.get_file_info(sims_info_dict=sims_info)

   # Access loaded data
   blocking_data = helper.blocking_prob_list
   erlang_data = helper.erlang_list

Statistical Plotting
--------------------

Add error bars and confidence intervals:

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np

   mean_bp = np.array([0.001, 0.005, 0.02])
   std_bp = np.array([0.0002, 0.001, 0.004])
   erlangs = np.array([300, 400, 500])

   plt.errorbar(erlangs, mean_bp, yerr=std_bp, capsize=5)
   plt.fill_between(erlangs, mean_bp - std_bp, mean_bp + std_bp, alpha=0.3)

Best Practices
==============

Publication-Quality Plots
--------------------------

1. **High Resolution**: Use DPI ≥ 300 for print, 150 for screen
2. **Vector Formats**: Prefer PDF/SVG for papers
3. **Font Sizes**: Labels ≥ 12pt, titles ≥ 14pt
4. **Color Blindness**: Use colorblind-friendly palettes
5. **Legends**: Clear, not blocking data

.. code-block:: python

   # Recommended settings for publications
   plt.rcParams.update({
       'font.size': 12,
       'axes.titlesize': 14,
       'axes.labelsize': 12,
       'xtick.labelsize': 11,
       'ytick.labelsize': 11,
       'legend.fontsize': 11,
       'figure.dpi': 150,
       'savefig.dpi': 300,
       'savefig.bbox': 'tight'
   })

Interactive Exploration
-----------------------

For data exploration, use interactive backends:

.. code-block:: python

   import matplotlib
   matplotlib.use('TkAgg')  # or 'Qt5Agg'

   # Or use plotly for web-based interaction
   import plotly.express as px

   fig = px.line(df, x='erlang', y='blocking_prob', color='algorithm')
   fig.show()

Common Issues
=============

Plots Not Saving
----------------

**Issue**: ``savefig()`` creates empty file

**Solutions**:
- Call ``plt.savefig()`` before ``plt.show()``
- Ensure directory exists: ``os.makedirs('plots', exist_ok=True)``
- Close previous figures: ``plt.close('all')``

Font Warnings
-------------

**Issue**: ``findfont: Font family ['sans-serif'] not found``

**Solution**:

.. code-block:: python

   import matplotlib.pyplot as plt
   plt.rcParams['font.family'] = 'DejaVu Sans'

Memory Issues with Large Plots
-------------------------------

**Issue**: Out of memory when plotting many data points

**Solutions**:
- Reduce data resolution (plot every Nth point)
- Use rasterization for dense scatter plots
- Clear figures after saving: ``plt.clf()``

Next Steps
==========

- :doc:`running_simulations` - Generate data for visualization
- :doc:`machine_learning` - Visualize ML model performance
- :doc:`reinforcement_learning` - Plot RL training metrics
- :doc:`../examples/advanced_visualization` - Advanced plotting examples

See Also
========

* :doc:`data_management` - Manage simulation output data
* :doc:`../examples/basic_simulation` - Complete simulation + plotting example
* :doc:`../developer/contributing` - Add custom visualization modules
