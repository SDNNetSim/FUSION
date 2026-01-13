========================
Advanced Visualization
========================

Create publication-quality plots and custom visualizations.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
========

FUSION provides powerful visualization capabilities through matplotlib and
a plugin-based plotting system.

Quick Start
===========

.. code-block:: python

   from fusion.visualization.plot_stats import plot_results

   # Auto-generate all standard plots
   plot_results('output/simulation_results.json', output_dir='plots/')

Custom Plots
============

Blocking Probability Curve
---------------------------

.. code-block:: python

   import matplotlib.pyplot as plt
   from fusion.visualization.plot_stats import load_results

   results = load_results('output/simulation_results.json')

   plt.figure(figsize=(10, 6))
   plt.plot(results['erlangs'], results['blocking'], 'o-', linewidth=2)
   plt.xlabel('Traffic Load (Erlangs)', fontsize=12)
   plt.ylabel('Blocking Probability', fontsize=12)
   plt.grid(True, alpha=0.3)
   plt.tight_layout()
   plt.savefig('blocking_curve.png', dpi=300)

Multi-Algorithm Comparison
---------------------------

.. code-block:: python

   import matplotlib.pyplot as plt

   algorithms = ['first_fit', 'best_fit', 'last_fit']
   fig, ax = plt.subplots(figsize=(12, 6))

   for alg in algorithms:
       results = load_results(f'output/{alg}_results.json')
       ax.plot(results['erlangs'], results['blocking'],
               'o-', label=alg.replace('_', ' ').title(), linewidth=2)

   ax.set_xlabel('Traffic Load (Erlangs)')
   ax.set_ylabel('Blocking Probability')
   ax.legend()
   ax.grid(True, alpha=0.3)
   plt.savefig('algorithm_comparison.png', dpi=300)

See Also
========

* :doc:`../user_guide/visualization` - Visualization guide
* :doc:`../api/visualization` - Visualization API
