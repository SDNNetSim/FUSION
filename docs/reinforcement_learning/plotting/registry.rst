Registry
=================

This module defines the `PLOTS` dictionary that maps standardized plot names to a tuple containing:
1. The function used to render the plot.
2. The function used to process the raw simulation data before plotting.
3. (Optional) Metadata or labels for the plot type.

.. list-table:: Plot Function Registry
   :header-rows: 1

   * - Plot Key
     - Plot Function
     - Processor Function

   * - memory_usage
     - ``plot_memory_usage``
     - ``process_memory_usage``

   * - time_usage
     - ``plot_time_usage``
     - ``process_time_usage``

   * - mod_usage
     - ``plot_modulation_usage``
     - ``process_modulation_usage``

   * - path_index
     - ``plot_path_index``
     - ``process_path_index``

   * - blocking
     - ``plot_blocking_probability``
     - ``process_blocking_probability``

   * - blocking_bw
     - ``plot_blocked_bandwidth``
     - ``process_blocking_bandwidth``

   * - rewards
     - ``plot_rewards_mean_var``
     - ``None``

   * - resource_stats
     - ``plot_resource_percent_delta_heatmaps``
     - ``None``
