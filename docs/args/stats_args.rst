
Statistics Arguments
====================

This documentation page focuses on the various statistics arguments employed in EON simulations.
These arguments track crucial network parameters, such as blocking probability, providing valuable insights
into network behavior. By understanding these arguments, you can gain a deeper understanding of your EON's
performance limitations and optimize its configuration for optimal results.

.. automodule:: arg_scripts.stats_args
    :members:
    :undoc-members:

StatsProps Attributes
---------------------

.. list-table::
   :header-rows: 1

   * - Attribute
     - Description

   * - snapshots_dict
     - Statistics snapshot dictionary (collected at various points)

   * - cores_dict
     - Tracks cores used throughout simulations

   * - weights_dict
     - Path weight metrics across requests

   * - mods_used_dict
     - Records modulation formats used

   * - block_bw_dict
     - Blocking statistics by bandwidth

   * - block_reasons_dict
     - Dictionary logging blocking reasons: distance, congestion, or XT violations

   * - link_usage_dict
     - Per-link usage and throughput tracking

   * - sim_block_list
     - Overall blocking probabilities per simulation

   * - sim_br_block_list
     - Blocking probabilities broken down by reason

   * - trans_list
     - Number of transponders used

   * - hops_list
     - Average number of hops used

   * - lengths_list
     - Average path lengths

   * - route_times_list
     - Average routing time per request

   * - xt_list
     - Average XT values

   * - bands_list
     - Bands allocated in each request

   * - start_slot_list
     - Slot index allocations (start)

   * - end_slot_list
     - Slot index allocations (end)

   * - modulation_list
     - List of modulations selected

   * - bandwidth_list
     - Bandwidth values chosen

   * - path_index_list
     - Index of path used for each request
