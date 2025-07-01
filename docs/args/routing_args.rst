
Routing Arguments
=================

The `RoutingProps` class stores all parameters used during path computation and routing logic.
These fields are assigned dynamically and support routing algorithms such as K-shortest path,
XT-aware routing, and precomputed path selection.

.. automodule:: arg_scripts.routing_args
    :members:
    :undoc-members:

RoutingProps Attributes
------------------------

.. list-table::
   :header-rows: 1

   * - Attribute
     - Type
     - Description

   * - paths_matrix
     - list
     - Matrix of potential paths for a single request

   * - mod_formats_matrix
     - list
     - Modulation formats corresponding to each path in ``paths_matrix``

   * - weights_list
     - list
     - List of path weights (e.g., length, XT, or other criteria)

   * - path_index_list
     - list
     - Index tracking for each path when using precomputed routing

   * - input_power
     - float
     - Input power in Watts

   * - freq_spacing
     - float
     - Frequency spacing in Hz

   * - mci_worst
     - float
     - Worst-case mutual coupling interference value

   * - max_link_length
     - float or None
     - Maximum link length in km

   * - span_len
     - float
     - Length of a single span in km

   * - max_span
     - int or None
     - Maximum number of spans considered in the network

   * - connection_index
     - int or None
     - Source-destination index for precalculated routing

   * - path_index
     - int or None
     - Index of selected path during spectrum assignment
