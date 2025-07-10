SDN Arguments
==============

A software-defined network (SDN) controller serves as the intelligent command center of the network. It centralizes
decision-making processes and uses software to dictate how network traffic should flow. This documentation page
provides insights into the essential arguments and parameters used to configure and manage the SDN controller
effectively.

.. automodule:: arg_scripts.sdn_args
    :members:
    :undoc-members:

SDNProps Attributes
-------------------

.. list-table:: empty_props
   :widths: 25 25
   :header-rows: 1

   * - Argument Name
     - Description

   * - path_list
     - A list of nodes for a single path

   * - was_routed
     - If the request was successfully routed or not

   * - topology
     - A graph of the current network topology

   * - net_spec_dict
     - The latest network spectrum database

   * - req_id
     - Current request ID number

   * - source
     - Current source node

   * - destination
     - Current destination node

   * - bandwidth
     - Current request bandwidth (Gbps)

   * - bandwidth_list
     - Bandwidths used for a request, multiple if request's segment was sliced

   * - modulation_list
     - Modulation formats used for a request

   * - core_list
     - Cores used for a request

   * - xt_list
     - Cross-talk cost for a request

   * - stat_key_list
     - Important statistics to be updated after each block/allocation

   * - num_trans
     - Number of transponders used for a request

   * - slots_needed
     - How many spectral slots a request needs

   * - single_core
     - Forced a request to allocate to the first core

   * - block_reason
     - Why a request was blocked; distance, congestion, none, xt, etc.

   * - mod_formats_dict
     - Mod formats valid for this bandwidth

   * - stat_key_list
     - Keys used to log slicing statistics