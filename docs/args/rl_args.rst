Reinforcement Learning Arguments
=================================

The reinforcement learning (RL) component of the simulator offers a set of configuration arguments for fine-tuning
the agent's behavior and training process. These arguments govern aspects like the action space, observation
space, and tracking important statistics.


.. list-table:: empty_ai_props
   :widths: 25 25
   :header-rows: 1

   * - Argument Name
     - Description
   * - k_paths
     - Number of paths the RL agent has to choose from
   * - cores_per_link
     - Number of cores the RL agent has to choose from
   * - spectral_slots
     - Number of slots the DRL agent has to choose from
   * - num_nodes
     - Number of nodes of the topology
   * - bandwidth_list
     - List of bandwidths used in the simulation
   * - arrival_list
     - Contains every arrival request
   * - depart_list
     - Contains every departure request
   * - mock_sdn_dict
     - Used in place of the sdn_dict for certain functionality
   * - source
     - Source node
   * - destination
     - Destination node
   * - paths_list
     - List of potential paths
   * - path_index
     - Index of the path chosen
   * - chosen_path
     - The chosen path
   * - core_index
     - Index of the core chosen

.. list-table:: empty_q_props
   :widths: 25 25
   :header-rows: 1

   * - Argument Name
     - Description
   * - epsilon
     - Current epsilon the agent is using
   * - epsilon_start
     - Epsilon at the start of a single iteration/episode
   * - epsilon_end
     - Epsilon at the end of a single iteration/episode
   * - epsilon_list
     - List to keep track of epsilon decay values
   * - is_training
     - Whether the agent is training or testing
   * - rewards_dict
     - Contains rewards for route and core q-tables
   * - errors_dict
     - Contains temporal difference errors for route and core q-tables
   * - sum_rewards_dict
     - The total sum of rewards for each iteration/episode
   * - sum_errors_dict
     - Total sum of temporal difference errors for each iteration/episode
   * - routes_matrix (path q-table)
     - Every possible combination of 'k' routes for the q-learning agent
   * - cores_matrix (core q-table)
     - Every possible 'c' cores to be selected on each link
   * - num_nodes
     - Number of nodes in the topology
   * - save_params_dict
     - Contains lists of important parameters for the q-learning agent to be saved


.. list-table:: empty_drl_props
   :widths: 25 25
   :header-rows: 1

   * - Argument Name
     - Description
   * - min_arrival
     - Minimum arrival time in the request distribution
   * - max_arrival
     - Maximum arrival time in the request distribution
   * - min_depart
     - Minimum departure time in the request distribution
   * - max_depart
     - Maximum departure time in the request distribution
   * - max_slots_needed
     - Maximum possible slots needed a request can have
   * - max_length
     - Maximum possible path length
   * - slice_space
     - Agent to decide whether to slice a light segment or not (0 or 1)
