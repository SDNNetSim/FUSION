Observation Arguments
======================

This module defines supported observation formats used in Gym environments. Each version
includes progressively richer features for reinforcement learning agents.

.. list-table:: Observation Space Dictionary
   :header-rows: 1

   * - Observation Key
     - Features Included

   * - obs_1
     - ``["source", "destination"]``

   * - obs_2
     - ``["source", "destination", "request_bandwidth"]``

   * - obs_3
     - ``["source", "destination", "holding_time"]``

   * - obs_4
     - ``["source", "destination", "request_bandwidth", "holding_time"]``

   * - obs_5
     - ``["source", "destination", "request_bandwidth", "holding_time", "slots_needed", "path_lengths"]``

   * - obs_6
     - ``["source", "destination", "request_bandwidth", "holding_time", "slots_needed", "path_lengths", "paths_cong"]``

   * - obs_7
     - ``["source", "destination", "request_bandwidth", "holding_time", "slots_needed", "path_lengths", "paths_cong", "available_slots"]``

   * - obs_8
     - ``["source", "destination", "request_bandwidth", "holding_time", "slots_needed", "path_lengths", "paths_cong", "available_slots", "is_feasible"]``
