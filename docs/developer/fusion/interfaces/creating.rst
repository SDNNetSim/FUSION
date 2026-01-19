.. _interfaces-creating:

================================
Creating Custom Implementations
================================

This guide explains how to create your own algorithm implementations that satisfy
FUSION's pipeline protocols. Whether you're creating a new routing algorithm,
spectrum assigner, or control policy, follow these patterns.

Overview
========

Creating a custom implementation involves:

1. Understanding which protocol to implement
2. Creating a class with the required methods
3. Ensuring method signatures match exactly
4. Testing with ``isinstance()`` and mypy
5. Registering with the pipeline factory (optional)

.. important::

   You do NOT need to inherit from any base class. Python's Protocol system uses
   structural typing; any class with matching methods satisfies the protocol.

Creating a Routing Algorithm
============================

Step 1: Understand the Protocol
-------------------------------

The ``RoutingPipeline`` protocol requires one method:

.. code-block:: python

   def find_routes(
       self,
       source: str,
       destination: str,
       bandwidth_gbps: int,
       network_state: NetworkState,
       *,
       forced_path: list[str] | None = None,
   ) -> RouteResult:
       ...

Step 2: Create Your Implementation
----------------------------------

.. code-block:: python

   # my_routing.py
   from __future__ import annotations

   from typing import TYPE_CHECKING

   if TYPE_CHECKING:
       from fusion.domain.network_state import NetworkState
       from fusion.domain.results import RouteResult


   class CongestionAwareRouting:
       """
       Routing algorithm that considers link congestion.

       Finds k-shortest paths and ranks them by current congestion level,
       preferring less congested routes.
       """

       def __init__(self, congestion_weight: float = 0.5) -> None:
           """
           Initialize router.

           :param congestion_weight: How much to weight congestion vs distance (0-1)
           """
           self.congestion_weight = congestion_weight

       def find_routes(
           self,
           source: str,
           destination: str,
           bandwidth_gbps: int,
           network_state: NetworkState,
           *,
           forced_path: list[str] | None = None,
       ) -> RouteResult:
           """
           Find routes considering congestion.

           :param source: Source node identifier
           :param destination: Destination node identifier
           :param bandwidth_gbps: Required bandwidth
           :param network_state: Current network state
           :param forced_path: Optional forced path (from grooming)
           :return: RouteResult with ranked candidate paths
           """
           from fusion.domain.results import RouteResult

           if forced_path is not None:
               # Use forced path from grooming
               return self._create_single_path_result(
                   forced_path, bandwidth_gbps, network_state
               )

           # Find candidate paths
           k = network_state.config.k_paths
           candidates = self._find_k_shortest(
               network_state.topology, source, destination, k
           )

           if not candidates:
               return RouteResult.empty("CongestionAware")

           # Rank by combined distance and congestion
           ranked = self._rank_by_congestion(candidates, network_state)

           # Get valid modulations for each path
           paths, weights = zip(*ranked)
           modulations = tuple(
               self._get_modulations(w, bandwidth_gbps) for w in weights
           )

           return RouteResult(
               paths=tuple(tuple(p) for p in paths),
               weights_km=tuple(weights),
               modulations=modulations,
               strategy_name="CongestionAware",
           )

       def _find_k_shortest(self, topology, source, dest, k):
           """Find k-shortest paths using Yen's algorithm."""
           import networkx as nx
           try:
               return list(nx.shortest_simple_paths(
                   topology, source, dest, weight="weight"
               ))[:k]
           except nx.NetworkXNoPath:
               return []

       def _rank_by_congestion(self, paths, network_state):
           """Rank paths by weighted combination of distance and congestion."""
           ranked = []
           for path in paths:
               distance = self._calculate_distance(path, network_state.topology)
               congestion = self._calculate_congestion(path, network_state)
               # Lower score is better
               score = (
                   (1 - self.congestion_weight) * distance +
                   self.congestion_weight * congestion * 1000  # Scale congestion
               )
               ranked.append((path, distance, score))

           # Sort by score, return path and distance
           ranked.sort(key=lambda x: x[2])
           return [(r[0], r[1]) for r in ranked]

       def _calculate_distance(self, path, topology):
           """Calculate total path distance in km."""
           total = 0
           for i in range(len(path) - 1):
               total += topology[path[i]][path[i+1]].get("weight", 1)
           return total

       def _calculate_congestion(self, path, network_state):
           """Calculate average congestion along path (0-1)."""
           # Implementation depends on how congestion is tracked
           # This is a simplified example
           return 0.0  # Replace with actual congestion calculation

       def _get_modulations(self, distance_km, bandwidth_gbps):
           """Get valid modulations for path distance."""
           # Simplified - use actual reach tables in production
           if distance_km < 1000:
               return ("16-QAM", "QPSK", "BPSK")
           elif distance_km < 2000:
               return ("QPSK", "BPSK")
           else:
               return ("BPSK",)

       def _create_single_path_result(self, path, bandwidth_gbps, network_state):
           """Create RouteResult for a single forced path."""
           from fusion.domain.results import RouteResult

           distance = self._calculate_distance(path, network_state.topology)
           modulations = self._get_modulations(distance, bandwidth_gbps)

           return RouteResult(
               paths=(tuple(path),),
               weights_km=(distance,),
               modulations=(modulations,),
               strategy_name="CongestionAware(forced)",
           )

Step 3: Test Protocol Compliance
--------------------------------

.. code-block:: python

   # test_my_routing.py
   import pytest
   from fusion.interfaces import RoutingPipeline
   from my_routing import CongestionAwareRouting


   class TestCongestionAwareRouting:
       """Tests for CongestionAwareRouting."""

       def test_satisfies_protocol(self) -> None:
           """Implementation should satisfy RoutingPipeline protocol."""
           router = CongestionAwareRouting()
           assert isinstance(router, RoutingPipeline)

       def test_find_routes_returns_route_result(self, mock_network_state) -> None:
           """find_routes should return RouteResult."""
           from fusion.domain.results import RouteResult

           router = CongestionAwareRouting()
           result = router.find_routes(
               source="0",
               destination="5",
               bandwidth_gbps=100,
               network_state=mock_network_state,
           )
           assert isinstance(result, RouteResult)

       def test_empty_result_when_no_path(self, disconnected_state) -> None:
           """Should return empty result when no path exists."""
           router = CongestionAwareRouting()
           result = router.find_routes(
               source="0",
               destination="99",  # Non-existent
               bandwidth_gbps=100,
               network_state=disconnected_state,
           )
           assert result.is_empty

Step 4: Use Your Implementation
-------------------------------

.. code-block:: python

   from fusion.core.pipeline_factory import PipelineSet
   from my_routing import CongestionAwareRouting

   # Create your router
   router = CongestionAwareRouting(congestion_weight=0.7)

   # Use in pipeline set
   pipelines = PipelineSet(
       routing=router,
       spectrum=existing_spectrum_pipeline,
       # ...
   )

   # Or create orchestrator directly
   from fusion.core.orchestrator import SDNOrchestrator

   orchestrator = SDNOrchestrator(
       config=simulation_config,
       pipelines=pipelines,
   )

Creating a Control Policy
=========================

Control policies decide which path option to select. They can be heuristics,
RL agents, or supervised/unsupervised models.

Step 1: Understand the Protocol
-------------------------------

.. code-block:: python

   class ControlPolicy(Protocol):
       def select_action(
           self,
           request: Request,
           options: list[PathOption],
           network_state: NetworkState,
       ) -> int: ...

       def update(self, request: Request, action: int, reward: float) -> None: ...

       def get_name(self) -> str: ...

Step 2: Create a Heuristic Policy
---------------------------------

.. code-block:: python

   # my_policies.py
   from __future__ import annotations

   from typing import TYPE_CHECKING

   if TYPE_CHECKING:
       from fusion.domain.network_state import NetworkState
       from fusion.domain.request import Request
       from fusion.modules.rl.adapter import PathOption


   class LoadBalancedPolicy:
       """
       Select paths to balance load across the network.

       Prefers paths with lower current utilization to spread
       traffic more evenly.
       """

       def __init__(self, utilization_threshold: float = 0.8) -> None:
           """
           Initialize policy.

           :param utilization_threshold: Avoid paths above this utilization
           """
           self.utilization_threshold = utilization_threshold

       def select_action(
           self,
           request: Request,
           options: list[PathOption],
           network_state: NetworkState,
       ) -> int:
           """
           Select path with lowest utilization.

           :param request: The request to serve
           :param options: Available path options
           :param network_state: Current network state
           :return: Index of selected path, or -1 if none feasible
           """
           best_index = -1
           best_utilization = float("inf")

           for i, opt in enumerate(options):
               if not opt.is_feasible:
                   continue

               # Calculate path utilization
               utilization = self._calculate_path_utilization(
                   opt.path, network_state
               )

               # Skip overloaded paths
               if utilization > self.utilization_threshold:
                   continue

               if utilization < best_utilization:
                   best_utilization = utilization
                   best_index = i

           # Fallback to first feasible if all are overloaded
           if best_index == -1:
               for i, opt in enumerate(options):
                   if opt.is_feasible:
                       return i

           return best_index

       def update(self, request: Request, action: int, reward: float) -> None:
           """No learning for heuristic policy."""
           pass

       def get_name(self) -> str:
           """Return policy name."""
           return f"LoadBalancedPolicy(threshold={self.utilization_threshold})"

       def _calculate_path_utilization(self, path, network_state) -> float:
           """Calculate average utilization along path."""
           # Implementation depends on how utilization is tracked
           # This is a simplified placeholder
           return 0.5

Step 3: Create an RL Policy Wrapper
-----------------------------------

.. code-block:: python

   class RLPolicyWrapper:
       """
       Wrap a trained RL agent as a ControlPolicy.

       This allows using agents trained with Stable-Baselines3
       or custom RL frameworks as control policies.
       """

       def __init__(self, agent, observation_builder, fallback_policy=None):
           """
           Initialize wrapper.

           :param agent: Trained RL agent with predict() method
           :param observation_builder: Converts state to observation
           :param fallback_policy: Policy to use if RL action is infeasible
           """
           self.agent = agent
           self.obs_builder = observation_builder
           self.fallback = fallback_policy or FirstFeasiblePolicy()
           self._last_obs = None

       def select_action(
           self,
           request: Request,
           options: list[PathOption],
           network_state: NetworkState,
       ) -> int:
           """Select action using RL agent."""
           # Build observation
           self._last_obs = self.obs_builder.build(
               request, options, network_state
           )

           # Get agent's action
           action, _ = self.agent.predict(self._last_obs, deterministic=True)
           action = int(action)

           # Validate action is feasible
           if 0 <= action < len(options) and options[action].is_feasible:
               return action

           # Fallback if RL chose infeasible action
           return self.fallback.select_action(request, options, network_state)

       def update(self, request: Request, action: int, reward: float) -> None:
           """Store experience for potential offline analysis."""
           # For online learning, you might update the agent here
           # For deployed policies, this might just log
           pass

       def get_name(self) -> str:
           """Return policy name."""
           return f"RLPolicy({type(self.agent).__name__})"


   class FirstFeasiblePolicy:
       """Simple fallback policy."""

       def select_action(self, request, options, network_state) -> int:
           for i, opt in enumerate(options):
               if opt.is_feasible:
                   return i
           return -1

       def update(self, request, action, reward) -> None:
           pass

       def get_name(self) -> str:
           return "FirstFeasiblePolicy"

Creating a Spectrum Assigner
============================

Step 1: Understand the Protocol
-------------------------------

.. code-block:: python

   class SpectrumPipeline(Protocol):
       def find_spectrum(
           self,
           path: list[str],
           modulation: str | list[str],
           bandwidth_gbps: int,
           network_state: NetworkState,
           *,
           connection_index: int | None = None,
           path_index: int = 0,
           use_dynamic_slicing: bool = False,
           snr_bandwidth: int | None = None,
           request_id: int | None = None,
           slice_bandwidth: int | None = None,
       ) -> SpectrumResult: ...

       def find_protected_spectrum(
           self,
           primary_path: list[str],
           backup_path: list[str],
           modulation: str,
           bandwidth_gbps: int,
           network_state: NetworkState,
       ) -> SpectrumResult: ...

Step 2: Create Your Implementation
----------------------------------

.. code-block:: python

   # my_spectrum.py
   from __future__ import annotations

   from typing import TYPE_CHECKING

   if TYPE_CHECKING:
       from fusion.domain.network_state import NetworkState
       from fusion.domain.results import SpectrumResult


   class BestFitSpectrum:
       """
       Best-fit spectrum assignment algorithm.

       Finds the smallest available spectrum gap that fits the request,
       reducing fragmentation compared to first-fit.
       """

       def find_spectrum(
           self,
           path: list[str],
           modulation: str | list[str],
           bandwidth_gbps: int,
           network_state: NetworkState,
           *,
           connection_index: int | None = None,
           path_index: int = 0,
           use_dynamic_slicing: bool = False,
           snr_bandwidth: int | None = None,
           request_id: int | None = None,
           slice_bandwidth: int | None = None,
       ) -> SpectrumResult:
           """Find best-fit spectrum along path."""
           from fusion.domain.results import SpectrumResult

           # Calculate slots needed
           effective_bw = slice_bandwidth or bandwidth_gbps
           mod = modulation[0] if isinstance(modulation, list) else modulation
           slots_needed = self._calculate_slots(effective_bw, mod)

           # Search for best fit across bands and cores
           best_fit = None
           best_gap_size = float("inf")

           for band in network_state.config.band_list:
               for core in range(network_state.config.cores_per_link):
                   gaps = self._find_gaps(path, core, band, network_state)
                   for start, gap_size in gaps:
                       if gap_size >= slots_needed and gap_size < best_gap_size:
                           best_fit = (start, core, band)
                           best_gap_size = gap_size

           if best_fit is None:
               return SpectrumResult.not_found(slots_needed)

           start, core, band = best_fit
           return SpectrumResult(
               is_free=True,
               start_slot=start,
               end_slot=start + slots_needed - 1,
               core=core,
               band=band,
               modulation=mod,
               slots_needed=slots_needed,
           )

       def find_protected_spectrum(
           self,
           primary_path: list[str],
           backup_path: list[str],
           modulation: str,
           bandwidth_gbps: int,
           network_state: NetworkState,
       ) -> SpectrumResult:
           """Find spectrum for both primary and backup paths."""
           from fusion.domain.results import SpectrumResult

           # Find spectrum for primary
           primary_result = self.find_spectrum(
               primary_path, modulation, bandwidth_gbps, network_state
           )
           if not primary_result.is_free:
               return primary_result

           # Find spectrum for backup (may use different core/band)
           backup_result = self.find_spectrum(
               backup_path, modulation, bandwidth_gbps, network_state
           )
           if not backup_result.is_free:
               return SpectrumResult.not_found(primary_result.slots_needed)

           # Return combined result
           return SpectrumResult(
               is_free=True,
               start_slot=primary_result.start_slot,
               end_slot=primary_result.end_slot,
               core=primary_result.core,
               band=primary_result.band,
               modulation=modulation,
               slots_needed=primary_result.slots_needed,
               backup_start_slot=backup_result.start_slot,
               backup_end_slot=backup_result.end_slot,
               backup_core=backup_result.core,
               backup_band=backup_result.band,
           )

       def _calculate_slots(self, bandwidth_gbps, modulation):
           """Calculate required slots for bandwidth and modulation."""
           # Simplified calculation - use actual formula in production
           bits_per_symbol = {"BPSK": 1, "QPSK": 2, "16-QAM": 4}.get(modulation, 2)
           slots = bandwidth_gbps // (12.5 * bits_per_symbol)  # 12.5 GHz per slot
           return max(1, int(slots)) + 1  # +1 for guard band

       def _find_gaps(self, path, core, band, network_state):
           """Find available gaps along path."""
           # Implementation depends on NetworkState API
           # Returns list of (start_slot, gap_size) tuples
           return []  # Replace with actual implementation

Best Practices
==============

1. **Don't Store NetworkState**

   Always receive ``network_state`` as a parameter. Never do:

   .. code-block:: python

      # BAD - storing state
      def __init__(self, network_state):
          self.state = network_state  # Don't do this!

      # GOOD - receive per call
      def find_routes(self, source, dest, bw, network_state):
          # Use network_state here
          pass

2. **Return Proper Result Objects**

   Always return the expected result type from ``fusion.domain.results``:

   .. code-block:: python

      from fusion.domain.results import RouteResult

      # Use factory methods when appropriate
      return RouteResult.empty("MyAlgorithm")  # For no routes found

3. **Include All Optional Parameters**

   Even if you don't use them, include optional parameters for compatibility:

   .. code-block:: python

      def find_spectrum(
          self,
          path,
          modulation,
          bandwidth_gbps,
          network_state,
          *,
          connection_index=None,  # Include even if unused
          path_index=0,
          use_dynamic_slicing=False,
          snr_bandwidth=None,
          request_id=None,
          slice_bandwidth=None,
      ):
          ...

4. **Test with isinstance()**

   Verify your implementation satisfies the protocol:

   .. code-block:: python

      from fusion.interfaces import RoutingPipeline

      router = MyRouter()
      assert isinstance(router, RoutingPipeline)

5. **Run mypy**

   Type check your implementation:

   .. code-block:: bash

      mypy my_routing.py --strict

6. **Document Your Algorithm**

   Include docstrings explaining the algorithm and its parameters:

   .. code-block:: python

      class MyRouter:
          """
          Brief description of the algorithm.

          Longer explanation of how it works, when to use it,
          and any important considerations.

          :param param1: Description of parameter
          :param param2: Description of parameter

          Example::

              router = MyRouter(param1=value)
              result = router.find_routes(...)
          """
