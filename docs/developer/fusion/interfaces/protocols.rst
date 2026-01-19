.. _interfaces-protocols:

==================
Pipeline Protocols
==================

This document provides detailed documentation for each pipeline protocol defined
in ``fusion/interfaces/pipelines.py`` and ``fusion/interfaces/control_policy.py``.

RoutingPipeline
===============

:Location: ``fusion/interfaces/pipelines.py``
:Purpose: Find candidate routes between source and destination nodes

The ``RoutingPipeline`` protocol defines how routing algorithms expose their
functionality to the orchestrator.

Protocol Definition
-------------------

.. code-block:: python

   @runtime_checkable
   class RoutingPipeline(Protocol):
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

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Type
     - Description
   * - ``source``
     - ``str``
     - Source node identifier
   * - ``destination``
     - ``str``
     - Destination node identifier
   * - ``bandwidth_gbps``
     - ``int``
     - Required bandwidth (used for modulation selection)
   * - ``network_state``
     - ``NetworkState``
     - Current network state (topology, configuration)
   * - ``forced_path``
     - ``list[str] | None``
     - If provided, use this path instead of searching (for partial grooming)

Return Value
------------

Returns a ``RouteResult`` (from ``fusion.domain.results``) containing:

- ``paths``: Candidate paths as tuples of node IDs
- ``weights_km``: Path distances in kilometers
- ``modulations``: Valid modulation formats per path
- ``strategy_name``: Name of routing algorithm used

Returns an empty ``RouteResult`` (``is_empty=True``) if no routes found.

Implementation Notes
--------------------

- Number of paths is limited by ``network_state.config.k_paths``
- Modulation formats are filtered by reach based on path weight
- Must NOT modify ``network_state`` (pure query method)
- Must NOT store ``network_state`` as instance attribute

Example Implementation
----------------------

.. code-block:: python

   from fusion.domain.network_state import NetworkState
   from fusion.domain.results import RouteResult

   class KShortestPathRouting:
       """K-shortest paths routing implementation."""

       def find_routes(
           self,
           source: str,
           destination: str,
           bandwidth_gbps: int,
           network_state: NetworkState,
           *,
           forced_path: list[str] | None = None,
       ) -> RouteResult:
           if forced_path is not None:
               # Use forced path from grooming
               paths = (tuple(forced_path),)
               weights = (self._calculate_weight(forced_path, network_state),)
           else:
               # Find k-shortest paths
               k = network_state.config.k_paths
               paths, weights = self._find_k_shortest(
                   network_state.topology, source, destination, k
               )

           # Filter modulations by reach
           modulations = self._get_valid_modulations(weights, bandwidth_gbps)

           return RouteResult(
               paths=paths,
               weights_km=weights,
               modulations=modulations,
               strategy_name="KShortestPath",
           )

SpectrumPipeline
================

:Location: ``fusion/interfaces/pipelines.py``
:Purpose: Find available spectrum slots along a path

The ``SpectrumPipeline`` protocol defines how spectrum assignment algorithms
find free spectrum for lightpath establishment.

Protocol Definition
-------------------

.. code-block:: python

   @runtime_checkable
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
       ) -> SpectrumResult:
           ...

       def find_protected_spectrum(
           self,
           primary_path: list[str],
           backup_path: list[str],
           modulation: str,
           bandwidth_gbps: int,
           network_state: NetworkState,
       ) -> SpectrumResult:
           ...

Key Parameters
--------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Type
     - Description
   * - ``path``
     - ``list[str]``
     - Ordered list of node IDs forming the route
   * - ``modulation``
     - ``str | list[str]``
     - Modulation format(s) to try
   * - ``bandwidth_gbps``
     - ``int``
     - Required bandwidth in Gbps
   * - ``network_state``
     - ``NetworkState``
     - Current network state
   * - ``use_dynamic_slicing``
     - ``bool``
     - If True, may return partial allocation

Return Value
------------

Returns a ``SpectrumResult`` containing:

- ``is_free``: True if spectrum was found
- ``start_slot``, ``end_slot``: Slot range (if found)
- ``core``: Core index (if found)
- ``band``: Band identifier (if found)
- ``modulation``: Confirmed modulation format
- ``slots_needed``: Number of slots required

Implementation Notes
--------------------

- Does NOT allocate spectrum (caller uses ``NetworkState.create_lightpath()``)
- Searches across all bands and cores based on allocation policy
- ``slots_needed`` includes guard band slots

GroomingPipeline
================

:Location: ``fusion/interfaces/pipelines.py``
:Purpose: Pack requests onto existing lightpaths with available capacity

Protocol Definition
-------------------

.. code-block:: python

   @runtime_checkable
   class GroomingPipeline(Protocol):
       def try_groom(
           self,
           request: Request,
           network_state: NetworkState,
       ) -> GroomingResult:
           ...

       def rollback_groom(
           self,
           request: Request,
           lightpath_ids: list[int],
           network_state: NetworkState,
       ) -> None:
           ...

Return Value
------------

Returns a ``GroomingResult`` containing:

- ``fully_groomed``: True if entire request was groomed
- ``partially_groomed``: True if some bandwidth was groomed
- ``bandwidth_groomed_gbps``: Amount successfully groomed
- ``remaining_bandwidth_gbps``: Amount still needing allocation
- ``lightpaths_used``: IDs of lightpaths used
- ``forced_path``: Suggested path for remainder (if partial)

Implementation Notes
--------------------

- MAY have side effects (modifies lightpath bandwidth allocations)
- Must support rollback via ``rollback_groom()``
- Does NOT modify spectrum (lightpaths already exist)

SNRPipeline
===========

:Location: ``fusion/interfaces/pipelines.py``
:Purpose: Validate signal-to-noise ratio meets modulation requirements

Protocol Definition
-------------------

.. code-block:: python

   @runtime_checkable
   class SNRPipeline(Protocol):
       def validate(
           self,
           lightpath: Lightpath,
           network_state: NetworkState,
       ) -> SNRResult:
           ...

       def recheck_affected(
           self,
           new_lightpath_id: int,
           network_state: NetworkState,
           *,
           affected_range_slots: int = 5,
           slicing_flag: bool = False,
       ) -> SNRRecheckResult:
           ...

Return Values
-------------

``validate()`` returns ``SNRResult``:

- ``passed``: True if SNR meets threshold
- ``snr_db``: Measured/calculated SNR value
- ``required_snr_db``: Threshold for modulation
- ``margin_db``: SNR margin above threshold
- ``failure_reason``: Why it failed (if applicable)

``recheck_affected()`` returns ``SNRRecheckResult``:

- ``all_pass``: True if all affected lightpaths still pass
- ``degraded_lightpath_ids``: Lightpaths now failing
- ``violations``: Dict mapping lightpath_id to SNR shortfall

Implementation Notes
--------------------

- Both methods are pure query methods (no side effects)
- ``recheck_affected()`` identifies existing lightpaths degraded by new allocation
- Used to trigger rollback if existing lightpaths become invalid

SlicingPipeline
===============

:Location: ``fusion/interfaces/pipelines.py``
:Purpose: Divide large requests into multiple smaller allocations

Protocol Definition
-------------------

.. code-block:: python

   @runtime_checkable
   class SlicingPipeline(Protocol):
       def try_slice(
           self,
           request: Request,
           path: list[str],
           modulation: str,
           bandwidth_gbps: int,
           network_state: NetworkState,
           *,
           max_slices: int | None = None,
           spectrum_pipeline: SpectrumPipeline | None = None,
           snr_pipeline: SNRPipeline | None = None,
           connection_index: int | None = None,
           path_index: int = 0,
           snr_accumulator: list[float] | None = None,
           path_weight: float | None = None,
       ) -> SlicingResult:
           ...

       def rollback_slices(
           self,
           lightpath_ids: list[int],
           network_state: NetworkState,
       ) -> None:
           ...

Return Value
------------

Returns a ``SlicingResult`` containing:

- ``success``: True if slicing is possible
- ``num_slices``: Number of slices needed
- ``slice_bandwidth_gbps``: Bandwidth per slice
- ``lightpaths_created``: IDs if allocated, empty for feasibility check
- ``total_bandwidth_gbps``: Total bandwidth allocated

Implementation Notes
--------------------

- Only called when single lightpath allocation fails
- By default, just checks feasibility (doesn't allocate)
- When ``spectrum_pipeline`` and ``snr_pipeline`` provided, performs full allocation
- Limited by ``max_slices`` or ``config.max_slices``

ControlPolicy
=============

:Location: ``fusion/interfaces/control_policy.py``
:Purpose: Select actions for resource allocation from available options

The ``ControlPolicy`` protocol defines how decision-making policies (heuristics,
RL agents, or supervised/unsupervised models) interface with the orchestrator.

Protocol Definition
-------------------

.. code-block:: python

   @runtime_checkable
   class ControlPolicy(Protocol):
       def select_action(
           self,
           request: Request,
           options: list[PathOption],
           network_state: NetworkState,
       ) -> int:
           ...

       def update(self, request: Request, action: int, reward: float) -> None:
           ...

       def get_name(self) -> str:
           ...

Methods
-------

select_action()
^^^^^^^^^^^^^^^

Select which path option to use for the given request.

:param request: The incoming request to serve
:param options: List of ``PathOption`` objects with feasibility info
:param network_state: Current network state (read-only)
:return: Path index (0 to len-1), or -1 if no valid action

.. note::

   Policies MUST only return indices where ``options[i].is_feasible`` is True.
   Returning an infeasible index is undefined behavior.

update()
^^^^^^^^

Update policy based on experience (for online learning).

:param request: The request that was served
:param action: The action (path index) that was taken
:param reward: The reward received

.. note::

   - Heuristic policies: implement as ``pass``
   - RL policies: update replay buffers, internal state
   - Supervised/unsupervised policies: implement as ``pass`` (inference only)

get_name()
^^^^^^^^^^

Return the policy name for logging and metrics.

:return: Human-readable policy name

Examples: ``"FirstFeasiblePolicy"``, ``"RLPolicy(PPO)"``, ``"SupervisedPolicy(pytorch)"``

Example Implementations
-----------------------

**Heuristic Policy (First Feasible)**

.. code-block:: python

   class FirstFeasiblePolicy:
       """Select the first feasible path option."""

       def select_action(self, request, options, network_state) -> int:
           for i, opt in enumerate(options):
               if opt.is_feasible:
                   return i
           return -1  # No feasible option

       def update(self, request, action, reward) -> None:
           pass  # Heuristic - no learning

       def get_name(self) -> str:
           return "FirstFeasiblePolicy"

**RL Policy Wrapper**

.. code-block:: python

   class RLControlPolicy:
       """Wrap an RL agent as a ControlPolicy."""

       def __init__(self, agent, observation_builder):
           self.agent = agent
           self.obs_builder = observation_builder

       def select_action(self, request, options, network_state) -> int:
           obs = self.obs_builder.build(request, options, network_state)
           action = self.agent.predict(obs)
           # Ensure action is feasible
           if 0 <= action < len(options) and options[action].is_feasible:
               return action
           # Fallback to first feasible
           for i, opt in enumerate(options):
               if opt.is_feasible:
                   return i
           return -1

       def update(self, request, action, reward) -> None:
           self.agent.store_transition(action, reward)

       def get_name(self) -> str:
           return f"RLPolicy({self.agent.algorithm_name})"

Protocol Compliance Checklist
=============================

When implementing a pipeline protocol, verify:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Check
     - How to Verify
   * - Method names match exactly
     - Compare with protocol definition
   * - Parameter names match
     - Include all required and optional parameters
   * - Parameter types match
     - Use same types or compatible subtypes
   * - Return type matches
     - Return the exact result type from ``fusion.domain.results``
   * - No stored NetworkState
     - Receive state as parameter, never as ``self.state``
   * - Passes isinstance() check
     - ``assert isinstance(impl, ProtocolClass)``
   * - Passes mypy
     - Run ``mypy`` on your implementation

Testing Protocol Compliance
---------------------------

.. code-block:: python

   from fusion.interfaces import RoutingPipeline

   def test_my_router_satisfies_protocol():
       """Verify implementation satisfies RoutingPipeline protocol."""
       router = MyCustomRouter()

       # Runtime check
       assert isinstance(router, RoutingPipeline)

       # Call with minimal arguments
       result = router.find_routes(
           source="0",
           destination="5",
           bandwidth_gbps=100,
           network_state=mock_state,
       )

       # Verify return type
       from fusion.domain.results import RouteResult
       assert isinstance(result, RouteResult)
