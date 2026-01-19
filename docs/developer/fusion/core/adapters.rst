.. _core-adapters:

========
Adapters
========

This document explains the adapter pattern used in FUSION to bridge the legacy
code with the new orchestrator-based pipeline architecture.

.. important::

   Adapters are **TEMPORARY MIGRATION LAYERS**. They will be replaced with clean
   implementations in v6.1.0. When contributing new features, prefer implementing
   directly against pipeline protocols rather than extending adapters.

Overview
========

The adapter pattern allows the orchestrator to use legacy components (Routing,
SpectrumAssignment, SnrMeasurements, Grooming) without rewriting them. Each
adapter:

1. Receives Phase 1 objects (``NetworkState``, ``SimulationConfig``)
2. Creates proxy objects for legacy code
3. Calls the legacy component methods
4. Converts legacy results to Result objects

.. code-block:: text

   +-------------------+     +-------------------+     +-------------------+
   |   Orchestrator    | --> |     Adapter       | --> |  Legacy Component |
   |                   |     |                   |     |                   |
   | Uses:             |     | Converts:         |     | Uses:             |
   | - NetworkState    |     | - NetworkState    |     | - engine_props    |
   | - SimulationConfig|     |   to SDNPropsProxy|     | - sdn_props       |
   | - Request         |     | - Result to       |     | - Properties      |
   |                   |     |   RouteResult     |     |   classes         |
   +-------------------+     +-------------------+     +-------------------+

Why Adapters?
=============

The adapter pattern was chosen for several reasons:

1. **Gradual Migration**
   - Legacy code continues to work during transition
   - No "big bang" rewrite required
   - Risk is spread across multiple releases

2. **Preserving Algorithms**
   - Complex algorithms (SNR calculations, spectrum assignment) are preserved
   - Only the interface changes, not the core logic
   - Reduces chance of introducing bugs

3. **Testing Strategy**
   - Adapters can be tested independently
   - Legacy behavior can be verified against adapter output
   - New implementations can be compared against adapters

4. **Parallel Development**
   - Teams can work on adapters and clean implementations simultaneously
   - Adapters provide a working baseline for comparison

Adapter Architecture
====================

Each adapter follows the same pattern:

.. code-block:: python

   class ExampleAdapter(ExamplePipeline):
       """
       ADAPTER: This class is a TEMPORARY MIGRATION LAYER.
       It will be replaced with a clean implementation in v6.1.0.
       """

       def __init__(self, config: SimulationConfig) -> None:
           """Store config, NOT NetworkState (stateless per-call design)."""
           self._config = config
           self._engine_props = config.to_engine_props()

       def pipeline_method(
           self,
           param1: str,
           param2: int,
           network_state: NetworkState,
       ) -> ResultType:
           """
           Pipeline interface method.

           1. Create proxy objects from NetworkState
           2. Call legacy component
           3. Convert result to Result object
           """
           # Create proxy for legacy code
           proxy = ProxyClass.from_network_state(network_state, ...)

           # Instantiate and call legacy component
           legacy = LegacyComponent(self._engine_props, proxy)
           legacy.do_something()

           # Convert to Result object
           return ResultType.from_legacy_props(legacy.props)

Proxy Classes
=============

Each adapter uses proxy classes to satisfy the legacy component's expectations:

SDNPropsProxy (RoutingAdapter)
------------------------------

Provides minimal interface for ``Routing`` class:

.. code-block:: python

   @dataclass
   class SDNPropsProxy:
       """Read-only proxy - mutations don't persist."""
       topology: nx.Graph
       source: str = ""
       destination: str = ""
       bandwidth: float = 0.0
       network_spectrum_dict: dict = field(default_factory=dict)
       lightpath_status_dict: dict = field(default_factory=dict)
       modulation_formats_dict: dict = field(default_factory=dict)

       @classmethod
       def from_network_state(
           cls,
           network_state: NetworkState,
           source: str,
           destination: str,
           bandwidth: float,
           modulation_formats_dict: dict | None = None,
       ) -> SDNPropsProxy:
           """Create from NetworkState with request context."""
           return cls(
               topology=network_state.topology,
               source=source,
               destination=destination,
               bandwidth=bandwidth,
               network_spectrum_dict=network_state.network_spectrum_dict,
               lightpath_status_dict=network_state.lightpath_status_dict,
               modulation_formats_dict=modulation_formats_dict or {},
           )

SDNPropsProxyForSpectrum (SpectrumAdapter)
------------------------------------------

Extended proxy for ``SpectrumAssignment`` class:

.. code-block:: python

   @dataclass
   class SDNPropsProxyForSpectrum:
       """Proxy for spectrum assignment operations."""
       topology: nx.Graph
       source: str = ""
       destination: str = ""
       bandwidth: float = 0.0
       network_spectrum_dict: dict = field(default_factory=dict)
       lightpath_status_dict: dict = field(default_factory=dict)
       slots_needed: int | None = None
       path_list: list[int] | None = None

Individual Adapters
===================

RoutingAdapter
--------------

:Location: ``fusion/core/adapters/routing_adapter.py``
:Size: 493 lines
:Wraps: ``fusion.core.routing.Routing``
:Protocol: ``RoutingPipeline``

**Purpose:** Find candidate routes between source and destination.

**Interface:**

.. code-block:: python

   class RoutingAdapter(RoutingPipeline):
       def find_routes(
           self,
           source: str,
           destination: str,
           bandwidth_gbps: int,
           network_state: NetworkState,
           *,
           forced_path: list[str] | None = None,
       ) -> RouteResult:
           """Find candidate routes."""

**Key responsibilities:**

- Create SDNPropsProxy from NetworkState
- Instantiate legacy Routing class
- Call ``routing.get_route()``
- Convert ``RoutingProps`` to ``RouteResult``
- Handle forced paths from partial grooming
- Sort modulation formats by efficiency

**Conversion logic:**

.. code-block:: python

   def _convert_route_props(self, route_props) -> RouteResult:
       # Convert lists to tuples (immutability)
       paths = tuple(tuple(str(n) for n in p) for p in route_props.paths_matrix)
       weights = tuple(route_props.weights_list)
       modulations = tuple(tuple(m) for m in route_props.modulation_formats_matrix)

       return RouteResult(
           paths=paths,
           weights_km=weights,
           modulations=modulations,
           strategy_name=self._engine_props.get("route_method", "legacy"),
       )

SpectrumAdapter
---------------

:Location: ``fusion/core/adapters/spectrum_adapter.py``
:Size: 567 lines
:Wraps: ``fusion.core.spectrum_assignment.SpectrumAssignment``
:Protocol: ``SpectrumPipeline``

**Purpose:** Find and allocate contiguous spectrum slots.

**Interface:**

.. code-block:: python

   class SpectrumAdapter(SpectrumPipeline):
       def find_spectrum(
           self,
           path: tuple[str, ...],
           modulations: tuple[str, ...],
           bandwidth_gbps: int,
           network_state: NetworkState,
           *,
           backup_path: tuple[str, ...] | None = None,
       ) -> SpectrumResult:
           """Find available spectrum on path."""

**Key responsibilities:**

- Create SDNPropsProxyForSpectrum from NetworkState
- Calculate slots needed from bandwidth and modulation
- Instantiate legacy SpectrumAssignment
- Call ``spectrum.get_spectrum()``
- Convert ``SpectrumProps`` to ``SpectrumResult``
- Handle backup paths for 1+1 protection

**Slot calculation:**

.. code-block:: python

   def _calculate_slots_needed(
       self,
       bandwidth_gbps: int,
       modulation: str,
   ) -> int:
       """Calculate spectrum slots needed."""
       mod_info = self._config.modulation_formats.get(modulation, {})
       bits_per_symbol = mod_info.get("bits_per_symbol", 2)
       bw_per_slot = self._config.bw_per_slot

       # Calculate required bandwidth with overhead
       data_rate = bandwidth_gbps * 1e9  # Convert to bps
       symbol_rate = data_rate / bits_per_symbol
       required_bw = symbol_rate / 1e9  # Convert to GHz

       slots = int(np.ceil(required_bw / bw_per_slot))
       return slots + self._config.guard_slots

SNRAdapter
----------

:Location: ``fusion/core/adapters/snr_adapter.py``
:Size: 402 lines
:Wraps: ``fusion.core.snr_measurements.SnrMeasurements``
:Protocol: ``SNRPipeline``

**Purpose:** Validate signal quality meets modulation requirements.

**Interface:**

.. code-block:: python

   class SNRAdapter(SNRPipeline):
       def validate_snr(
           self,
           path: tuple[str, ...],
           spectrum: SpectrumResult,
           network_state: NetworkState,
       ) -> SNRResult:
           """Validate SNR for allocated spectrum."""

**Key responsibilities:**

- Create proxy objects for SNR calculations
- Instantiate legacy SnrMeasurements
- Call ``snr.check_snr()`` or ``snr.check_gsnr()``
- Convert result to ``SNRResult``
- Handle different SNR types (calculation, external data)

GroomingAdapter
---------------

:Location: ``fusion/core/adapters/grooming_adapter.py``
:Size: 381 lines
:Wraps: ``fusion.core.grooming.Grooming``
:Protocol: ``GroomingPipeline``

**Purpose:** Find existing lightpath capacity for requests.

**Interface:**

.. code-block:: python

   class GroomingAdapter(GroomingPipeline):
       def try_groom(
           self,
           request: Request,
           network_state: NetworkState,
       ) -> GroomingResult:
           """Try to groom request onto existing lightpaths."""

**Key responsibilities:**

- Create proxy for Grooming class
- Call ``grooming.handle_grooming()``
- Track partial grooming with remaining bandwidth
- Identify forced path for new lightpath
- Convert to ``GroomingResult``

Migration Status
================

.. list-table::
   :header-rows: 1
   :widths: 20 15 30 35

   * - Adapter
     - Status
     - Coverage
     - Notes
   * - RoutingAdapter
     - Complete
     - All routing algorithms
     - Ready for clean implementation
   * - SpectrumAdapter
     - Complete
     - All allocation strategies
     - Multi-core and multi-band supported
   * - SNRAdapter
     - Complete
     - Basic SNR validation
     - GSNR integration in progress
   * - GroomingAdapter
     - Complete
     - Full and partial grooming
     - Ready for clean implementation

What Still Needs Migration
--------------------------

1. **Clean Implementations**
   - Rewrite adapters as direct pipeline implementations
   - Remove proxy class dependency
   - Use NetworkState and config directly

2. **External SNR Data**
   - Improve integration with pre-calculated SNR files
   - Better handling of multi-fiber vs. MCF modes

3. **Slicing Pipeline**
   - Currently implemented directly (not via adapter)
   - Needs integration testing with adapters

4. **Protection Pipeline**
   - Currently implemented directly
   - Backup path allocation via SpectrumAdapter

Contributing to Adapters
========================

Guidelines for Working with Adapters
------------------------------------

1. **Don't extend adapters for new features**
   - If adding a new pipeline stage, implement directly against protocols
   - Adapters are temporary and will be removed

2. **Bug fixes are welcome**
   - If you find a bug in adapter conversion logic, please fix it
   - Add tests to prevent regression

3. **Document edge cases**
   - If you discover edge cases not handled, document them
   - Consider if the issue is in the adapter or legacy code

4. **Maintain backward compatibility**
   - Adapter output should match legacy behavior
   - Breaking changes require coordination

Testing Adapters
----------------

Each adapter has corresponding tests:

.. code-block:: bash

   # Run adapter tests
   pytest fusion/core/adapters/tests/ -v

   # Run specific adapter test
   pytest fusion/core/adapters/tests/test_routing_adapter.py -v

**Test strategy:**

1. Unit tests verify adapter conversion logic
2. Integration tests verify adapter + legacy produce correct results
3. Comparison tests verify adapter output matches legacy path

Example: Creating a New Adapter
-------------------------------

If you need to create an adapter for a new legacy component:

.. code-block:: python

   """
   NewAdapter - Adapts legacy NewComponent to NewPipeline protocol.

   ADAPTER: This class is a TEMPORARY MIGRATION LAYER.
   It will be replaced with a clean implementation in v6.1.0.
   """

   from fusion.interfaces.pipelines import NewPipeline

   class NewPropsProxy:
       """Minimal proxy for NewComponent."""

       @classmethod
       def from_network_state(cls, network_state, ...):
           return cls(...)

   class NewAdapter(NewPipeline):
       """Adapts legacy NewComponent to NewPipeline protocol."""

       def __init__(self, config: SimulationConfig) -> None:
           self._config = config
           self._engine_props = config.to_engine_props()

       def pipeline_method(self, ..., network_state) -> NewResult:
           # 1. Create proxy
           proxy = NewPropsProxy.from_network_state(network_state, ...)

           # 2. Call legacy
           from fusion.core.new_component import NewComponent
           legacy = NewComponent(self._engine_props, proxy)
           legacy.do_work()

           # 3. Convert result
           return NewResult.from_legacy(legacy.props)

See Also
========

- :doc:`architecture` - Legacy vs. orchestrator architecture
- :doc:`orchestrator` - Pipeline flow documentation
- :doc:`data_structures` - Result object documentation
- ``fusion/interfaces/pipelines.py`` - Pipeline protocol definitions
