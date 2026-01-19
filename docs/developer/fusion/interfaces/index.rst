.. _interfaces-module:

=================
Interfaces Module
=================

Overview
========

.. admonition:: At a Glance
   :class: tip

   :Purpose: Define contracts (protocols) for pluggable algorithm components
   :Location: ``fusion/interfaces/``
   :Key Files: ``pipelines.py``, ``control_policy.py``, ``router.py``, ``spectrum.py``
   :Depends On: ``fusion.domain`` (type references only)
   :Used By: ``fusion.core``, ``fusion.pipelines``, ``fusion.modules``

The ``interfaces`` module defines **contracts** that algorithm implementations must satisfy.
It uses Python's ``typing.Protocol`` classes to specify what methods a routing algorithm,
spectrum assigner, or control policy must provide without prescribing how they work internally.

**Why this module matters:**

- Enables pluggable algorithms: swap routing strategies without changing orchestration code
- Provides type safety: mypy catches incompatible implementations before runtime
- Documents expected behavior: protocols serve as executable specification
- Supports testing: mock implementations satisfy protocols for unit testing

.. rubric:: In This Section

* :doc:`protocols` - Detailed documentation of each pipeline protocol
* :doc:`creating` - How to create your own algorithm implementations

**When you work here:**

- Understanding how algorithms plug into the simulation pipeline
- Creating a new routing, spectrum, or SNR algorithm
- Adding a custom RL or heuristic control policy
- Debugging why an implementation doesn't satisfy a protocol

How Interfaces Differ from Domain and Modules
=============================================

New developers often confuse interfaces, domain objects, and module implementations.
Here's how they relate:

.. code-block:: text

   +------------------+     +------------------+     +------------------+
   |    Interfaces    |     |      Domain      |     |     Modules      |
   |   (Contracts)    |     |  (Data Objects)  |     |(Implementations) |
   +------------------+     +------------------+     +------------------+
          |                        |                        |
          | "What methods          | "What data             | "How to actually
          |  must exist"           |  flows between"        |  compute routes"
          |                        |                        |
          v                        v                        v
   +------------------+     +------------------+     +------------------+
   | RoutingPipeline  |     | RouteResult      |     | KShortestPath    |
   |   Protocol:      |     |   (frozen):      |     |   Class:         |
   |   find_routes()  |<----|   paths          |<----| find_routes()    |
   |   -> RouteResult |     |   weights_km     |     |   returns        |
   +------------------+     +------------------+     +------------------+
          ^                        ^                        |
          |                        |                        |
          | type checks            | data types             | implements
          | implementation         | for results            | protocol
          |                        |                        |
   +------+--------+        +------+--------+        +------+--------+
   | Orchestrator  |------->| NetworkState  |<-------| Algorithm     |
   | uses protocol |        | passed to     |        | reads from    |
   | type hints    |        | algorithms    |        | state         |
   +---------------+        +---------------+        +---------------+

.. list-table:: Module Comparison
   :header-rows: 1
   :widths: 18 27 27 28

   * - Aspect
     - **Interfaces**
     - **Domain**
     - **Modules**
   * - Purpose
     - Define method contracts
     - Define data structures
     - Implement algorithms
   * - Contains
     - Protocol classes (no logic)
     - Dataclasses (no logic)
     - Classes with algorithm logic
   * - Example
     - ``RoutingPipeline``
     - ``RouteResult``
     - ``KShortestPathRouting``
   * - Mutability
     - N/A (type definitions)
     - Mostly immutable
     - May have internal state
   * - Location
     - ``fusion/interfaces/``
     - ``fusion/domain/``
     - ``fusion/modules/routing/``

**In short:**

- **Interfaces**: Define *what methods* an algorithm must have (the contract)
- **Domain**: Define *what data* flows between components (the vocabulary)
- **Modules**: Provide *actual implementations* of algorithms (the logic)

Understanding Policies vs Interfaces
====================================

Another common confusion is between control policies and pipeline interfaces:

Control Policy
   A decision-making component that **selects** which path to use from available options.
   Policies can be heuristics (first-fit, shortest), RL agents (PPO, DQN), or
   supervised/unsupervised learning models.

Pipeline Interface
   A computation component that **generates** options or validates constraints.
   Pipelines find routes, check spectrum availability, or validate SNR.

.. code-block:: text

   Request arrives
        |
        v
   +------------------+
   | RoutingPipeline  |  <-- Interface: generates candidate paths
   | find_routes()    |
   +------------------+
        |
        | returns paths=[A-B-C, A-D-C, A-E-C]
        v
   +------------------+
   | SpectrumPipeline |  <-- Interface: checks spectrum for each path
   | find_spectrum()  |
   +------------------+
        |
        | returns options with feasibility info
        v
   +------------------+
   | ControlPolicy    |  <-- Policy: SELECTS which option to use
   | select_action()  |      (heuristic, RL, or supervised/unsupervised)
   +------------------+
        |
        | returns selected path index
        v
   +------------------+
   | Orchestrator     |
   | allocates path   |
   +------------------+

**Key difference:**

- Pipelines are *stateless computations* that don't make decisions
- Policies are *decision-makers* that choose among computed options

Key Concepts
============

Understanding these concepts is essential before working with interfaces:

Protocol (typing.Protocol)
   A Python typing construct that defines structural subtyping. Unlike abstract base
   classes, protocols don't require inheritance; any class with matching methods
   satisfies the protocol. This enables duck typing with type safety.

   .. code-block:: python

      from typing import Protocol

      class RoutingPipeline(Protocol):
          def find_routes(self, source, dest, bw, state) -> RouteResult: ...

      # Any class with find_routes() satisfies RoutingPipeline
      # No inheritance required!

runtime_checkable
   A decorator that enables ``isinstance()`` checks on protocols at runtime.
   All FUSION pipeline protocols are runtime_checkable.

   .. code-block:: python

      @runtime_checkable
      class RoutingPipeline(Protocol): ...

      # Now you can do:
      if isinstance(my_router, RoutingPipeline):
          result = my_router.find_routes(...)

Pipeline
   A component that performs one stage of request processing. Pipelines receive
   ``NetworkState`` as a parameter (never store it) and return immutable result
   objects.

Stateless Design
   Pipelines and the orchestrator don't store ``NetworkState``. State is passed
   per-call, enabling easier testing and potential parallelization.

Architecture
============

Module Structure
----------------

.. code-block:: text

   fusion/interfaces/
   |-- __init__.py           # Public API exports
   |-- pipelines.py          # Pipeline protocols (Routing, Spectrum, etc.)
   |-- control_policy.py     # ControlPolicy protocol for path selection
   |-- router.py             # Legacy: AbstractRoutingAlgorithm ABC
   |-- spectrum.py           # Legacy: AbstractSpectrumAssigner ABC
   |-- snr.py                # Legacy: AbstractSNRMeasurer ABC
   |-- agent.py              # Legacy: AgentInterface ABC
   |-- factory.py            # Legacy: AlgorithmFactory, SimulationPipeline
   |-- README.md             # Module documentation
   |-- TODO.md               # Legacy removal roadmap
   `-- tests/                # Unit tests
       |-- test_pipelines.py
       |-- test_control_policy.py
       `-- ...

Pipeline Protocols
------------------

The main interfaces are defined in ``pipelines.py``:

.. list-table:: Pipeline Protocols
   :header-rows: 1
   :widths: 25 30 45

   * - Protocol
     - Key Method
     - Purpose
   * - ``RoutingPipeline``
     - ``find_routes()``
     - Find candidate paths between source and destination
   * - ``SpectrumPipeline``
     - ``find_spectrum()``
     - Find available spectrum slots along a path
   * - ``GroomingPipeline``
     - ``try_groom()``
     - Pack requests onto existing lightpaths
   * - ``SNRPipeline``
     - ``validate()``
     - Check signal quality meets modulation requirements
   * - ``SlicingPipeline``
     - ``try_slice()``
     - Divide large requests into smaller allocations

Control Policy Protocol
-----------------------

The ``ControlPolicy`` protocol in ``control_policy.py`` defines the decision-making interface:

.. code-block:: python

   @runtime_checkable
   class ControlPolicy(Protocol):
       def select_action(
           self,
           request: Request,
           options: list[PathOption],
           network_state: NetworkState,
       ) -> int:
           """Select which path option to use. Returns index or -1."""
           ...

       def update(self, request: Request, action: int, reward: float) -> None:
           """Update policy based on feedback (for RL policies)."""
           ...

       def get_name(self) -> str:
           """Return policy name for logging."""
           ...

How Interfaces Connect to the Simulator
=======================================

The orchestrator uses pipeline protocols to process requests:

.. code-block:: text

   SimulationEngine
        |
        | creates
        v
   +------------------+
   | SDNOrchestrator  |
   |                  |
   | routing: RoutingPipeline      <-- Protocol type hints
   | spectrum: SpectrumPipeline
   | grooming: GroomingPipeline?
   | snr: SNRPipeline?
   | slicing: SlicingPipeline?
   | policy: ControlPolicy?
   +------------------+
        |
        | handle_arrival(request, network_state)
        v
   +------------------+
   | 1. routing.find_routes()      --> RouteResult
   | 2. spectrum.find_spectrum()   --> SpectrumResult
   | 3. snr.validate()             --> SNRResult
   | 4. policy.select_action()     --> int (path index)
   | 5. network_state.allocate()   --> lightpath_id
   +------------------+
        |
        v
   AllocationResult (success or blocked)

**Key points:**

1. The orchestrator type-hints pipelines using protocol types
2. Actual implementations come from ``fusion/pipelines/`` or adapters
3. Any class satisfying the protocol can be substituted
4. This enables easy testing with mock implementations

Components
==========

pipelines.py
------------

:Purpose: Define pipeline protocols for the orchestrator
:Key Classes: ``RoutingPipeline``, ``SpectrumPipeline``, ``GroomingPipeline``, ``SNRPipeline``, ``SlicingPipeline``

Contains the protocol definitions that all pipeline implementations must satisfy.
Each protocol specifies:

- Required method signatures
- Parameter types (using domain objects)
- Return types (using result objects)
- Optional parameters with defaults

See :doc:`protocols` for detailed documentation of each protocol.

control_policy.py
-----------------

:Purpose: Define the decision-making interface for path selection
:Key Classes: ``ControlPolicy``
:Type Alias: ``PolicyAction`` (int)

Defines the protocol for control policies that select actions from available options.
Supports three types of policies:

- **Heuristic policies**: Rule-based selection (first-fit, shortest-path)
- **RL policies**: Reinforcement learning agents (PPO, DQN, IQL)
- **Supervised/unsupervised policies**: Pre-trained models for inference

Legacy Files (Deprecated)
-------------------------

The following files contain legacy abstract base classes that are being replaced
by the protocol-based approach:

- ``router.py``: ``AbstractRoutingAlgorithm`` ABC
- ``spectrum.py``: ``AbstractSpectrumAssigner`` ABC
- ``snr.py``: ``AbstractSNRMeasurer`` ABC
- ``agent.py``: ``AgentInterface`` ABC
- ``factory.py``: ``AlgorithmFactory``, ``SimulationPipeline``

.. warning::

   Legacy ABCs are deprecated and will be removed in a future version.
   New implementations should use the Protocol-based interfaces in ``pipelines.py``
   and ``control_policy.py``.

Dependencies
============

This Module Depends On
----------------------

- ``fusion.domain`` - Type references for ``Request``, ``NetworkState``, result objects
- ``typing`` - ``Protocol``, ``runtime_checkable``, type hints

Modules That Depend On This
---------------------------

- ``fusion.core.orchestrator`` - Uses protocol type hints for pipelines
- ``fusion.core.pipeline_factory`` - Creates implementations satisfying protocols
- ``fusion.pipelines/`` - Implements the pipeline protocols
- ``fusion.modules/rl/`` - Implements ControlPolicy protocol

Development Guide
=================

Getting Started
---------------

1. Read the `Key Concepts`_ section above
2. Understand the difference between interfaces and implementations
3. Review :doc:`protocols` for protocol signatures
4. See :doc:`creating` for step-by-step implementation guides

Common Tasks
------------

**Creating a new routing algorithm**

See :doc:`creating` for a complete guide. Quick overview:

1. Create a class with ``find_routes()`` method matching ``RoutingPipeline``
2. Accept ``NetworkState`` as parameter (don't store it)
3. Return ``RouteResult`` from ``fusion.domain.results``
4. Register with pipeline factory if needed

**Creating a custom control policy**

1. Create a class with ``select_action()``, ``update()``, ``get_name()``
2. Accept ``PathOption`` list and return selected index
3. Implement ``update()`` for RL (or ``pass`` for heuristics)

**Adding a new protocol method**

1. Add the method signature to the protocol in ``pipelines.py``
2. Update all implementations (check with mypy)
3. Update mock implementations in tests
4. Document the new method

Testing
=======

:Test Location: ``fusion/interfaces/tests/``
:Run Tests: ``pytest fusion/interfaces/tests/ -v``

**Test files:**

- ``test_pipelines.py`` - Protocol isinstance checks, mock implementations
- ``test_control_policy.py`` - ControlPolicy protocol tests

**Testing with protocols:**

Protocols enable easy testing with mock implementations:

.. code-block:: python

   from fusion.interfaces import RoutingPipeline

   class MockRouter:
       """Mock that satisfies RoutingPipeline for testing."""

       def find_routes(self, source, dest, bw, state, *, forced_path=None):
           from fusion.domain.results import RouteResult
           return RouteResult.empty("mock")

   # Use in tests
   def test_orchestrator_handles_no_routes():
       mock_router = MockRouter()
       assert isinstance(mock_router, RoutingPipeline)  # Protocol check passes
       orchestrator = create_orchestrator(routing=mock_router)
       result = orchestrator.handle_arrival(request, state)
       assert result.block_reason == BlockReason.NO_PATH

Troubleshooting
===============

**Issue: Implementation doesn't satisfy protocol**

:Symptom: mypy error "Argument has incompatible type"
:Cause: Method signature doesn't match protocol exactly
:Solution: Check parameter types, return type, and optional parameters match

**Issue: isinstance() check fails on valid implementation**

:Symptom: ``isinstance(obj, Protocol)`` returns False unexpectedly
:Cause: Protocol missing ``@runtime_checkable`` decorator, or method missing
:Solution: Ensure all required methods exist with correct names

**Issue: Confusion about where to add logic**

:Symptom: Unsure whether code belongs in interface, domain, or module
:Solution:

- If it's a method signature: interfaces
- If it's a data structure: domain
- If it's algorithm logic: modules (e.g., ``fusion/modules/routing/``)

Related Documentation
=====================

- :ref:`domain-module` - Domain objects used by interfaces
- :ref:`core-module` - How core uses interfaces
- :ref:`core-orchestrator` - Pipeline flow through orchestrator
- :ref:`core-architecture` - Legacy vs orchestrator architecture

.. toctree::
   :maxdepth: 2
   :caption: Contents

   protocols
   creating
