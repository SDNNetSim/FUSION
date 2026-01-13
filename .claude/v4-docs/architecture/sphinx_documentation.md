# Sphinx Documentation for V4 Domain Model

This document describes how to document Phase 1 domain objects for Sphinx autodoc.

## Docstring Format

Use Sphinx-style docstrings (compatible with existing `docs/conf.py`):

```python
@dataclass(frozen=True)
class SimulationConfig:
    """
    Immutable simulation configuration.

    This dataclass holds all configuration parameters for a simulation run.
    It is created once at simulation start and never modified.

    :param network_name: Name of the network topology file
    :param cores_per_link: Number of spatial cores per fiber link
    :param band_list: Tuple of spectrum bands to use ("c", "l", "s")
    :param band_slots: Mapping of band name to number of slots
    :param guard_slots: Guard band slots between allocations
    :param num_requests: Total number of requests to generate
    :param erlang: Traffic load in Erlangs
    :param holding_time: Mean holding time for requests
    :param route_method: Routing algorithm name
    :param k_paths: Number of candidate paths for KSP
    :param allocation_method: Spectrum allocation algorithm
    :param grooming_enabled: Whether grooming is active
    :param slicing_enabled: Whether slicing is active
    :param max_slices: Maximum slices per request
    :param snr_enabled: Whether SNR validation is active
    :param snr_type: Type of SNR calculation
    :param snr_recheck: Whether to recheck SNR after allocation
    :param can_partially_serve: Allow partial bandwidth allocation
    :param modulation_formats: Modulation format specifications
    :param mod_per_bw: Bandwidth to modulation mapping
    :param snr_thresholds: Required SNR per modulation format

    Example::

        config = SimulationConfig.from_engine_props(engine_props)
        print(f"Running {config.num_requests} requests on {config.network_name}")
    """
```

## Method Documentation

```python
@classmethod
def from_engine_props(cls, engine_props: dict) -> "SimulationConfig":
    """
    Create configuration from legacy engine_props dictionary.

    This factory method converts the legacy dictionary-based configuration
    to the new typed dataclass format.

    :param engine_props: Legacy configuration dictionary
    :type engine_props: dict
    :returns: New SimulationConfig instance
    :rtype: SimulationConfig
    :raises KeyError: If required keys are missing (none currently required)

    Example::

        engine_props = {"network": "USbackbone60", "k_paths": 3, ...}
        config = SimulationConfig.from_engine_props(engine_props)

    .. note::
        Missing keys use sensible defaults. See source for default values.
    """
```

## Property Documentation

```python
@property
def endpoint_key(self) -> tuple[str, str]:
    """
    Canonical (sorted) endpoint tuple for lookups.

    Returns a tuple of (source, destination) sorted alphabetically.
    This ensures consistent keys regardless of direction.

    :returns: Sorted tuple of endpoints
    :rtype: tuple[str, str]

    Example::

        req = Request(source="Z", destination="A", ...)
        assert req.endpoint_key == ("A", "Z")
    """
    return tuple(sorted([self.source, self.destination]))
```

## Enum Documentation

```python
class RequestStatus(Enum):
    """
    Lifecycle states for a network request.

    A request progresses through these states during simulation:

    .. graphviz::

        digraph states {
            PENDING -> ROUTED;
            PENDING -> BLOCKED;
            ROUTED -> RELEASED;
        }

    :cvar PENDING: Request created, not yet processed
    :cvar ROUTED: Successfully allocated
    :cvar BLOCKED: Failed allocation (terminal)
    :cvar RELEASED: Departed, resources freed (terminal)
    """
    PENDING = auto()
    ROUTED = auto()
    BLOCKED = auto()
    RELEASED = auto()
```

## Module Documentation

Each module should have a module-level docstring:

```python
"""
Domain model for V4 architecture.

This module contains the core domain objects that replace dictionary-based
data structures in the legacy codebase.

Classes:
    SimulationConfig: Immutable simulation configuration
    Request: Network service request with lifecycle
    Lightpath: Allocated optical path with capacity
    RequestStatus: Request lifecycle states
    BlockReason: Request blocking reasons

Example::

    from fusion.domain import SimulationConfig, Request, Lightpath

    config = SimulationConfig.from_engine_props(engine_props)
    request = Request.from_legacy_dict(time_key, request_dict)

.. seealso::
    :doc:`/architecture/domain_model` for design rationale
"""
```

## RST Documentation Files

### docs/api/domain.rst

```rst
Domain Model
============

.. automodule:: fusion.domain
   :members:
   :undoc-members:
   :show-inheritance:

SimulationConfig
----------------

.. autoclass:: fusion.domain.config.SimulationConfig
   :members:
   :undoc-members:
   :show-inheritance:

Request
-------

.. autoclass:: fusion.domain.request.Request
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: fusion.domain.request.RequestStatus
   :members:
   :undoc-members:

.. autoclass:: fusion.domain.request.BlockReason
   :members:
   :undoc-members:

Lightpath
---------

.. autoclass:: fusion.domain.lightpath.Lightpath
   :members:
   :undoc-members:
   :show-inheritance:
```

### docs/api/results.rst

```rst
Result Objects
==============

.. automodule:: fusion.domain.results
   :members:
   :undoc-members:
   :show-inheritance:

RouteResult
-----------

.. autoclass:: fusion.domain.results.RouteResult
   :members:
   :undoc-members:
   :show-inheritance:

SpectrumResult
--------------

.. autoclass:: fusion.domain.results.SpectrumResult
   :members:
   :undoc-members:
   :show-inheritance:

AllocationResult
----------------

.. autoclass:: fusion.domain.results.AllocationResult
   :members:
   :undoc-members:
   :show-inheritance:
```

## Sphinx Configuration

The existing `docs/conf.py` should work. Verify these settings:

```python
# docs/conf.py

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # For Google/NumPy style docstrings
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'myst_parser',  # For Markdown support
]

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'member-order': 'bysource',
}

# Napoleon settings (if using Google style)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
```

## Building Documentation

```bash
cd docs
make html

# View locally
open _build/html/index.html

# Check for warnings
make html 2>&1 | grep -i warning
```

## Type Hints in Documentation

Sphinx autodoc extracts type hints from annotations:

```python
def find_routes(
    self,
    source: str,
    destination: str,
    bandwidth_gbps: int,
    network_state: NetworkState,
    forced_path: list[str] | None = None,
) -> RouteResult:
    """Find candidate routes."""
    ...
```

Sphinx will display the types automatically. For complex types, add explicit `:type:` and `:rtype:` directives.

## Cross-References

Link to other documented items:

```python
"""
Create a lightpath.

:returns: New lightpath object
:rtype: :class:`~fusion.domain.lightpath.Lightpath`

.. seealso::
    :meth:`release_lightpath` for deallocation
    :class:`~fusion.domain.results.SpectrumResult` for spectrum assignment
"""
```

## Versioning

Add version info for new classes:

```python
class SimulationConfig:
    """
    Immutable simulation configuration.

    .. versionadded:: 4.0
        Replaces legacy ``engine_props`` dictionary.
    """
```
