# Component 8: Simulation Integration

**File:** `fusion/sim/network_simulator.py`
**Estimated Time:** 30 minutes
**Dependencies:** Components 1, 2 (Properties, Grooming)

## Overview

Initialize grooming data structures in the simulation engine and ensure proper reset between iterations.

## Changes Required

### 1. Add Grooming Initialization to reset()

Add grooming structure initialization to the simulation reset method:

```python
def reset(self) -> None:
    """
    Reset simulation state for new iteration.

    Clears all tracking dictionaries and counters to prepare
    for a fresh simulation run.
    """
    # Existing resets
    # ... existing code ...

    # NEW: Reset grooming structures
    if self.engine_props.get('is_grooming_enabled', False):
        self.sdn_obj.sdn_props.reset_lightpath_id_counter()
        self.sdn_obj.sdn_props.lightpath_status_dict = {}
        self.sdn_obj.grooming_obj.grooming_props.lightpath_status_dict = {}
        self.sdn_obj.sdn_props.lp_bw_utilization_dict = {}

        logger.debug("Reset grooming structures for new iteration")
```

### 2. Initialize Transponder Usage Dictionary

Add transponder tracking initialization:

```python
def init_iter(
    self,
    iteration: int,
    seed: int | None = None,
    print_flag: bool = True,
    trial: int | None = None
) -> None:
    """
    Initialize a simulation iteration.

    :param iteration: Iteration number
    :type iteration: int
    :param seed: Random seed for reproducibility
    :type seed: int | None
    :param print_flag: Whether to print iteration info
    :type print_flag: bool
    :param trial: Trial number for multi-trial runs
    :type trial: int | None
    """
    # Existing initialization
    # ... existing code ...

    # NEW: Initialize transponder usage per node
    if self.engine_props.get('transponder_usage_per_node', False):
        self._init_transponder_usage()


def _init_transponder_usage(self) -> None:
    """
    Initialize transponder usage tracking for all nodes.

    Sets up the transponder_usage_dict with initial transponder
    counts for each node in the network.
    """
    if self.sdn_obj.sdn_props.topology is None:
        logger.warning("Cannot initialize transponder usage: topology not set")
        return

    self.sdn_obj.sdn_props.transponder_usage_dict = {}

    # Get initial transponder count from config
    initial_transponders = self.engine_props.get('transponders_per_node', 10)

    for node in self.sdn_obj.sdn_props.topology.nodes():
        self.sdn_obj.sdn_props.transponder_usage_dict[node] = {
            'available_transponder': initial_transponders,
            'total_transponder': initial_transponders
        }

    logger.debug(
        "Initialized transponder usage for %d nodes (%d transponders each)",
        len(self.sdn_obj.sdn_props.transponder_usage_dict),
        initial_transponders
    )
```

### 3. Add Grooming Statistics Collection

Add statistics collection for grooming metrics:

```python
def _collect_iteration_stats(self) -> None:
    """
    Collect statistics at end of iteration.

    Gathers all performance metrics including grooming-specific
    statistics if grooming is enabled.
    """
    # Existing statistics collection
    # ... existing code ...

    # NEW: Collect grooming statistics
    if self.engine_props.get('is_grooming_enabled', False):
        self._collect_grooming_stats()


def _collect_grooming_stats(self) -> None:
    """
    Collect grooming-specific statistics.

    Calculates and stores metrics related to traffic grooming
    performance including grooming success rate and bandwidth utilization.
    """
    if not hasattr(self, 'grooming_stats'):
        self.grooming_stats = {
            'fully_groomed': 0,
            'partially_groomed': 0,
            'not_groomed': 0,
            'lightpaths_created': 0,
            'lightpaths_released': 0,
            'avg_lightpath_utilization': []
        }

    # Count grooming outcomes
    # (This would be updated during request processing)

    # Calculate average lightpath utilization
    if self.sdn_obj.sdn_props.lp_bw_utilization_dict:
        utilizations = [
            lp_info['utilization']
            for lp_info in self.sdn_obj.sdn_props.lp_bw_utilization_dict.values()
        ]
        avg_util = sum(utilizations) / len(utilizations) if utilizations else 0.0
        self.grooming_stats['avg_lightpath_utilization'].append(avg_util)

        logger.info(
            "Grooming stats: %d lightpaths, avg utilization: %.2f%%",
            len(utilizations), avg_util
        )
```

### 4. Update Request Processing

Ensure grooming flags are tracked during request processing:

```python
def _process_arrival_request(self, request: dict) -> None:
    """
    Process a new arrival request.

    :param request: Request dictionary with source, dest, bandwidth, etc.
    :type request: dict
    """
    # Existing request setup
    # ... existing code ...

    # Allocate request (grooming happens inside sdn_controller)
    self.sdn_obj.allocate_request()

    # NEW: Track grooming outcome
    if self.engine_props.get('is_grooming_enabled', False):
        if self.sdn_obj.sdn_props.was_groomed:
            self.grooming_stats['fully_groomed'] += 1
        elif self.sdn_obj.sdn_props.was_partially_groomed:
            self.grooming_stats['partially_groomed'] += 1
        else:
            self.grooming_stats['not_groomed'] += 1

        # Track new lightpaths
        if self.sdn_obj.sdn_props.was_new_lp_established:
            self.grooming_stats['lightpaths_created'] += len(
                self.sdn_obj.sdn_props.was_new_lp_established
            )
```

### 5. Add Configuration Validation

Validate grooming configuration at startup:

```python
def _validate_grooming_config(self) -> None:
    """
    Validate grooming-related configuration.

    Checks that grooming configuration options are consistent
    and compatible with other simulation settings.
    """
    if not self.engine_props.get('is_grooming_enabled', False):
        return

    # Check for required settings
    if 'transponders_per_node' not in self.engine_props:
        logger.warning(
            "transponders_per_node not set, using default value of 10"
        )
        self.engine_props['transponders_per_node'] = 10

    # Validate SNR rechecking settings
    if self.engine_props.get('snr_recheck', False):
        if self.engine_props.get('snr_type') in ['None', None]:
            logger.warning(
                "snr_recheck enabled but snr_type is None - rechecking will be skipped"
            )

    # Validate partial service setting
    if self.engine_props.get('can_partially_serve', False):
        logger.info("Partial service allocation enabled")

    logger.debug("Grooming configuration validated")
```

Call validation in initialization:

```python
def __init__(self, engine_props: dict[str, Any]) -> None:
    """Initialize network simulator."""
    # ... existing init code ...

    # NEW: Validate grooming config
    self._validate_grooming_config()
```

## Testing

Create tests in `fusion/sim/tests/test_network_simulator.py`:

```python
def test_grooming_initialization(engine_props):
    """Test grooming structures are initialized."""
    engine_props['is_grooming_enabled'] = True

    sim = NetworkSimulator(engine_props)
    sim.reset()

    assert hasattr(sim.sdn_obj.sdn_props, 'lightpath_status_dict')
    assert sim.sdn_obj.sdn_props.lightpath_status_dict == {}
    assert sim.sdn_obj.sdn_props.lightpath_counter == 0


def test_transponder_usage_initialization(engine_props, topology):
    """Test transponder usage dict is created."""
    engine_props['transponder_usage_per_node'] = True
    engine_props['transponders_per_node'] = 15

    sim = NetworkSimulator(engine_props)
    sim.sdn_obj.sdn_props.topology = topology
    sim._init_transponder_usage()

    assert sim.sdn_obj.sdn_props.transponder_usage_dict is not None
    for node in topology.nodes():
        assert node in sim.sdn_obj.sdn_props.transponder_usage_dict
        assert sim.sdn_obj.sdn_props.transponder_usage_dict[node]['available_transponder'] == 15


def test_grooming_stats_collection(engine_props):
    """Test grooming statistics are collected."""
    engine_props['is_grooming_enabled'] = True

    sim = NetworkSimulator(engine_props)
    sim._collect_grooming_stats()

    assert hasattr(sim, 'grooming_stats')
    assert 'fully_groomed' in sim.grooming_stats
    assert 'partially_groomed' in sim.grooming_stats
```

Run tests:

```bash
python -m pylint fusion/sim/network_simulator.py
python -m mypy fusion/sim/network_simulator.py
python -m pytest fusion/sim/tests/test_network_simulator.py -v -k grooming
```

## Validation Checklist

- [ ] `reset()` method clears grooming structures
- [ ] Lightpath ID counter reset implemented
- [ ] Transponder usage initialization added
- [ ] `_init_transponder_usage()` method created
- [ ] Grooming statistics collection added
- [ ] Request processing tracks grooming outcomes
- [ ] Configuration validation added
- [ ] All methods have proper type hints
- [ ] Logging added for debugging
- [ ] Code passes pylint and mypy
- [ ] Unit tests created and passing

## Next Component

After completing this component, proceed to: [Component 9: Statistics](09-statistics.md)
