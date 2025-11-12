# Component 1: Properties

**File:** `fusion/core/properties.py`
**Estimated Time:** 30 minutes
**Dependencies:** None (foundation component)

## Overview

Update the core properties module to support grooming functionality. This involves:
1. Adding a new `GroomingProps` class
2. Extending `SDNProps` with grooming-related attributes
3. Extending `SpectrumProps` with lightpath tracking

## Changes Needed

### 1. Add GroomingProps Class

Add this new class to `fusion/core/properties.py`:

```python
class GroomingProps:
    """
    Main properties used for traffic grooming operations.

    This class manages properties specific to traffic grooming,
    including grooming type selection and lightpath status tracking.
    """

    def __init__(self) -> None:
        """Initialize grooming properties with default values."""
        self.grooming_type: str | None = None  # Grooming method selection
        self.lightpath_status_dict: dict | None = None  # Established lightpath status

    def __repr__(self) -> str:
        """
        Return string representation of GroomingProps.

        :return: String representation with all properties
        :rtype: str
        """
        return f"GroomingProps({self.__dict__})"
```

**Note:** v5.5 had a typo "gooming_type" - fixed in v6.0

### 2. Extend SDNProps

Add the following attributes to the `SDNProps.__init__()` method:

```python
# Lightpath tracking dictionaries
self.lightpath_status_dict: dict | None = None
self.transponder_usage_dict: dict | None = None
self.lp_bw_utilization_dict: dict | None = None

# Grooming state flags
self.was_groomed: bool | None = None
self.was_partially_groomed: bool = False
self.was_partially_routed: bool = False
self.was_new_lp_established: list[int] = []

# Lightpath resource tracking
self.lightpath_id_list: list[int] = []
self.lightpath_bandwidth_list: list[float] = []
self.remaining_bw: int | float | str | None = None

# Lightpath ID counter
self.lightpath_counter: int = 0
```

Update `stat_key_list` to include grooming-related keys:

```python
self.stat_key_list: list[str] = [
    "modulation_list",
    "crosstalk_list",
    "core_list",
    "band_list",
    "start_slot_list",
    "end_slot_list",
    "lightpath_bandwidth_list",  # NEW
    "lightpath_id_list",         # NEW
    "remaining_bw",              # NEW
]
```

Add new methods to `SDNProps`:

```python
def get_lightpath_id(self) -> int:
    """
    Generate and return a new unique lightpath ID.

    :return: New lightpath ID
    :rtype: int
    """
    self.lightpath_counter += 1
    return self.lightpath_counter

def reset_lightpath_id_counter(self) -> None:
    """
    Reset the lightpath ID counter to zero.

    Called at the start of each simulation iteration.
    """
    self.lightpath_counter = 0
```

Update the `reset_params()` method to include grooming fields:

```python
def reset_params(self) -> None:
    """
    Reset statistical tracking lists to empty state.

    This method clears all lists used for tracking per-request statistics,
    typically called before processing a new request.
    """
    # Existing resets
    self.modulation_list = []
    self.crosstalk_list = []
    self.core_list = []
    self.band_list = []
    self.start_slot_list = []
    self.end_slot_list = []
    self.bandwidth_list = []

    # NEW: Grooming-related resets
    self.lightpath_bandwidth_list = []
    self.lightpath_id_list = []
    self.lightpath_id_list = []
    self.path_list = []
    self.was_new_lp_established = []
    self.lp_bw_utilization_dict = {}

    # Reset flags
    self.was_routed = None
    self.was_groomed = None
    self.was_partially_groomed = False
    self.was_partially_routed = False
    self.remaining_bw = None
    self.is_sliced = None
    self.path_weight = None
```

### 3. Extend SpectrumProps

Add lightpath tracking to `SpectrumProps.__init__()`:

```python
# Lightpath tracking (for grooming)
self.lightpath_id: int | None = None
self.lightpath_bandwidth: float | None = None
```

## Lightpath Status Dictionary Structure

The `lightpath_status_dict` has the following structure:

```python
lightpath_status_dict = {
    (source, dest): {  # Tuple of sorted node IDs
        lightpath_id: {
            "path": [node1, node2, node3, ...],
            "path_weight": float,
            "core": int,
            "band": str,
            "start_slot": int,
            "end_slot": int,
            "mod_format": str,
            "lightpath_bandwidth": float,
            "remaining_bandwidth": float,
            "snr_cost": float,
            "is_degraded": bool,
            "requests_dict": {
                request_id: bandwidth_allocated
            },
            "time_bw_usage": {
                timestamp: utilization_percentage
            }
        }
    }
}
```

## Transponder Usage Dictionary Structure

```python
transponder_usage_dict = {
    node_id: {
        "available_transponder": int
    }
}
```

## Testing

After implementing, test with:

```bash
# Lint
python -m pylint fusion/core/properties.py

# Type check
mypy fusion/core/properties.py

# Unit tests
python -m pytest fusion/core/tests/test_properties.py -v -k grooming
```

## Validation Checklist

- [ ] `GroomingProps` class added with type hints
- [ ] `SDNProps` extended with all 10+ new attributes
- [ ] `SDNProps.get_lightpath_id()` method added
- [ ] `SDNProps.reset_lightpath_id_counter()` method added
- [ ] `SDNProps.reset_params()` updated to reset grooming fields
- [ ] `SDNProps.stat_key_list` includes grooming keys
- [ ] `SpectrumProps` has `lightpath_id` and `lightpath_bandwidth`
- [ ] All new attributes have proper type hints
- [ ] Docstrings added for new methods
- [ ] Code passes pylint and mypy

## Next Component

After completing this component, proceed to: [Component 2: Grooming Module](02-grooming-module.md)
