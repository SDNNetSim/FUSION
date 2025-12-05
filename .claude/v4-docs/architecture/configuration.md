# Configuration Architecture

This document describes the `SimulationConfig` design and feature flag system.

## Overview

`SimulationConfig` replaces the mutable `engine_props` dictionary with an immutable, typed configuration object. It is created once at simulation start and never modified.

## SimulationConfig Structure

### Network Configuration

```python
# Topology
network_name: str       # e.g., "USbackbone60"
cores_per_link: int     # Spatial cores per fiber (1-7 typical)

# Spectrum
band_list: tuple[str, ...] # ("c",) or ("c", "l") or ("c", "l", "s")
band_slots: dict[str, int] # {"c": 320, "l": 320, "s": 320}
guard_slots: int           # Guard band between allocations (typically 1)
```

### Traffic Configuration

```python
num_requests: int    # Total requests to generate
erlang: float        # Traffic load in Erlangs
holding_time: float  # Mean holding time
```

### Routing Configuration

```python
route_method: str       # "k_shortest_path", "1plus1_protection", etc.
k_paths: int            # Number of candidate paths
allocation_method: str  # "first_fit", "best_fit", etc.
```

### Feature Flags

```python
# Grooming
grooming_enabled: bool        # Enable lightpath reuse
can_partially_serve: bool     # Allow partial bandwidth allocation

# Slicing
slicing_enabled: bool         # Enable bandwidth slicing
max_slices: int              # Maximum slices per request (2-8)

# SNR
snr_enabled: bool            # Enable SNR validation
snr_type: str | None         # "snr_e2e", "snr_segment", None
snr_recheck: bool            # Recheck SNR after allocation
```

### Modulation Configuration

```python
modulation_formats: dict[str, Any]    # Format specifications
mod_per_bw: dict[str, list[str]]     # Bandwidth -> valid modulations
snr_thresholds: dict[str, float]      # Modulation -> required SNR
```

---

## Feature Flag Interactions

### Grooming + Slicing

When both enabled, grooming is attempted first:

```
1. Try grooming existing lightpaths
   - If fully_groomed: done
   - If partially_groomed: continue with remaining bandwidth

2. Try standard allocation

3. If standard fails and slicing enabled:
   - Try slicing the (remaining) bandwidth
```

### SNR with Everything

SNR validation applies regardless of other features:

```
Standard allocation:
  create_lightpath -> snr.validate -> accept/reject

Grooming:
  No SNR check (reusing existing validated lightpath)

Slicing:
  Each slice: create_lightpath -> snr.validate -> accept/reject

Protection:
  Both paths: snr.validate on primary AND backup
```

### Feature Flag Matrix

| Feature | grooming_enabled | slicing_enabled | snr_enabled | Effect |
|---------|------------------|-----------------|-------------|--------|
| Basic KSP | False | False | False | K-shortest path, first-fit |
| + Grooming | True | False | False | Reuse lightpaths when possible |
| + Slicing | False | True | False | Split large requests |
| + SNR | False | False | True | Validate signal quality |
| Full | True | True | True | All features active |

---

## Legacy Conversion

### from_engine_props()

```python
@classmethod
def from_engine_props(cls, engine_props: dict) -> "SimulationConfig":
    """Create from legacy engine_props dictionary."""
    return cls(
        # Network
        network_name=engine_props.get("network", ""),
        cores_per_link=engine_props.get("cores_per_link", 1),
        band_list=tuple(engine_props.get("band_list", ["c"])),
        band_slots={
            "c": engine_props.get("c_band", 320),
            "l": engine_props.get("l_band", 0),
            "s": engine_props.get("s_band", 0),
        },
        guard_slots=engine_props.get("guard_slots", 1),

        # Traffic
        num_requests=engine_props.get("num_requests", 1000),
        erlang=engine_props.get("erlang", 100.0),
        holding_time=engine_props.get("holding_time", 1.0),

        # Routing
        route_method=engine_props.get("route_method", "k_shortest_path"),
        k_paths=engine_props.get("k_paths", 3),
        allocation_method=engine_props.get("allocation_method", "first_fit"),

        # Features
        grooming_enabled=engine_props.get("is_grooming_enabled", False),
        slicing_enabled=engine_props.get("max_segments", 1) > 1,
        max_slices=engine_props.get("max_segments", 1),
        snr_enabled=engine_props.get("snr_type") not in (None, "None", ""),
        snr_type=engine_props.get("snr_type"),
        snr_recheck=engine_props.get("snr_recheck", False),
        can_partially_serve=engine_props.get("can_partially_serve", False),

        # Modulation
        modulation_formats=engine_props.get("modulation_formats_dict", {}),
        mod_per_bw=engine_props.get("mod_per_bw", {}),
        snr_thresholds=engine_props.get("req_snr", {}),
    )
```

### to_engine_props()

```python
def to_engine_props(self) -> dict:
    """Convert back to legacy format for compatibility."""
    return {
        "network": self.network_name,
        "cores_per_link": self.cores_per_link,
        "band_list": list(self.band_list),
        "c_band": self.band_slots.get("c", 0),
        "l_band": self.band_slots.get("l", 0),
        "s_band": self.band_slots.get("s", 0),
        "guard_slots": self.guard_slots,
        "num_requests": self.num_requests,
        "erlang": self.erlang,
        "holding_time": self.holding_time,
        "route_method": self.route_method,
        "k_paths": self.k_paths,
        "allocation_method": self.allocation_method,
        "is_grooming_enabled": self.grooming_enabled,
        "max_segments": self.max_slices,
        "snr_type": self.snr_type,
        "snr_recheck": self.snr_recheck,
        "can_partially_serve": self.can_partially_serve,
        "modulation_formats_dict": self.modulation_formats,
        "mod_per_bw": self.mod_per_bw,
        "req_snr": self.snr_thresholds,
    }
```

---

## Key Mapping

| Legacy Key | V4 Field | Notes |
|------------|----------|-------|
| `network` | `network_name` | Renamed for clarity |
| `cores_per_link` | `cores_per_link` | Same |
| `band_list` | `band_list` | List -> Tuple |
| `c_band`, `l_band`, `s_band` | `band_slots` | Combined into dict |
| `is_grooming_enabled` | `grooming_enabled` | Removed `is_` prefix |
| `max_segments` | `max_slices`, `slicing_enabled` | Split into count and flag |
| `snr_type` | `snr_type`, `snr_enabled` | Added explicit flag |
| `modulation_formats_dict` | `modulation_formats` | Removed `_dict` suffix |

---

## Validation

### At Creation Time

```python
def __post_init__(self):
    """Validate configuration."""
    if self.cores_per_link < 1:
        raise ValueError("cores_per_link must be >= 1")

    if not self.band_list:
        raise ValueError("band_list cannot be empty")

    for band in self.band_list:
        if band not in ("c", "l", "s"):
            raise ValueError(f"Invalid band: {band}")

    if self.k_paths < 1:
        raise ValueError("k_paths must be >= 1")

    if self.slicing_enabled and self.max_slices < 2:
        raise ValueError("max_slices must be >= 2 when slicing enabled")
```

### Validation Function

```python
def validate_config(config: SimulationConfig) -> list[str]:
    """
    Validate configuration, return list of errors.

    This is for validation after creation (e.g., from legacy dict
    that might have inconsistent values).
    """
    errors = []

    # Network checks
    if config.cores_per_link < 1:
        errors.append("cores_per_link must be >= 1")

    for band in config.band_list:
        if config.band_slots.get(band, 0) <= 0:
            errors.append(f"band_slots[{band}] must be > 0")

    # Feature checks
    if config.snr_enabled and not config.snr_type:
        errors.append("snr_type required when snr_enabled")

    if config.slicing_enabled and config.max_slices < 2:
        errors.append("max_slices must be >= 2 when slicing enabled")

    return errors
```

---

## Usage Patterns

### Creating Configuration

```python
# From legacy dict
config = SimulationConfig.from_engine_props(engine_props)

# Direct construction (for tests)
config = SimulationConfig(
    network_name="test",
    cores_per_link=7,
    band_list=("c", "l"),
    band_slots={"c": 320, "l": 320, "s": 0},
    guard_slots=1,
    num_requests=1000,
    erlang=100.0,
    holding_time=1.0,
    route_method="k_shortest_path",
    k_paths=3,
    allocation_method="first_fit",
    grooming_enabled=True,
    slicing_enabled=False,
    max_slices=1,
    snr_enabled=True,
    snr_type="snr_e2e",
    snr_recheck=False,
    can_partially_serve=False,
    modulation_formats={...},
    mod_per_bw={...},
    snr_thresholds={...},
)
```

### Checking Features

```python
# In orchestrator
if self.config.grooming_enabled and self.grooming:
    groom_result = self.grooming.try_groom(request, network_state)

if self.config.slicing_enabled and self.slicing:
    slice_result = self.slicing.try_slice(...)

if self.config.snr_enabled and self.snr:
    if not self.snr.validate(lightpath, network_state):
        network_state.release_lightpath(lightpath.lightpath_id)
```

### Accessing Band Info

```python
# Get slots for a band
c_slots = config.band_slots.get("c", 0)

# Iterate over bands
for band in config.band_list:
    slots = config.band_slots[band]
    print(f"{band}-band: {slots} slots")
```

---

## Immutability

Configuration is frozen - attempts to modify raise an error:

```python
config = SimulationConfig.from_engine_props(engine_props)

# This raises FrozenInstanceError
config.k_paths = 5

# To get modified config, create new instance
from dataclasses import replace
new_config = replace(config, k_paths=5)
```

Note: `dataclasses.replace()` works with frozen dataclasses and creates a new instance with specified fields changed.
