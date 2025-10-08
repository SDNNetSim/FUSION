# Component 3: Configuration

**Files:** `fusion/configs/schema.py`, `fusion/configs/templates/`
**Estimated Time:** 30 minutes
**Dependencies:** None (can be done in parallel)

## Overview

Add grooming-related configuration options to v6.0's schema-based configuration system. This differs from v5.5's INI-based approach and requires adding fields to the config schema.

## New Configuration Fields

### Grooming Settings

```python
# In simulation_settings section
is_grooming_enabled: bool = False
can_partially_serve: bool = False
transponder_usage_per_node: bool = False
blocking_type_ci: bool = False
fragmentation_metrics: list = []
frag_calc_step: int = 1000
save_start_end_slots: bool = False
```

### SNR Settings

```python
# In snr_settings section
snr_recheck: bool = False
recheck_adjacent_cores: bool = False
recheck_crossband: bool = False
```

## Implementation Steps

### 1. Update Config Schema

Locate the appropriate schema file in `fusion/configs/schemas/` and add the new fields with proper validation.

Example structure:

```python
class SimulationSettings(BaseModel):
    """Simulation configuration settings."""

    # ... existing fields ...

    # Grooming settings
    is_grooming_enabled: bool = Field(
        default=False,
        description="Enable traffic grooming to pack requests onto existing lightpaths"
    )
    can_partially_serve: bool = Field(
        default=False,
        description="Allow partial service allocation when full bandwidth unavailable"
    )
    transponder_usage_per_node: bool = Field(
        default=False,
        description="Track transponder availability per network node"
    )
    blocking_type_ci: bool = Field(
        default=False,
        description="Enable confidence interval blocking type tracking"
    )
    fragmentation_metrics: list = Field(
        default_factory=list,
        description="List of fragmentation metrics to calculate"
    )
    frag_calc_step: int = Field(
        default=1000,
        ge=1,
        description="Step interval for fragmentation calculation"
    )
```

```python
class SNRSettings(BaseModel):
    """SNR measurement configuration."""

    # ... existing fields ...

    # Grooming-related SNR settings
    snr_recheck: bool = Field(
        default=False,
        description="Recheck SNR after spectrum allocation for grooming"
    )
    recheck_adjacent_cores: bool = Field(
        default=False,
        description="Consider adjacent core interference in SNR recheck"
    )
    recheck_crossband: bool = Field(
        default=False,
        description="Consider cross-band interference in SNR recheck"
    )
```

### 2. Update Configuration Templates

Add the new fields to the INI template used for config file generation.

File: `fusion/configs/templates/default_config.ini`

```ini
[simulation_settings]
# ... existing settings ...

# Traffic Grooming
is_grooming_enabled = False
can_partially_serve = False
transponder_usage_per_node = False
blocking_type_ci = False
fragmentation_metrics = []
frag_calc_step = 1000

[snr_settings]
# ... existing settings ...

# SNR Rechecking for Grooming
snr_recheck = False
recheck_adjacent_cores = False
recheck_crossband = False
```

### 3. Update CLI Arguments (if applicable)

If the config system supports CLI overrides, add the new arguments:

```python
# In fusion/cli/ or relevant CLI module
parser.add_argument(
    '--is-grooming-enabled',
    action='store_true',
    help='Enable traffic grooming'
)
parser.add_argument(
    '--can-partially-serve',
    action='store_true',
    help='Allow partial service allocation'
)
# ... etc for other options
```

## Configuration Field Descriptions

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `is_grooming_enabled` | bool | False | Master switch for grooming functionality |
| `can_partially_serve` | bool | False | Allow partial bandwidth allocation |
| `transponder_usage_per_node` | bool | False | Track transponder usage per node |
| `blocking_type_ci` | bool | False | Enable confidence interval tracking |
| `fragmentation_metrics` | list | [] | Metrics to calculate for fragmentation |
| `frag_calc_step` | int | 1000 | Interval for fragmentation calculations |
| `snr_recheck` | bool | False | Recheck SNR after allocation |
| `recheck_adjacent_cores` | bool | False | Include adjacent core interference |
| `recheck_crossband` | bool | False | Include cross-band interference |

## Example Configuration

```ini
[simulation_settings]
# Enable grooming with partial service support
is_grooming_enabled = True
can_partially_serve = True
transponder_usage_per_node = True
fragmentation_metrics = ["entropy", "external_fragmentation"]
frag_calc_step = 500

[snr_settings]
# Enable SNR rechecking for grooming
snr_recheck = True
recheck_adjacent_cores = True
recheck_crossband = False
```

## Testing

Test configuration loading:

```python
# Test config validation
from fusion.configs.config import Config

config = Config.from_file("test_grooming.ini")
assert config.simulation.is_grooming_enabled == True
assert config.simulation.can_partially_serve == True
assert config.snr.snr_recheck == True
```

Run validation:

```bash
# Validate schema
python -m pytest fusion/configs/tests/test_schema.py -v -k grooming

# Test config loading
python -m pytest fusion/configs/tests/test_config.py -v
```

## Validation Checklist

- [ ] Schema updated with grooming fields
- [ ] All fields have proper types and defaults
- [ ] Field descriptions added
- [ ] Validation rules added (where needed)
- [ ] Template INI file updated
- [ ] CLI arguments added (if applicable)
- [ ] Test config files created
- [ ] Schema validation tests pass
- [ ] Config loading tests pass
- [ ] Documentation updated

## Next Component

After completing this component, proceed to: [Component 4: SDN Controller](04-sdn-controller.md)
