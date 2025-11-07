# Traffic Grooming Port: v5.5 → v6.0 Overview

## Summary

This document outlines the plan to port the traffic grooming feature from FUSION v5.5 (main branch) to v6.0 (release/6.0.0 branch). Traffic grooming allows the simulator to efficiently pack multiple requests onto existing lightpaths, improving resource utilization.

## Source

- **Feature Branch:** `feature/grooming-new`
- **Target Branch:** `release/6.0.0`
- **Pull Request:** #105 (not yet merged to main)

## Key Feature Capabilities

Traffic grooming adds the following capabilities:

1. **End-to-End Grooming** - Assign new requests to existing lightpaths with available bandwidth
2. **Partial Grooming** - Groom part of a request, establish new lightpath for remainder
3. **Lightpath Tracking** - Track established lightpaths and their bandwidth utilization
4. **Transponder Management** - Track transponder availability per node
5. **SNR Rechecking** - Validate signal quality after spectrum allocation
6. **Bandwidth Utilization Metrics** - Time-weighted bandwidth usage statistics

## Architecture Mapping

| v5.5 (old)                      | v6.0 (new)                           |
|---------------------------------|--------------------------------------|
| `src/grooming.py`               | `fusion/core/grooming.py`            |
| `arg_scripts/grooming_args.py`  | `fusion/core/properties.py`          |
| `arg_scripts/sdn_args.py`       | `fusion/core/properties.py`          |
| `src/sdn_controller.py`         | `fusion/core/sdn_controller.py`      |
| `src/spectrum_assignment.py`    | `fusion/core/spectrum_assignment.py` |
| `src/snr_measurements.py`       | `fusion/core/snr_measurements.py`    |
| `src/engine.py`                 | `fusion/sim/network_simulator.py`    |
| `helper_scripts/sim_helpers.py` | `fusion/utils/network.py`            |
| `arg_scripts/config_args.py`    | `fusion/configs/schema.py`           |

## Component Breakdown

1. **Properties** (30 min) - Add GroomingProps, extend SDNProps and SpectrumProps
2. **Grooming Module** (45 min) - Port core grooming logic
3. **Configuration** (30 min) - Add config schema for grooming settings
4. **SDN Controller** (2-3 hrs) - Integrate grooming into request handling
5. **Spectrum Assignment** (1 hr) - Add lightpath ID tracking
6. **SNR Measurements** (1 hr) - Add SNR rechecking
7. **Helper Functions** (15 min) - Port bandwidth utilization helper
8. **Simulation** (30 min) - Initialize grooming structures
9. **Statistics** (45 min) - Add grooming metrics
10. **Tests** (2 hrs) - Port and adapt test suite

**Total estimated time:** 10-12 hours

## Execution Order

Components must be implemented in dependency order:

```
1. Properties (foundation)
   ↓
2. Grooming Module (core logic)
   ↓
3. Helper Functions (utilities)
   ↓
4. Spectrum Assignment (resource tracking)
   ↓
5. SNR Measurements (quality checks)
   ↓
6. SDN Controller (integration)
   ↓
7. Configuration (enable/disable)
   ↓
8. Simulation (initialization)
   ↓
9. Statistics (reporting)
   ↓
10. Tests (validation)
```

## Critical Dependencies

1. **Properties first** - All components depend on updated property classes
2. **Grooming before SDN** - SDN controller uses Grooming class
3. **Helpers before usage** - Helper functions must exist before being called
4. **Tests incremental** - Test each component as it's completed

## Major Risks

1. **SDN Controller Complexity** - v6.0 has significantly different structure from v5.5
2. **Configuration System Changes** - v6.0 uses schema-based config vs v5.5 INI files
3. **Breaking Changes** - Core component modifications may affect existing functionality
4. **Test Adaptation** - v5.5 tests will need significant updates for v6.0

## Validation Strategy

After each component:

1. **Lint:** `python -m pylint fusion/core/[component].py`
2. **Type Check:** `python -m mypy fusion/core/[component].py`
3. **Unit Test:** `python -m pytest tests/test_[component].py -v`
4. **Integration Test:** Run minimal simulation with grooming enabled

## New Configuration Options

```ini
[simulation_settings]
is_grooming_enabled = False
can_partially_serve = False
transponder_usage_per_node = False
blocking_type_ci = False
fragmentation_metrics = []
frag_calc_step = 1000

[snr_settings]
snr_recheck = False
recheck_adjacent_cores = False
recheck_crossband = False
```

## Files Modified Summary

**New Files:**
- `fusion/core/grooming.py` (new)
- `grooming_port/*.md` (documentation)
- `tests/test_grooming.py` (new)

**Modified Files:**
- `fusion/core/properties.py` (extend SDNProps, SpectrumProps, add GroomingProps)
- `fusion/core/sdn_controller.py` (grooming integration)
- `fusion/core/spectrum_assignment.py` (lightpath tracking)
- `fusion/core/snr_measurements.py` (SNR rechecking)
- `fusion/sim/network_simulator.py` (initialization)
- `fusion/configs/schema.py` (new config options)
- `fusion/utils/network.py` (new helper function)
- `fusion/reporting/statistics.py` (grooming metrics)
- Multiple test files

## Next Steps

Start with Component 1: [Properties](01-properties.md)
