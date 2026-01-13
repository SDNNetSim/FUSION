# ADR-0005: Legacy Compatibility Strategy

## Status

Accepted

## Context

The V4 migration must coexist with legacy code during the transition period. Key questions:
1. How do new domain objects interoperate with legacy dicts?
2. How long do we maintain compatibility?
3. When and how do we remove legacy support?

## Decision

### Phase 1: Additive Only

Phase 1 creates new domain objects without modifying existing code:

```python
# New code - additive
from fusion.domain import SimulationConfig, Request, Lightpath

# Existing code - unchanged
engine_props = load_config()  # Still works
sdn_controller.handle_event(request_dict, 'arrival')  # Still works
```

### Conversion Methods

All domain objects provide bidirectional conversion:

```python
# Legacy -> V4
config = SimulationConfig.from_engine_props(engine_props)
request = Request.from_legacy_dict(time_key, request_dict)
lightpath = Lightpath.from_legacy_dict(lp_id, lp_dict)

# V4 -> Legacy
engine_props = config.to_engine_props()
request_dict = request.to_legacy_dict()
lp_dict = lightpath.to_legacy_dict()
```

### Phases 2-3: Dual Path

NetworkState provides legacy compatibility properties:

```python
class NetworkState:
    @property
    def network_spectrum_dict(self) -> dict:
        """DEPRECATED: Legacy format for backward compatibility."""
        return self._build_legacy_spectrum_dict()

    @property
    def lightpath_status_dict(self) -> dict:
        """DEPRECATED: Legacy format for backward compatibility."""
        return self._build_legacy_lightpath_dict()
```

### Phase 4: Deprecation Warnings

Add warnings to legacy access patterns:

```python
@property
def network_spectrum_dict(self) -> dict:
    warnings.warn(
        "network_spectrum_dict is deprecated. Use NetworkState methods.",
        DeprecationWarning,
        stacklevel=2,
    )
    return self._build_legacy_spectrum_dict()
```

### Phase 5-6: Removal

After verifying no code uses legacy patterns:

```bash
# Check for usages
grep -r "network_spectrum_dict" fusion/ --include="*.py"
grep -r "lightpath_status_dict" fusion/ --include="*.py"

# If clean, remove the properties
```

## Removal Criteria

Before removing legacy support:

- [ ] All code paths use new APIs
- [ ] `run_comparison.py` passes with new architecture
- [ ] Grep for legacy patterns returns only the compatibility code itself
- [ ] Deprecation warnings have been active for one release

## Consequences

### Positive

1. **Safe migration**: Legacy code continues to work
2. **Incremental adoption**: Can migrate one module at a time
3. **Verification**: Can compare old and new paths
4. **Reversibility**: Can revert if issues arise

### Negative

1. **Code duplication**: Conversion methods add code
2. **Maintenance burden**: Must keep conversions in sync
3. **Performance**: Conversion has overhead

### Mitigations

- Conversion is typically O(n) where n is small (number of fields)
- Only used at boundaries, not in hot paths
- Removed in Phase 6, not permanent

## Implementation Notes

### Testing Roundtrips

Every conversion must be tested:

```python
def test_config_roundtrip():
    """from_engine_props -> to_engine_props preserves data."""
    original = {...}  # Sample engine_props
    config = SimulationConfig.from_engine_props(original)
    result = config.to_engine_props()

    for key in original:
        assert result[key] == original[key]
```

### Handling Missing Keys

Use defaults for missing keys:

```python
@classmethod
def from_engine_props(cls, engine_props: dict) -> "SimulationConfig":
    return cls(
        k_paths=engine_props.get("k_paths", 3),  # Default if missing
        # ...
    )
```

### Type Coercion

Handle type differences:

```python
# Legacy might have string "None"
snr_type = engine_props.get("snr_type")
snr_enabled = snr_type not in (None, "None", "")  # Handle all cases
```

## Related Decisions

- ADR-0001: Frozen Dataclasses
- Phase 6 documentation will detail removal process
