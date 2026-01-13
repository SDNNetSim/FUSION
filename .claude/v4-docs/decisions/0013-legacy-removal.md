# ADR-0013: Legacy Removal Strategy

## Status

Accepted

## Context

With the V4 architecture migration progressing through Phases 1-5, the codebase now contains parallel implementations:

1. **Legacy path**: `engine_props` dicts, `SDNController`, `SDNProps`/`StatsProps`/`RoutingProps`, direct spectrum array manipulation
2. **V4 path**: `SimulationConfig`, `SDNOrchestrator`, domain objects (`Request`, `Lightpath`, `NetworkState`), pipeline architecture

Maintaining both paths creates:
- **Code duplication**: Two implementations of the same logic
- **Maintenance burden**: Bug fixes must be applied to both paths
- **Testing overhead**: Tests must cover both paths
- **Confusion risk**: Contributors unsure which path to use
- **Technical debt**: Transitional shims accumulate

Phase 6 must decide how to handle legacy code.

## Decision

### Core Decision: Remove Legacy Code Entirely

The legacy simulation path (old `SDNController`, Props classes, `engine_props` dicts) is officially deprecated and will be removed in Phase 6.

### What Constitutes "Legacy" in This Context

| Category | Legacy Components | V4 Replacement |
|----------|-------------------|----------------|
| Configuration | `engine_props` dict | `SimulationConfig` dataclass |
| State Management | `SDNProps`, `StatsProps`, `RoutingProps` | `NetworkState`, `StatsCollector`, `RouteResult` |
| Spectrum Access | `network_spectrum_dict`, direct array manipulation | `NetworkState.get_link_spectrum()`, `NetworkState.is_spectrum_available()` |
| Lightpath Access | `lightpath_status_dict`, nested dicts | `NetworkState.get_lightpath()`, `Lightpath` objects |
| Request Handling | `SDNController.handle_event()` | `SDNOrchestrator.handle_arrival()` |
| Routing | Direct `Routing` class usage | `RoutingPipeline` protocol |
| RL Integration | `mock_handle_arrival()`, duplicated feasibility | `RLSimulationAdapter`, `UnifiedSimEnv` |

### Shims Allowed to Survive

Some conversion methods may survive Phase 6 for external compatibility:

| Shim | Survival Criteria | Expiration |
|------|-------------------|------------|
| `SimulationConfig.from_engine_props()` | CLI still parses legacy INI format | Remove when CLI migrated |
| `Request.from_legacy_dict()` | External tools generate legacy format | Remove when external tools migrated |

These shims:
- Must emit `DeprecationWarning` on every call
- Must be documented as deprecated in docstrings
- Must be tracked in a "Legacy Compatibility" section of docs
- Must be reviewed for removal in the next major version

### Shims That Must Be Removed

| Shim | Reason for Removal |
|------|-------------------|
| `to_engine_props()` | No internal usage after Phase 6 |
| `to_legacy_dict()` | No internal usage after Phase 6 |
| `network_spectrum_dict` property | Performance overhead, encourages anti-patterns |
| `lightpath_status_dict` property | Performance overhead, encourages anti-patterns |
| All adapter classes | Native pipelines replace them |
| `use_orchestrator` feature flag | V4 is the only path |

### Guarantees for V4 Stability Before Removal

Before any legacy code is removed, the following must be true:

1. **Functional Parity**: `run_comparison.py` demonstrates V4 produces statistically equivalent results across all configuration combinations
2. **Test Coverage**: V4 code has >= 80% test coverage
3. **Documentation**: All V4 components have up-to-date documentation
4. **Performance**: V4 path shows no significant performance regression (< 10% slower)
5. **CI Green**: All CI checks pass with V4 path

## Alternatives Considered

### Alternative 1: Keep Legacy Path Indefinitely

**Description**: Maintain both old and new paths permanently, allowing users to choose.

**Pros**:
- No breaking changes for existing users
- Gradual migration at user's pace

**Cons**:
- Permanent maintenance burden (2x code)
- Bug fixes must be applied twice
- Testing complexity doubles
- New contributors confused about which path to use
- Technical debt never resolved

**Rejected because**: The maintenance cost is unsustainable, and the V4 architecture provides clear benefits that justify migration.

### Alternative 2: Dual-Maintain for Two Major Versions

**Description**: Keep legacy for versions X.0 and X.1, remove in X.2.

**Pros**:
- Users get extended time to migrate
- Clear deprecation timeline

**Cons**:
- Extends maintenance burden significantly
- Delays benefits of clean codebase
- Must backport fixes to legacy path

**Rejected because**: The V4 architecture is designed for functional parity; users should not need extended migration time if parity is verified.

### Alternative 3: Legacy as Optional Plugin

**Description**: Move legacy code to a separate package (e.g., `fusion-legacy`).

**Pros**:
- Core package is clean
- Users who need legacy can install plugin

**Cons**:
- Plugin maintenance still required
- Synchronization issues between core and plugin
- Adds complexity to installation/testing

**Rejected because**: The complexity of maintaining a plugin outweighs the benefit, especially since functional parity means users should migrate to V4.

### Alternative 4: Remove Without Deprecation Warnings

**Description**: Delete legacy code without prior deprecation warnings.

**Pros**:
- Fastest path to clean codebase
- No shim maintenance

**Cons**:
- Breaking change without warning
- Users may be surprised
- External tools may break

**Rejected because**: A deprecation period (during Phases 4-5) is necessary to give users visibility into upcoming changes.

## Consequences

### Positive

1. **Reduced Code Size**: Removing Props classes, adapters, and SDNController eliminates ~2000+ lines
2. **Single Source of Truth**: Only one way to manage network state (NetworkState)
3. **Simpler Testing**: No need for dual-path tests
4. **Clear Architecture**: New contributors learn only V4 patterns
5. **Performance**: No legacy property reconstruction overhead
6. **Maintainability**: Bug fixes apply to one implementation

### Negative

1. **Breaking Change**: Users with custom code using legacy APIs must migrate
2. **External Tool Updates**: Tools generating legacy format must be updated
3. **Migration Effort**: One-time effort to update all call sites

### Mitigations

- Clear migration guide in documentation
- `from_legacy_dict()` shims for external tool compatibility (with deprecation warnings)
- Examples showing before/after for common patterns (see `before_after_examples.md`)
- Deprecation warnings active during Phases 4-5 to give visibility

## Implementation Notes

### Removal Order

1. **P6.1**: Remove internal-only legacy structures (no external impact)
   - `network_spectrum_dict` property
   - `lightpath_status_dict` property
   - `mock_handle_arrival()`
   - Props classes (if no external usage)

2. **P6.2**: Remove adapters and shims (internal migration)
   - All adapter classes
   - `to_legacy_dict()` methods
   - `use_orchestrator` feature flag

3. **P6.3**: Remove deprecated modules (final cleanup)
   - `SDNController` class
   - Any remaining Props classes
   - Update documentation

### Verification Before Each Removal

```bash
# Search for usages
grep -r "<legacy_item>" fusion/ --include="*.py"

# Ensure no usages outside definition
# Then run tests
pytest fusion/tests/ -v

# Verify comparison still passes
python tests/run_comparison.py --all-configs
```

### Handling External Dependencies

If external tools depend on legacy APIs:

1. Document the deprecation in CHANGELOG
2. Provide migration examples
3. Keep `from_legacy_dict()` with deprecation warning
4. Set removal target for next major version

## Related Decisions

- [ADR-0005: Legacy Compatibility Strategy](./0005-legacy-compatibility.md) - established the transitional approach
- [ADR-0006: NetworkState Authority](./0006-networkstate-authority.md) - defines the replacement for legacy state
- [ADR-0007: Orchestrator Design](./0007-orchestrator-design.md) - defines the replacement for SDNController

## References

- [Phase 6: Legacy Removal](../migration/phase_6_legacy_removal.md)
- [Phase 6 Testing](../testing/phase_6_testing.md)
- [Final Migration Checklist](../migration/final_migration_checklist.md)
