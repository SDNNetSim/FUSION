# Phase 6: Legacy Removal and Final Cleanup

## Overview

Phase 6 is the final migration phase. It removes legacy data structures, deprecated modules, and transitional shims that were maintained for backward compatibility during Phases 2-5. Upon completion, V4 becomes the sole supported simulation path.

## Prerequisites

Before starting Phase 6:

- [ ] Phase 5 complete (ML control + protection verified)
- [ ] All heuristic and ML policies functional
- [ ] Protection pipeline tested
- [ ] `run_comparison.py` passes with V4 path for all configurations
- [ ] No external code depends on legacy APIs (verified by deprecation warning monitoring)
- [ ] All Phase 1-5 documentation complete and accurate

## Objectives

1. **Remove unused legacy data structures** (engine_props dicts, network_spectrum_dict, lightpath_status_dict)
2. **Replace remaining legacy call sites** with V4 domain model, pipelines, and orchestrator
3. **Delete deprecated modules** and feature flags
4. **Update documentation and tests** to treat V4 as the only path

---

## Micro-Phases

### P6.1: Remove Unused Legacy Data Structures

**Goal**: Delete legacy data structures that are provably dead (no references outside their definitions).

**Preconditions**:
- [ ] Phase 5 complete
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] Grep verification confirms no external usages

**Targeted Legacy Structures**:

| Structure | Location | Replacement | Verification |
|-----------|----------|-------------|--------------|
| `network_spectrum_dict` property | `fusion/domain/network_state.py` | `NetworkState.get_link_spectrum()`, `NetworkState.is_spectrum_available()` | `grep -r "network_spectrum_dict" fusion/ --include="*.py"` returns only the property definition |
| `lightpath_status_dict` property | `fusion/domain/network_state.py` | `NetworkState.get_lightpath()`, `NetworkState.get_lightpaths_between()` | `grep -r "lightpath_status_dict" fusion/ --include="*.py"` returns only the property definition |
| `SDNProps` class | `fusion/core/properties.py` | `Request`, `AllocationResult`, `NetworkState` | `grep -r "SDNProps" fusion/ --include="*.py"` returns no instantiations |
| `StatsProps` class | `fusion/core/properties.py` | `StatsCollector` | `grep -r "StatsProps" fusion/ --include="*.py"` returns no instantiations |
| `RoutingProps` class | `fusion/core/properties.py` | `RouteResult` | `grep -r "RoutingProps" fusion/ --include="*.py"` returns no instantiations |
| Old RL `mock_handle_arrival()` | `fusion/modules/rl/utils/general_utils.py` | `RLSimulationAdapter.apply_action()` | `grep -r "mock_handle_arrival" fusion/ --include="*.py"` returns no calls |
| `_legacy_lightpath_status_dict` internal | `fusion/domain/network_state.py` | Native `_lightpaths` dict | Internal reference only |

**Actions**:

1. Run grep verification for each structure
2. Remove properties/classes with zero external references
3. Update any remaining internal references to use V4 APIs
4. Run full test suite after each removal

**Exit Criteria P6.1**:
- [ ] All listed legacy properties removed from NetworkState
- [ ] All listed Props classes removed or marked deprecated
- [ ] `mock_handle_arrival()` deleted
- [ ] Unit tests pass
- [ ] Integration tests pass

---

### P6.2: Replace Remaining Legacy Call Sites

**Goal**: Convert any remaining legacy API usages to V4 domain model/pipelines/orchestrator. Remove transitional shims and adapters.

**Preconditions**:
- [ ] P6.1 complete
- [ ] All P6.1 tests pass

**Targeted Shims and Adapters**:

| Shim/Adapter | Location | Purpose | Replacement |
|--------------|----------|---------|-------------|
| `SimulationConfig.to_engine_props()` | `fusion/domain/config.py` | CLI backward compatibility | Direct `SimulationConfig` usage or remove if CLI migrated |
| `Request.to_legacy_dict()` | `fusion/domain/request.py` | Conversion for old SDN | Remove (no old SDN) |
| `Request.from_legacy_dict()` | `fusion/domain/request.py` | Parsing old format | Keep if external input uses legacy format, else remove |
| `Lightpath.to_legacy_dict()` | `fusion/domain/lightpath.py` | Conversion for old stats | Remove (stats use StatsCollector) |
| `RoutingAdapter` | `fusion/core/adapters/routing_adapter.py` | Wrap old routing for pipeline | Remove (use native KSPRoutingPipeline) |
| `SpectrumAdapter` | `fusion/core/adapters/spectrum_adapter.py` | Wrap old spectrum for pipeline | Remove (use native FirstFitSpectrumPipeline) |
| `GroomingAdapter` | `fusion/core/adapters/grooming_adapter.py` | Wrap old grooming for pipeline | Remove (use native GroomingPipeline) |
| `SNRAdapter` | `fusion/core/adapters/snr_adapter.py` | Wrap old SNR for pipeline | Remove (use native SNRPipeline) |
| `SlicingAdapter` | `fusion/core/adapters/slicing_adapter.py` | Wrap old slicing for pipeline | Remove (use native SlicingPipeline) |
| `use_orchestrator` feature flag | `fusion/core/simulation.py` | Toggle old vs new path | Remove (always use orchestrator) |

**Actions**:

1. Identify remaining usages of each shim via grep
2. Convert call sites to use native V4 implementations
3. Delete adapter files from `fusion/core/adapters/`
4. Remove feature flag logic from SimulationEngine
5. Run test suite after each conversion

**Exit Criteria P6.2**:
- [ ] All adapter classes in `fusion/core/adapters/` deleted
- [ ] `to_legacy_dict()` methods removed from domain objects
- [ ] `use_orchestrator` feature flag removed
- [ ] SimulationEngine always uses SDNOrchestrator
- [ ] Integration tests pass
- [ ] `run_comparison.py` passes (comparing V4 against stored baseline)

---

### P6.3: Delete Deprecated Modules and Update Documentation

**Goal**: Remove deprecated modules entirely, update all documentation and tests to reflect V4 as the only supported path.

**Preconditions**:
- [ ] P6.2 complete
- [ ] All P6.2 tests pass
- [ ] `run_comparison.py` passes

**Targeted Modules for Deletion**:

| Module/File | Reason | Dependency Check |
|-------------|--------|------------------|
| `fusion/core/sdn_controller.py` | Replaced by SDNOrchestrator | Grep for imports |
| `fusion/core/properties.py` | Replaced by domain objects | Grep for imports (may keep for legacy CLI) |
| `fusion/core/adapters/` directory | No longer needed | Should be empty after P6.2 |
| `fusion/legacy/` directory | Transitional adapters | Should be empty or removable |
| Old RL environment files | Replaced by UnifiedSimEnv | Grep for imports |

**Targeted Modules for Deprecation (if keeping for external compatibility)**:

| Module/File | Deprecation Strategy |
|-------------|---------------------|
| `fusion/core/sdn_controller.py` | Move to `fusion/legacy/sdn_controller.py`, add deprecation warning |
| `fusion/core/properties.py` | Add deprecation warnings to all classes |

**Documentation Updates**:

| Document | Required Changes |
|----------|------------------|
| `docs/conf.py` | Remove references to deprecated modules |
| `docs/architecture/` | Update to reference only V4 structures |
| `ARCHITECTURE.md` | Update top-level overview |
| `README.md` | Update quickstart examples |
| `CONTRIBUTING.md` | Update development workflow |
| Sphinx API docs | Regenerate without deprecated modules |

**Test Updates**:

| Test Category | Required Changes |
|---------------|------------------|
| Unit tests for Props classes | Delete or move to legacy test directory |
| Unit tests for old SDNController | Delete or move to legacy test directory |
| Integration tests | Update to use only V4 path |
| `run_comparison.py` | Convert to regression test against stored baseline (no dual-path) |

**Actions**:

1. Move deprecated modules to `fusion/legacy/` (if keeping) or delete entirely
2. Update all import statements across codebase
3. Regenerate Sphinx documentation
4. Update README and CONTRIBUTING
5. Update or delete obsolete tests
6. Convert `run_comparison.py` to baseline regression mode
7. Final full test suite run

**Exit Criteria P6.3**:
- [ ] Deprecated modules deleted or moved to `fusion/legacy/`
- [ ] No imports of deleted modules in main codebase
- [ ] All documentation updated
- [ ] Sphinx docs build without warnings for deleted modules
- [ ] All tests pass
- [ ] CI pipeline passes

---

## Rollback Strategy

If Phase 6 changes cause critical issues that cannot be resolved:

### P6.1 Rollback

1. Revert commits that removed legacy properties
2. Legacy properties restored; no functional impact
3. Re-run tests to verify restoration

### P6.2 Rollback

1. Revert commits that removed adapters
2. Restore `use_orchestrator` feature flag
3. Set flag to `False` to use old path
4. Re-run tests to verify restoration

### P6.3 Rollback

1. Revert commits that deleted deprecated modules
2. Restore import statements
3. Re-run tests to verify restoration

### Full Phase 6 Rollback

If all micro-phases must be reverted:

```bash
# Identify commit before Phase 6 started
git log --oneline | grep "Phase 5 complete"

# Create rollback branch
git checkout -b rollback-phase-6 <commit-hash>

# Cherry-pick any non-Phase-6 fixes if needed
git cherry-pick <fix-commit-hashes>

# Verify tests pass
pytest fusion/tests/ -v
python tests/run_comparison.py --compare-paths
```

---

## Verification Commands

### Pre-Phase Verification

```bash
# Verify Phase 5 is complete
pytest fusion/tests/policies/ -v
pytest fusion/tests/pipelines/test_protection_pipeline.py -v
python tests/run_comparison.py --compare-paths --all-configs

# Check for deprecation warnings in logs (should be minimal/none from internal code)
python -W default::DeprecationWarning -c "from fusion.core import simulation; ..."
```

### P6.1 Verification

```bash
# Grep verification for each legacy structure
grep -r "network_spectrum_dict" fusion/ --include="*.py" | grep -v "def network_spectrum_dict"
grep -r "lightpath_status_dict" fusion/ --include="*.py" | grep -v "def lightpath_status_dict"
grep -r "SDNProps" fusion/ --include="*.py"
grep -r "mock_handle_arrival" fusion/ --include="*.py"

# Run tests after each removal
pytest fusion/tests/ -v --tb=short
```

### P6.2 Verification

```bash
# Verify no adapters remain
ls fusion/core/adapters/  # Should be empty or not exist
ls fusion/legacy/adapters/  # Should be empty or not exist

# Verify feature flag removed
grep -r "use_orchestrator" fusion/ --include="*.py"  # Should return nothing

# Run tests
pytest fusion/tests/ -v
python tests/run_comparison.py --seed 42
```

### P6.3 Verification

```bash
# Verify deprecated modules removed/moved
ls fusion/core/sdn_controller.py  # Should not exist or be in legacy/
ls fusion/core/properties.py  # Should not exist or be in legacy/

# Verify documentation builds
cd docs && make html

# Full test suite
pytest fusion/tests/ -v --cov=fusion --cov-report=term-missing

# Regression test
python tests/run_comparison.py --baseline-mode --all-configs
```

---

## Files Summary

### Files to Delete

```
fusion/domain/network_state.py    # Remove legacy properties only
fusion/core/properties.py         # Delete or move to legacy/
fusion/core/sdn_controller.py     # Delete or move to legacy/
fusion/core/adapters/             # Delete entire directory
fusion/legacy/adapters/           # Delete entire directory
fusion/modules/rl/utils/general_utils.py  # Remove mock_handle_arrival
```

### Files to Modify

```
fusion/domain/config.py           # Remove to_engine_props() if not needed
fusion/domain/request.py          # Remove to_legacy_dict(), from_legacy_dict() if not needed
fusion/domain/lightpath.py        # Remove to_legacy_dict() if not needed
fusion/core/simulation.py         # Remove feature flag, always use orchestrator
fusion/core/pipeline_factory.py   # Remove adapter creation logic
```

### Files to Update (Documentation)

```
docs/conf.py
docs/architecture/*.md
ARCHITECTURE.md
README.md
CONTRIBUTING.md
.claude/v4-docs/migration/overview.md
```

---

## Timeline and Dependencies

```
Phase 5 Complete
       |
       v
    P6.1: Remove Unused Legacy Structures (2-3 PRs)
       |
       v
    P6.2: Replace Legacy Call Sites (3-5 PRs)
       |
       v
    P6.3: Delete Deprecated Modules (2-3 PRs)
       |
       v
Phase 6 Complete - V4 is the sole supported path
```

---

## Related Documents

- [ADR-0005: Legacy Compatibility Strategy](../decisions/0005-legacy-compatibility.md)
- [ADR-0013: Legacy Removal](../decisions/0013-legacy-removal.md)
- [Phase 6 Testing Strategy](../testing/phase_6_testing.md)
- [Final Migration Checklist](./final_migration_checklist.md)
- [Phase 5: ML Control + Protection](./phase_5_ml_protection.md)
