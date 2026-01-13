# Phase 1: Core Domain Model

**Version**: v5-final-docs
**Scope**: Phase 1 only (P1.1 through P1.5)
**Constraint**: Additive changes only; `run_comparison.py` must continue to pass

---

## Overview

Phase 1 establishes the typed domain model for FUSION v5. All changes are **additive** - no modifications to existing function signatures or behavior.

### Goals

1. **Type Safety**: Replace untyped dictionaries with frozen/typed dataclasses
2. **Immutability**: Configuration and result objects are frozen after creation
3. **Legacy Compatibility**: Bidirectional adapters (`from_legacy_dict`, `to_legacy_dict`)
4. **Testability**: 90-95% test coverage for all new domain objects

---

## Sub-Phases

| Sub-Phase | Description | Files Created |
|-----------|-------------|---------------|
| [P1.0](P1.0_gap_analysis/P1.0.index.md) | Gap Analysis | (documentation only) |
| [P1.1](P1.1_domain_scaffolding/P1.1.index.md) | Domain Scaffolding | `fusion/domain/__init__.py`, `config.py` |
| [P1.2](P1.2_request_wrapper/P1.2.index.md) | Request Wrapper | `fusion/domain/request.py`, enums |
| [P1.3](P1.3_lightpath_wrapper/P1.3.index.md) | Lightpath Wrapper | `fusion/domain/lightpath.py` |
| [P1.4](P1.4_result_objects/P1.4.index.md) | Result Objects | `fusion/domain/results.py` |
| [P1.5](P1.5_stats_collector/P1.5.index.md) | StatsCollector Skeleton | `fusion/stats/collector.py` |

### P1.0 Gap Analysis Contents

The P1.0 gap analysis provides comprehensive documentation ensuring Phase 1 completeness:

| Document | Purpose |
|----------|---------|
| [P1.0.index.md](P1.0_gap_analysis/P1.0.index.md) | Gap summary and verification checklist |
| [P1.0.a_state_centralization_map.md](P1.0_gap_analysis/P1.0.a_state_centralization_map.md) | Complete field-by-field mapping from legacy Props classes |
| [P1.0.b_sdnprops_distribution.md](P1.0_gap_analysis/P1.0.b_sdnprops_distribution.md) | Detailed SDNProps field distribution plan |
| [P1.0.c_complete_enum_catalog.md](P1.0_gap_analysis/P1.0.c_complete_enum_catalog.md) | All domain enums with complete values |

---

## Execution Order

Execute sub-phases in order:

1. **P1.0 (Read First)**: Gap analysis - read for context before implementation
2. **P1.1**: Domain scaffolding and SimulationConfig
3. **P1.2**: Request wrapper with enums
4. **P1.3**: Lightpath wrapper
5. **P1.4**: Result objects
6. **P1.5**: StatsCollector skeleton

Within each sub-phase, execute micro-tasks in alphabetical order:
- `P1.X.a_*` before `P1.X.b_*` before `P1.X.c_*`, etc.

---

## How to Use These Docs

Each micro-task file is self-contained. To execute a task:

1. Load the micro-task markdown file
2. Load all files listed in the "Context to load" section
3. Follow the instructions to produce the specified outputs
4. Run verification commands listed in the task

---

## Key Constraints

### Hard Rules (from `phase_1_core_model.md`)

1. **`run_comparison.py` must pass unchanged**
2. **All new code is additive only** - no modifications to existing signatures
3. **Existing tests must continue to pass**
4. **All new files only** - never modify existing source files

### Design Requirements

1. **SimulationConfig** must be a frozen dataclass
2. **Result objects** (RouteResult, SpectrumResult, etc.) must be frozen
3. **Request and Lightpath** are mutable dataclasses (state changes during lifecycle)
4. **Full type annotations** on all new code
5. **Enums for status values** - no magic strings
6. **Roundtrip conversion** - `from_legacy_dict()` -> `to_legacy_dict()` must preserve data

### Quality Requirements

| Module | Target Coverage | Test File |
|--------|-----------------|-----------|
| `fusion/domain/config.py` | 95% | `fusion/tests/domain/test_config.py` |
| `fusion/domain/request.py` | 95% | `fusion/tests/domain/test_request.py` |
| `fusion/domain/lightpath.py` | 95% | `fusion/tests/domain/test_lightpath.py` |
| `fusion/domain/results.py` | 90% | `fusion/tests/domain/test_results.py` |
| `fusion/stats/collector.py` | 90% | `fusion/tests/stats/test_collector.py` |

---

## New File Locations

```
fusion/
├── domain/
│   ├── __init__.py         # Public API exports
│   ├── config.py           # SimulationConfig
│   ├── request.py          # Request, RequestStatus, BlockReason
│   ├── lightpath.py        # Lightpath
│   └── results.py          # All result dataclasses
├── stats/
│   ├── __init__.py
│   └── collector.py        # StatsCollector
└── tests/
    ├── domain/
    │   ├── __init__.py
    │   ├── test_config.py
    │   ├── test_request.py
    │   ├── test_lightpath.py
    │   └── test_results.py
    └── stats/
        ├── __init__.py
        └── test_collector.py
```

---

## Reference Documents

### V5 Gap Analysis (read first)
- `P1.0_gap_analysis/P1.0.index.md` - Gap summary and verification checklist
- `P1.0_gap_analysis/P1.0.a_state_centralization_map.md` - Complete Props -> domain mapping
- `P1.0_gap_analysis/P1.0.b_sdnprops_distribution.md` - SDNProps field distribution
- `P1.0_gap_analysis/P1.0.c_complete_enum_catalog.md` - All domain enums

### V4 Architecture Docs
- `.claude/v4-docs/migration/phase_1_core_model.md` - Authoritative Phase 1 specification
- `.claude/v4-docs/architecture/domain_model.md` - Domain object design
- `.claude/v4-docs/architecture/result_objects.md` - Result object specifications
- `.claude/v4-docs/architecture/stats_and_metrics.md` - StatsCollector design

### V2/V3 Plans
- `.claude/ARCHITECTURE_REFACTOR_PLAN_V2.md` - Original inventory and analysis
- `.claude/ARCHITECTURE_REFACTOR_PLAN_V3.md` - Detailed walkthroughs and examples
- `.claude/v4-migration-plan.md` - Migration overview
