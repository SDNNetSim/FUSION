# Phase 2: State Management Migration

## Overview

Phase 2 establishes `NetworkState` as the **single source of truth** for all mutable network state during simulation, replacing the scattered dictionary-based state management (`engine_props`, `sdn_props`, `stats_props`).

**Status**: Documentation complete, ready for implementation
**Prerequisite**: Phase 1 (Core Domain Model) complete
**Scope**: Additive only - no breaking changes to existing code

## Goals

1. **Centralized State**: All network state flows through `NetworkState`
2. **Type Safety**: Full type annotations, mypy --strict compliant
3. **Legacy Compatibility**: Temporary shim properties for migration
4. **Pipeline Interfaces**: Type-safe protocols for all pipeline components
5. **Adapter Pattern**: Wrap legacy implementations without modification

## Architecture

```
SimulationEngine
       │
       ▼
  NetworkState  ◄──── Single instance per simulation
       │
       ├── _topology: nx.Graph (read-only after init)
       ├── _spectrum: dict[link, LinkSpectrum]
       ├── _lightpaths: dict[int, Lightpath]
       └── _config: SimulationConfig (immutable)
```

## Sub-phases

| Sub-phase | Directory | Description |
|-----------|-----------|-------------|
| P2.1 | `P2.1_network_state_core/` | NetworkState and LinkSpectrum with read-only operations |
| P2.2 | `P2.2_network_state_writes_legacy/` | Write methods and legacy compatibility properties |
| P2.3 | `P2.3_pipeline_protocols/` | Type-safe pipeline interfaces (Routing, Spectrum, etc.) |
| P2.4 | `P2.4_legacy_adapters/` | Adapters wrapping legacy implementations |

## Execution Order

```
P2.1 NetworkState Core
       │
       ▼
P2.2 Write Methods + Legacy Compat
       │
       ▼
P2.3 Pipeline Protocols
       │
       ▼
P2.4 Legacy Adapters
       │
       ▼
Phase 2 Complete
```

**Within each sub-phase**, execute micro-tasks in alphabetical order (a, b, c, ...).

## Critical Constraints

### Must Follow
- **Additive only**: No changes to existing function signatures
- **`run_comparison.py` must pass unchanged** throughout Phase 2
- **NetworkState is a state container, NOT a routing algorithm**
- **Single instance policy**: Exactly ONE NetworkState per simulation
- **Pass-by-reference**: Pipelines receive NetworkState as parameter, never store it

### Must Avoid
- Caching NetworkState internally in any component
- Direct numpy array manipulation outside NetworkState
- Hardcoding paths or magic strings
- Circular imports between domain/interfaces modules

## Files Created by Phase 2

### New Source Files
```
fusion/
├── domain/
│   └── network_state.py        # P2.1, P2.2
├── interfaces/
│   ├── __init__.py             # P2.3
│   └── pipelines.py            # P2.3
└── core/
    └── adapters/
        ├── __init__.py         # P2.4
        ├── routing_adapter.py  # P2.4
        ├── spectrum_adapter.py # P2.4
        ├── grooming_adapter.py # P2.4
        └── snr_adapter.py      # P2.4
```

### New Test Files
```
fusion/tests/
├── domain/
│   └── test_network_state.py   # P2.1, P2.2
└── adapters/
    ├── test_routing_adapter.py # P2.4
    ├── test_spectrum_adapter.py # P2.4
    └── ...
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| NetworkState owns spectrum arrays | Single point of mutation prevents inconsistencies |
| LinkSpectrum is a dataclass | Simple, typed container for per-link state |
| Protocols are typing.Protocol | Type-only, no runtime overhead, no circular imports |
| Legacy properties return exact shapes | Enables gradual migration without breaking callers |
| Adapters implement protocols | Clean interface while wrapping legacy code |

## Phase 2 Exit Criteria

- [ ] `NetworkState` instantiates correctly from topology
- [ ] `LinkSpectrum` manages per-link spectrum with band/core support
- [ ] Read methods return correct data (`get_lightpath`, `is_spectrum_available`)
- [ ] Write methods mutate state correctly (`create_lightpath`, `release_lightpath`)
- [ ] Legacy properties match `sdn_props` format exactly
- [ ] All pipeline protocols defined and pass mypy --strict
- [ ] All adapters implement their protocols
- [ ] Adapters produce identical results to legacy code
- [ ] `pytest fusion/tests/domain/test_network_state.py` passes
- [ ] `pytest fusion/tests/adapters/` passes
- [ ] `run_comparison.py` still passes (no behavioral changes)
- [ ] `ruff check` and `mypy --strict` pass on all new files

## Gap Analysis

**Important**: A comprehensive gap analysis was completed to identify missing documentation and ensure all legacy state sources are properly mapped. See `P2.0_gap_analysis.md` for:

- All legacy state containers and their authority mapping
- Missing fields in lightpath_status_dict entries
- Guard band encoding convention
- Protected lightpath (1+1) allocation details
- Pipeline protocol rollback contracts
- Stateless adapter pattern requirements

Key additions from gap analysis:
- `P2.1.shared_context_state_authority.md` - Complete state authority mapping
- Extended `P2.2.b_design_write_methods.md` - Guard bands, protection, bandwidth management
- Extended `P2.3.d_design_grooming_snr_slicing_protocols.md` - Complete result types, SNR recheck
- Extended `P2.4.index.md` - SlicingAdapter, stateless adapter pattern

## Reference Documents

- **Gap Analysis**: `P2.0_gap_analysis.md` (critical - read first)
- **Migration Spec**: `.claude/v4-docs/migration/phase_2_state_management.md`
- **Phase 1 Docs**: `.claude/v5-final-docs/phase-1-core-domain/`
- **Domain Model**: `.claude/v4-docs/architecture/domain_model.md`
- **NetworkState Authority**: `.claude/v4-docs/decisions/0006-networkstate-authority.md`
- **Orchestrator Design**: `.claude/v4-docs/decisions/0007-orchestrator-design.md`

## Navigation

- [P2.1 NetworkState Core](./P2.1_network_state_core/P2.1.index.md)
- [P2.2 Write Methods + Legacy Compat](./P2.2_network_state_writes_legacy/P2.2.index.md)
- [P2.3 Pipeline Protocols](./P2.3_pipeline_protocols/P2.3.index.md)
- [P2.4 Legacy Adapters](./P2.4_legacy_adapters/P2.4.index.md)
