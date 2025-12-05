# Phase 3: SDNOrchestrator Integration

## Overview

Phase 3 introduces the `SDNOrchestrator` and `PipelineFactory`, creating a thin coordination layer that routes requests through pipelines. This phase addresses the core architectural concern: SDN as a router of pipelines, not a place where algorithm logic lives.

## Prerequisites

Before starting Phase 3, ensure:
- **Phase 1 complete**: Domain objects (`SimulationConfig`, `Request`, `Lightpath`, result types)
- **Phase 2 complete**: `NetworkState` with legacy compatibility, pipeline protocols, adapters

## Objectives

1. Create `PipelineFactory` and `PipelineSet`
2. Create `SDNOrchestrator` with pipeline coordination
3. Add feature flag to switch between old and new paths
4. Integrate `StatsCollector` with orchestrator
5. Verify with `run_comparison.py`

## Sub-phases

| Sub-phase | Directory | Description | Files |
|-----------|-----------|-------------|-------|
| P3.1 | `P3.1_pipeline_factory/` | Factory that produces pipelines based on config | 7 |
| P3.2 | `P3.2_sdn_orchestrator/` | Thin coordination layer for request handling | 10 |
| P3.3 | `P3.3_feature_flag_and_wiring/` | Feature flag and simulation integration | 7 |
| P3.4 | `P3.4_stats_integration/` | StatsCollector integration with orchestrator | 6 |
| P3.5 | `P3.5_run_comparison_and_rollback/` | Verification and rollback planning | 5 |
| P3.6 | `P3.6_gap_analysis.md` | Gap analysis against V3 specifications | 1 |

## Key Architectural Principles

### SDN as Pipeline Router

```
OLD Architecture:              NEW Architecture:
SDNController                  SDNOrchestrator
  +-- grooming logic             +-- calls GroomingPipeline
  +-- slicing logic              +-- calls RoutingPipeline
  +-- routing logic              +-- calls SpectrumPipeline
  +-- spectrum logic             +-- calls SNRPipeline
  +-- many if/else               +-- calls SlicingPipeline
                                 +-- thin coordination only
```

### Orchestrator Rules

| Rule | Description |
|------|-------------|
| No algorithm logic | K-shortest-path, first-fit, SNR calculation belong in pipelines |
| No numpy access | Only through NetworkState methods |
| No state storage | Receives `NetworkState` per call, never stores it |
| Size limits | < 200 lines total, < 50 lines per method |

### Feature Flag

- Default: `use_orchestrator=False` (uses old SDNController path)
- New path activated by: `use_orchestrator=True`
- Both paths produce equivalent results

## Files Created by Phase 3

### New Files

| File | Purpose |
|------|---------|
| `fusion/core/pipeline_factory.py` | Factory for creating pipelines |
| `fusion/core/orchestrator.py` | Thin coordination layer |
| `fusion/pipelines/__init__.py` | Pipeline package |

### Modified Files

| File | Changes |
|------|---------|
| `fusion/core/simulation.py` | Feature flag, orchestrator integration |
| `fusion/stats/collector.py` | Record from AllocationResult |

## Navigation

1. Start with [P3.1 Pipeline Factory](P3.1_pipeline_factory/P3.1.index.md)
2. Then [P3.2 SDN Orchestrator](P3.2_sdn_orchestrator/P3.2.index.md)
3. Then [P3.3 Feature Flag & Wiring](P3.3_feature_flag_and_wiring/P3.3.index.md)
4. Then [P3.4 Stats Integration](P3.4_stats_integration/P3.4.index.md)
5. Finally [P3.5 Run Comparison](P3.5_run_comparison_and_rollback/P3.5.index.md)

## Gap Analysis Additions (V3 Compliance)

The following documents were added to address gaps identified when comparing v5 Phase 3 docs against the V2/V3 architecture refactor plans and v4 ADRs:

### HIGH Priority Documents

| Document | Path | Description |
|----------|------|-------------|
| Grooming Rollback | `P3.2_sdn_orchestrator/P3.2.f_grooming_rollback_specification.md` | Complete rollback semantics when partial grooming fails |
| Protection Pipeline | `P3.2_sdn_orchestrator/P3.2.g_protection_pipeline_integration.md` | 1+1 protection stages and atomicity |
| Congestion Handling | `P3.2_sdn_orchestrator/P3.2.h_congestion_handling_specification.md` | SNR recheck and utilization rollback |

### MEDIUM Priority Documents

| Document | Path | Description |
|----------|------|-------------|
| Routing Strategy | `P3.1_pipeline_factory/P3.1.e_routing_strategy_pattern.md` | Strategy pattern for pluggable routing |
| Feature Flag Interactions | `P3.3_feature_flag_and_wiring/P3.3.e_feature_flag_interactions.md` | Flag matrix and validation rules |

### Updated Shared Contexts

| Document | Update |
|----------|--------|
| `P3.2.shared_context_orchestrator_responsibilities.md` | Added NetworkState sharing rules (V3 compliance) |
| `P3.5.shared_context_run_comparison_expectations.md` | Added non-equivalence indicators, bias detection |

### Gap Analysis Summary

See `P3.6_gap_analysis.md` for the complete gap analysis including:
- Gap categories and severity
- V3 specification references
- Remediation status

## Related Documentation

### v4 Architecture Docs
- `.claude/v4-docs/migration/phase_3_orchestrator.md` - Phase 3 specification
- `.claude/v4-docs/architecture/orchestration.md` - Orchestrator design
- `.claude/v4-docs/architecture/pipelines.md` - Pipeline implementations
- `.claude/v4-docs/architecture/protection_pipeline.md` - Protection stages
- `.claude/v4-docs/decisions/0007-orchestrator-design.md` - ADR for orchestrator
- `.claude/v4-docs/decisions/0008-routing-strategy-pattern.md` - ADR for routing
- `.claude/v4-docs/decisions/0012-protection-pipeline.md` - ADR for protection

### Phase 1-2 Docs (v5)
- `.claude/v5-final-docs/phase-1-core-domain/` - Domain objects
- `.claude/v5-final-docs/phase-2-state-management/` - NetworkState and adapters

### V3 Architecture Reference
- `.claude/ARCHITECTURE_REFACTOR_PLAN_V3.md` - Canonical architecture source

## Exit Criteria

Phase 3 is complete when:
- [ ] `PipelineFactory` creates correct pipelines for all config combinations
- [ ] `SDNOrchestrator` coordinates pipelines correctly
- [ ] Feature flag switches cleanly between paths
- [ ] `StatsCollector` records all metrics from orchestrator
- [ ] `run_comparison.py` passes with both paths
- [ ] No performance regression (< 5% slower)
- [ ] Code passes ruff, mypy, and tests
