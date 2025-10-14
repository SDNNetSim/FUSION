# Phase 7: Project Management

## 60 - Minimal Work Breakdown

**Section Reference**: Section 9 - Minimal Work Breakdown

**Purpose**: Provide estimated effort for each major component to support project planning and resource allocation.

---

## Task Breakdown

| Task | Module | Estimated Days | Priority |
|------|--------|----------------|----------|
| Failure module (F1/F3/F4) | `fusion/modules/failures/` | 1.5-2 | P0 |
| K-path cache + features + masks | `fusion/modules/routing/k_path_cache.py` | 1 | P0 |
| 1+1 protection + switchover | `fusion/modules/routing/one_plus_one_protection.py` | 1.5-2 | P1 |
| RL policy interface + loaders | `fusion/modules/rl/policies/` | 1.5 | P1 |
| Action masking + fallback | `fusion/modules/rl/policies/action_masking.py` | 0.5 | P1 |
| Dataset logger | `fusion/reporting/dataset_logger.py` | 1 | P1 |
| Recovery metrics + timing | `fusion/reporting/statistics.py` | 1 | P1 |
| Configuration extension | `fusion/configs/` | 0.5 | P0 |
| Integration (SimulationEngine, SDNController) | `fusion/core/` | 1.5-2 | P0 |
| Unit tests + fixtures | `tests/` in each module | 2-3 | P0 |
| Integration tests | `tests/integration/` | 1 | P1 |
| Documentation (READMEs, docstrings) | All modules | 1 | P2 |
| **Total** | | **13-17 days** | |

---

## Implementation Sequence

### Phase 1: Foundation (Days 1-2)
1. Failure module
2. Configuration extension

### Phase 2: Core Features (Days 3-6)
3. K-path cache
4. RL policy interface
5. Action masking
6. Dataset logger

### Phase 3: Protection & Metrics (Days 7-10)
7. 1+1 protection
8. Recovery metrics
9. Integration

### Phase 4: Testing & Documentation (Days 11-13)
10. Unit tests
11. Integration tests
12. Documentation

---

## Dependencies

```
Failure Module
  └─> K-Path Cache
       ├─> RL Policies
       │    └─> Action Masking
       │         └─> Integration
       └─> 1+1 Protection
            └─> Recovery Metrics
                 └─> Dataset Logger
```

---

**Related Documents**:
- [61-risks-mitigations.md](61-risks-mitigations.md) (Risk management)
- [63-usage-workflow.md](63-usage-workflow.md) (Usage examples)
