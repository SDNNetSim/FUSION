# Phase 7: Project Management

## 61 - Risks & Mitigations

**Section Reference**: Section 8 - Risks & Mitigations

**Purpose**: Identify potential risks to survivability v1 implementation and provide mitigation strategies.

---

## Risk Matrix

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| **Narrow offline logs → poor generalization** | High | Medium | Domain randomization, epsilon-mix, conservative RL (IQL) |
| **Incorrect action masking → artificial wins/losses** | High | Medium | Extensive unit tests, cross-validation with path feasibility |
| **Timing misinterpretation** | Medium | Low | Clear documentation, parameterized latencies, no micro-benchmarks |
| **State-feature drift** | Medium | Medium | Version state schema, store schema with datasets |
| **K-path cache memory explosion** | Medium | Low | Limit K ≤ 5, compress paths, lazy loading |
| **RL model loading failures** | Low | High | Fallback to heuristic, graceful error handling |

---

## Detailed Mitigations

### 1. Narrow Offline Logs → Poor Generalization

**Risk**: RL policies trained on narrow behavior distribution fail under different conditions.

**Mitigation**:
- Use **epsilon-mix** (10% exploration) during dataset collection
- Apply **domain randomization**: vary loads, topologies, failure types
- Use **conservative offline RL (IQL)** that stays close to behavior policy
- Validate on held-out scenarios before deployment

**Indicators**:
- High variance in RL policy performance across seeds
- Poor performance outside training distribution

---

### 2. Incorrect Action Masking → Artificial Performance

**Risk**: Bugs in action masking give RL policy unfair advantage or disadvantage.

**Mitigation**:
- **Extensive unit tests** for `compute_action_mask()`
- **Cross-validate** with `FailureManager.is_path_feasible()`
- **Log mask statistics**: % masked, distribution across requests
- **Compare baselines** with same masking logic

**Indicators**:
- Unexpectedly perfect or terrible RL performance
- Inconsistent BP between heuristic and RL with same paths

---

### 3. Timing Misinterpretation

**Risk**: Readers misinterpret parameterized timing as detailed SDN simulation.

**Mitigation**:
- **Clear documentation** that timing is parameterized, not simulated
- Use **realistic defaults** (50ms switchover, 100ms restoration)
- **Cite sources** for timing parameters
- Avoid **micro-benchmarks** (<1ms granularity)

**Indicators**:
- Questions about SDN controller implementation details
- Requests for sub-millisecond timing precision

---

### 4. State-Feature Drift

**Risk**: State feature definitions change, breaking compatibility with saved datasets/models.

**Mitigation**:
- **Version state schema** in dataset files
- **Store schema** with datasets (JSON header)
- **Validate features** during model loading
- **Document feature definitions** explicitly

**Indicators**:
- Model loading errors after code changes
- Unexplained drops in model performance

---

### 5. K-Path Cache Memory Explosion

**Risk**: Large topologies (N>100) or high K exhaust memory.

**Mitigation**:
- **Limit K ≤ 5** in configuration schema
- **Compress paths**: store as node lists, not full objects
- **Lazy loading**: compute paths on-demand for large topologies
- **Memory monitoring**: fail gracefully if budget exceeded

**Indicators**:
- OOM errors during cache initialization
- Excessive memory usage (>1GB)

---

### 6. RL Model Loading Failures

**Risk**: Model file missing, corrupted, or incompatible.

**Mitigation**:
- **Fallback to heuristic** if model loading fails
- **Graceful error handling** with informative messages
- **Validate model files** at startup (check existence, load test)
- **Document model format** requirements

**Indicators**:
- Simulation crashes on startup
- FileNotFoundError or model loading exceptions

---

## Monitoring & Detection

### Automated Checks
```python
# In SimulationEngine.__init__()
def _validate_survivability_setup(self) -> None:
    """Run health checks for survivability features."""
    if self.k_path_cache:
        mem_mb = self.k_path_cache.get_memory_estimate_mb()
        if mem_mb > 500:
            logger.warning(f"K-path cache using {mem_mb:.1f}MB (>500MB)")

    if self.dataset_logger:
        if self.transition_count > 1e6:
            logger.warning("Dataset has >1M transitions, consider splitting")
```

---

## Acceptance Criteria

- [x] All high-impact risks have mitigation plans
- [x] Mitigation strategies implemented where feasible
- [x] Monitoring checks in place for key risks
- [x] Documentation warns about known limitations

---

**Related Documents**:
- [50-testing.md](../phase6-quality/50-testing.md) (Testing mitigations)
- [62-traceability.md](62-traceability.md) (Paper claim verification)
