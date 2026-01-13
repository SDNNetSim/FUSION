# Phase 1: Foundation & Setup

## 01 - Scope Boundaries

**Purpose**: Define clear boundaries for v1 implementation to avoid scope creep and manage expectations.

---

## SHALL (v1 Requirements)

These features **MUST** be implemented in v1:

### 1. Failure Module
- ✅ Link failure (F1)
- ✅ Node failure (adjacent links)
- ✅ SRLG failure (F3)
- ✅ Geographic failure (F4, hop-radius based)
- ✅ Failure repair at specified time
- ✅ Path feasibility checking with failures

### 2. Protection Mechanisms
- ✅ 1+1 disjoint protection routing
- ✅ Link-disjoint primary and backup paths
- ✅ Spectrum reservation on both paths
- ✅ Protection switchover with configurable latency (default 50ms)
- ✅ Restoration with configurable latency (default 100ms)

### 3. K-Path Infrastructure
- ✅ K-path candidate pre-computation and caching (K=4 default)
- ✅ Path ordering by hops/length
- ✅ Path feature extraction:
  - path_hops
  - min_residual_slots
  - frag_indicator
  - failure_mask
  - dist_to_disaster_centroid

### 4. RL Policy Support
- ✅ PathPolicy interface for all policies
- ✅ Baseline policies (KSP-FF, 1+1)
- ✅ BC (Behavior Cloning) policy inference
- ✅ IQL (Implicit Q-Learning) policy inference
- ✅ Action masking based on failure and spectrum
- ✅ Heuristic fallback when all actions masked

### 5. Dataset Generation
- ✅ Offline dataset logging to JSONL
- ✅ State/action/reward/mask/meta tuple format
- ✅ Epsilon-mix for behavior diversity (ε=0.1 default)

### 6. Metrics & Reporting
- ✅ Blocking probability (overall + failure window)
- ✅ Recovery time (mean, P95)
- ✅ Fragmentation proxy metrics
- ✅ Decision time logging
- ✅ Multi-seed aggregation (mean, std, CI95)
- ✅ CSV export for batch results

### 7. Configuration System
- ✅ Survivability experiment template (INI)
- ✅ Schema validation for new parameters
- ✅ Failure settings (type, timing, targets)
- ✅ Protection settings (mode, latencies)
- ✅ RL policy settings (type, model paths, device)
- ✅ Dataset logging settings

### 8. Determinism
- ✅ Seed all RNGs (Python, NumPy, PyTorch)
- ✅ Record seed in all outputs
- ✅ Full reproducibility with same seed

---

## SHALL NOT (v1 Exclusions)

To maintain v1 scope and avoid feature creep, these features are **explicitly excluded**:

### 1. Physical Layer Modeling
- ❌ QoT/OSNR/impairment modeling
- ❌ Modulation format adaptation
- ❌ Nonlinear effects (FWM, XPM, SPM)
- ❌ EDFA noise accumulation

**Rationale**: Focus on network-level survivability. Physical layer can be added in v2.

### 2. SDN Controller Simulation
- ❌ Real controller internals (queues, RPCs, flow rules)
- ❌ OpenFlow protocol emulation
- ❌ Controller failover/redundancy

**Rationale**: Timing is **parameterized** (protection_switchover_ms, restoration_latency_ms), not simulated.

### 3. Advanced RL Infrastructure
- ❌ Hierarchical/meta/multi-agent RL
- ❌ Recurrent PPO with LSTM
- ❌ Distributional RL (C51, QR-DQN, IQN)
- ❌ Online fine-tuning during simulation

**Rationale**: v1 focuses on **offline inference** only. Conservative offline RL (BC, IQL) is sufficient.

### 4. Spectrum Management
- ❌ Spectrum defragmentation algorithms
- ❌ Dynamic spectrum reallocation
- ❌ Multi-band (C+L) unless time permits

**Rationale**: Use existing First-Fit/Best-Fit/Last-Fit. Defrag adds complexity beyond v1 goals.

### 5. Arbitrary Route Construction
- ❌ Constructing paths link-by-link
- ❌ Partial path modification

**Rationale**: Policies select from **K pre-computed candidate paths**. This simplifies action space.

### 6. Multiple Failure Events
- ❌ Multiple failures per simulation run
- ❌ Stochastic repair times
- ❌ Cascading failure models

**Rationale**: v1 supports **one failure event per run** for tractability. Multi-event is v2+.

### 7. Interpretability Features
- ❌ Policy distillation (decision trees)
- ❌ SHAP values / feature importance
- ❌ Monotonic probes

**Rationale**: Focus on performance first. Interpretability is post-v1.

### 8. Advanced Protection
- ❌ Revert-to-primary with hysteresis
- ❌ Shared backup path pooling
- ❌ M:N protection schemes

**Rationale**: v1 implements **1+1 dedicated protection** only.

---

## Nice-to-Have (Post-v1 Candidates)

Features to consider **after v1 is validated**:

### Priority: High (Likely v2)

1. **Recurrent PPO Comparator**
   - LSTM-based policy for temporal dependencies
   - Compare against BC/IQL on transient failures
   - **Effort**: 3-5 days

2. **Multiple Failure Events**
   - Stochastic repair times (exponential distribution)
   - Multiple failures per run (Poisson arrivals)
   - **Effort**: 2-3 days

3. **Multi-Band (C+L) Sensitivity**
   - Extend spectrum to C+L bands
   - Measure BP improvement with doubled capacity
   - **Effort**: 1-2 days

### Priority: Medium (Likely v3)

4. **Distributional RL Heads**
   - C51/QR-DQN/IQN for tail risk modeling
   - Quantile-based risk-aware policies
   - **Effort**: 4-6 days

5. **Policy Distillation**
   - Distill RL policy to depth-3 decision tree
   - CART with max_depth=3, compare BP loss
   - **Effort**: 2-3 days

6. **Monotonic Probes**
   - Feature importance validation
   - Ensure BP decreases as residual slots increase
   - **Effort**: 1-2 days

7. **Revert-to-Primary Behavior**
   - Switch back to primary after repair
   - Hysteresis timing to avoid flapping
   - **Effort**: 1 day

### Priority: Low (Research Extensions)

8. **Online Fine-Tuning**
   - 10-50k steps of on-policy data
   - Conservative fine-tuning with KL constraints
   - **Effort**: 5-7 days

9. **GNN Encoders**
   - Graph neural network for state representation
   - Compare against hand-crafted features
   - **Effort**: 7-10 days

10. **Meta-Learning / Hierarchical RL**
    - Task distribution across topologies/loads
    - Meta-RL for rapid adaptation
    - **Effort**: 10-15 days

---

## Out of Scope (Indefinitely)

Features that are **not planned** for this project line:

1. **Real-time Hardware-in-the-Loop**
   - Physical testbed integration
   - Hardware SDN controllers

2. **Optical Circuit Switching**
   - Wavelength-routed networks (non-EON)
   - Fixed-grid DWDM

3. **Network Slicing / Multi-Tenancy**
   - QoS classes and SLAs
   - Tenant isolation

4. **Full-Stack Orchestration**
   - VM placement and migration
   - Service function chaining

---

## Scope Decision Matrix

Use this matrix to decide if a feature belongs in v1:

| Question | v1 Yes | v1 No |
|----------|--------|-------|
| Does it directly support survivability testing (BP, recovery time)? | ✅ | ❌ |
| Does it require < 2 days of implementation? | ✅ | ❌ |
| Is it required for fair baseline comparison (KSP-FF, 1+1)? | ✅ | ❌ |
| Does it integrate cleanly with existing FUSION architecture? | ✅ | ❌ |
| Is it testable with unit + integration tests? | ✅ | ❌ |

**Example Applications**:

- **Q**: Should we add GNN encoders?
  **A**: No → Not required for survivability testing, > 7 days effort

- **Q**: Should we add fragmentation proxy metrics?
  **A**: Yes → Required for paper claims, < 1 day effort, integrates with existing statistics

- **Q**: Should we add online fine-tuning?
  **A**: No → Not required for v1 offline RL focus, > 5 days effort, post-v1

---

## Scope Change Process

If a feature is proposed during implementation:

1. **Evaluate** against the decision matrix above
2. **Check effort** against 13-17 day budget (see [60-work-breakdown.md](../phase7-management/60-work-breakdown.md))
3. **Assess priority**:
   - **Must-have for paper claims?** → Consider adding to v1
   - **Nice-to-have but not critical?** → Defer to post-v1
   - **Out of scope entirely?** → Document and close

4. **Update documentation** if scope changes are approved

---

## Summary Table

| Category | Count | Status |
|----------|-------|--------|
| **SHALL (v1)** | 8 major areas | Required |
| **SHALL NOT (v1)** | 8 exclusion areas | Explicitly out |
| **Nice-to-Have** | 10 features | Post-v1 candidates |
| **Out of Scope** | 4 areas | Not planned |

---

## Next Steps

After understanding scope boundaries:

1. **Review** [02-module-summary.md](02-module-summary.md) for module breakdown
2. **Check** [03-version-control.md](03-version-control.md) for Git workflow
3. **Proceed** to Phase 2 for implementation

---

**Related Documents**:
- [00-overview.md](00-overview.md) (Project context)
- [60-work-breakdown.md](../phase7-management/60-work-breakdown.md) (13-17 day budget)
- [64-checklist.md](../phase7-management/64-checklist.md) (Final checklist)
