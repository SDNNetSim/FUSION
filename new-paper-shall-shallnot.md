# EON Simulator v1 Additions — SHALL / SHALL NOT Specification

**Purpose:** Define the minimal, testable changes required to support **offline + conservative RL** evaluation for survivability in an Elastic Optical Network (EON) simulator. This document enumerates **scope, requirements, non‑goals, APIs, metrics, and acceptance tests** for v1.

---

## 0) Scope (v1)

v1 enables stress-testing KSP‑FF, 1+1 protection, and an **offline RL policy (BC → IQL)** with **action masking + heuristic fallback** under **F1 (link), F3 (SRLG), F4 (geo radius=2)** failures. We will **not** implement detailed SDN internals, QoT, or spectrum defragmentation algorithms.

---

## 1) SHALL — Mandatory v1 Requirements

### 1.1 Failure/Disaster Module
- **SHALL** implement failure injection functions:
  - `fail_link(link_id, t_fail, t_repair)`
  - `fail_node(node_id, t_fail, t_repair)`
  - `fail_srlg(srlg_id, t_fail, t_repair)` (SRLG = set of links)
  - `fail_geo(center_node_id, hop_radius, t_fail, t_repair)` (affects all links whose shortest-path hop distance to center ≤ radius).
- **SHALL** support **one event per run** (v1), with parameters sourced from config.
- **SHALL** maintain an **active failure set** and expose a **path feasibility** query respecting failures.

### 1.2 1+1 Disjoint Protection + Restoration
- **SHALL** add a provisioning option `protection_mode ∈ {"none","1plus1"}`.
- For `1plus1`, the simulator **SHALL**:
  - Find **link-disjoint** primary/backup paths (Yen + Suurballe or two disjoint runs of Yen with link banning).
  - **Reserve spectrum** on both paths at setup (FF).
  - On failure affecting the primary, **SHALL** switch to backup with fixed **protection_switchover_ms** (default 50 ms).
  - **SHALL** release backup when primary is repaired **only if** policy is configured to revert (`revert_to_primary: true/false`).

### 1.3 Candidate Path Generation
- **SHALL** compute and cache **K shortest simple paths** per (src,dst) using Yen’s algorithm; ordering = **hops** (default).
- **SHALL** expose candidate paths API with **per-path failure flags** and **min residual slots**.

### 1.4 RL Policy Integration (Offline Inference)
- **SHALL** accept a **pluggable policy interface**:
  ```python
  class PathPolicy:
      def select(self, state: Dict, action_mask: List[bool]) -> int:
          ...
  ```
- **SHALL** provide a **BC policy** loader (PyTorch `.pt`) and an **IQL policy** loader (optional in v1 if model is exported to Torch).
- **SHALL** compute the **action mask** each decision: mask infeasible paths (failed links or insufficient contiguous slots). If all masked, **SHALL** fall back to **KSP‑FF** (or 1+1 route when configured).

### 1.5 Offline Dataset Logging
- **SHALL** log tuples `(s, a, r, s', action_mask, backup_available_flag, meta)` to parquet/JSONL with schema defined in §5.
- **SHALL** support behavior logging for **KSP‑FF** and **1+1 + restoration** policies, with an `epsilon_mix` option to pick second‑best path with probability **p ∈ [0,0.2]**.

### 1.6 Recovery Time Modeling (Emulated SDN)
- **SHALL** add two timing parameters:
  - `protection_switchover_ms` (default 50)
  - `restoration_latency_ms` (default 100; sweepable)
- **SHALL** measure per‑event **recovery_time_ms** as the time between failure and the moment **all restorable connections** affected are up (backup or restoration).
- **SHALL** track **per‑request decision_time_ms** (policy + path compute) separately from the fixed restoration latency.

### 1.7 Metrics & Reporting
- **SHALL** compute:
  - **Blocking Probability (BP)** overall and within **failure window** `[t_fail, t_fail+Δ]` (Δ configurable, default 1000 arrivals).
  - **Bandwidth Blocking Probability**.
  - **Recovery time**: mean and **P95** per scenario.
  - **Fragmentation proxy** per path request (e.g., `1 − largest_contig_block / total_free_slots`).
  - **Runtime**: decision_time_ms stats.
  - **Seed variance**: aggregate over ≥5 seeds (export CSV).
- **SHALL** export results to CSV/JSON for plotting.

### 1.8 Configuration
- **SHALL** load a single YAML/JSON run config (see §4) to control traffic (loads), failures, timing, K, and policy choices.

### 1.9 Determinism & Seeds
- **SHALL** seed all RNGs (Python, NumPy, Torch) and record `seed` in outputs.

---

## 2) SHALL NOT — v1 Non‑Goals / Exclusions

- **SHALL NOT** model QoT/OSNR/impairments, modulation adaptation, or spectrum defragmentation algorithms.
- **SHALL NOT** simulate real SDN controller internals (queues, RPCs, flow rules); timing is **parameterized**, not simulated.
- **SHALL NOT** implement hierarchical/meta/multi‑agent RL infrastructure in v1.
- **SHALL NOT** support arbitrary route construction (only K‑candidate path choice).
- **SHALL NOT** change spectrum assignment policy from **First‑Fit** in v1.

---

## 3) Nice‑to‑Have (Post‑v1)

- Recurrent PPO comparator; Distributional head for critic (QR‑DQN/C51).
- Multi‑band (C+L) sensitivity.
- Multiple failure events per run; stochastic repair times.
- Policy distillation report (depth‑3 tree) and monotonic probes.
- Revert‑to‑primary behavior toggles and hysteresis timing.

---

## 4) Configuration (Example YAML)

```yaml
topology: NSFNET_14
spectrum:
  slots_per_link: 80
paths:
  K: 4
  ordering: hops
traffic:
  loads_erlang: [50, 100, 150]      # choose one per run
  demand_slots: [1,3]               # inclusive uniform
  arrival: poisson
  holding: exponential
failure:
  type: F3                          # F0|F1|F3|F4
  t_fail_arrival_index: uniform_mid  # or numeric
  t_repair_after_arrivals: 1000
  srlg_links: [ (u1,v1), (u2,v2) ]  # for F3
  geo:
    center_node: 5
    hop_radius: 2                   # for F4
sdn_timing:
  protection_switchover_ms: 50
  restoration_latency_ms: 100
policy:
  mode: "ksp_ff"                    # ksp_ff | one_plus_one | rl_bc | rl_iql
  epsilon_mix_second_best: 0.1      # for dataset logging
  fallback_on_all_masked: "ksp_ff"  # or "one_plus_one"
logging:
  dataset_out: "datasets/nsfnet_v1.jsonl"
  results_out: "results/nsfnet_v1.csv"
  seed: 12345
```

---

## 5) Data & Telemetry Schemas

### 5.1 Logged Transition (JSONL / Parquet)
```json
{
  "t": 12345,
  "seed": 12345,
  "src": 3, "dst": 9, "slots_needed": 2, "est_hold": 1.7,
  "is_disaster": 0,
  "paths": [
    {"hops": 5, "min_residual": 8, "frag": 0.42, "failure_mask": 0, "dist_to_centroid": 1},
    {"hops": 4, "min_residual": 3, "frag": 0.55, "failure_mask": 0, "dist_to_centroid": 2},
    {"hops": 6, "min_residual": 9, "frag": 0.33, "failure_mask": 1, "dist_to_centroid": 0},
    {"hops": 7, "min_residual": 11, "frag": 0.37, "failure_mask": 0, "dist_to_centroid": 3}
  ],
  "action_mask": [true, true, false, true],
  "a": 0,                      // chosen path index
  "r": 1,                      // +1 accept, -1 block
  "accepted": 1,
  "backup_available_flag": 1,  // for 1+1
  "decision_time_ms": 0.35,
  "restoration_latency_ms": 100,
  "bp_window_tag": "pre|fail|post"
}
```

### 5.2 Results Row (CSV/JSON)
```
topology, load, failure, K, policy, seed, BP_overall, BP_window_fail_mean, BP_window_fail_p95,
recovery_time_mean_ms, recovery_time_p95_ms, frag_proxy_mean, decision_time_mean_ms
```

---

## 6) Public APIs (Python)

```python
# 6.1 Failure module
def fail_link(link_id: int, t_fail: int, t_repair: int): ...
def fail_node(node_id: int, t_fail: int, t_repair: int): ...
def fail_srlg(links: list[tuple[int,int]], t_fail: int, t_repair: int): ...
def fail_geo(center_node: int, hop_radius: int, t_fail: int, t_repair: int): ...

# 6.2 Candidate paths
def get_k_paths(src: int, dst: int, K: int) -> list[list[int]]: ...
def path_features(path: list[int]) -> dict: ...  # hops, min_residual, frag, failure_mask, dist

# 6.3 Policies
class PathPolicy:
    def select(self, state: dict, action_mask: list[bool]) -> int: ...

class KSPFFPolicy(PathPolicy): ...
class OnePlusOnePolicy(PathPolicy): ...
class RLBCPolicy(PathPolicy): ...      # loads Torch model, pure inference
class RLIQLPolicy(PathPolicy): ...     # loads Torch model, pure inference

# 6.4 Dataset logging
def log_transition(transition: dict): ...
```

---

## 7) Acceptance Tests (must pass)

1. **Failure correctness**: a path traversing any failed link is infeasible; all such paths are masked during failure.
2. **1+1 behavior**: when primary fails, backup is activated after `protection_switchover_ms`; BP in-window reflects zero blocks for protected demands with available backup.
3. **Restoration latency accounting**: restoration actions complete after `restoration_latency_ms`; recovery_time_ms is computed accordingly.
4. **Mask + fallback**: when action_mask disables all K paths, the simulator falls back to configured baseline without crash.
5. **Metrics reproducibility**: rerunning with the same `seed` yields identical BP and recovery aggregates.
6. **Dataset integrity**: logged tuples contain valid masks and features; no NaNs; schema validated.

---

## 8) Performance Budgets

- **Decision-time overhead** (policy + masking + path compute): **≤ 2 ms** per request on commodity CPU for K≤5.
- **Failure processing** (mask updates) under F4 radius=2 on NSFNET: **≤ 10 ms** amortized per affected request.
- **Logging throughput**: ≥ 50k transitions/minute to JSONL/Parquet.

---

## 9) Risks & Mitigations

- **Narrow logs → poor generalization**: mitigate with **domain randomization**; widen behavior sources; epsilon-mix second-best path.
- **Incorrect masking → artificial wins/losses**: unit-test masks; cross-check with path feasibility engine.
- **Timing misinterpretation**: keep restoration latency parameterized and clearly documented; never claim controller micro-benchmarks.
- **State-feature drift**: version state schema and store alongside datasets.

---

## 10) Minimal Work Breakdown

- Failure module (F1/F3/F4): 1–2 days
- K-path cache + features + masks: 1 day
- 1+1 protection/reservation + switchover: 1–2 days
- Timing model & recovery metrics: 0.5–1 day
- Policy interface + BC loader: 1 day
- Dataset logger + schema validation: 1 day
- Reports (CSV/JSON) + seeds: 0.5 day
- Tests & sanity plots: 1 day

---

## 11) Out of Scope (explicit)

- GNN encoders, meta/hierarchical/multi-agent RL infrastructure
- QoT/impairment-aware routing, spectrum defragmentation
- Multi-event cascading failures, real controller emulation

---

## 12) Traceability to Paper Claims

- **BP & variance** → §1.7 metrics, seeds §1.9
- **Recovery time** → §1.6 timing & measurement
- **Failure types** → §1.1
- **Safety (mask + fallback)** → §1.4
- **Offline dataset** → §1.5 and §5
- **Baseline fairness (KSP-FF, 1+1)** → §1.2 and §1.3

---

*End of v1 SHALL / SHALL NOT spec.*
