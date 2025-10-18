# Offline + Conservative RL for Survivable EON Routing — v1 Plan, Justifications, and Experiment Slice

**Thesis:** Train **offline** from strong heuristic behavior (KSP‑FF, 1+1 + restoration) in a **domain‑randomized** EON simulator; use **conservative offline RL** with a **safety mask + heuristic fallback**; stress with **single, SRLG, and geo failures**; report **BP, recovery time, fragmentation, runtime, and seed variance** with 95% CIs. The angle is **robustness & stability** under disasters, not chasing tiny BP wins.

---

## Why RL at all (with a baseline fallback)? — deployment‑credible reasons

1. **Adaptive restoration under abnormal conditions.** Heuristics are static; RL **conditions on current load + failure footprint + fragmentation** to pick a *different* K‑path when it helps. We will prove value as **lower BP during/after failure windows** or **faster BP recovery to baseline** vs KSP‑FF/1+1.
2. **Safety without risk.** Safety mask + fallback means RL never does worse than the baseline when it’s unsure. Upside is “free” in deployment risk terms.
3. **Policy portability & operator knobs.** One reward can encode trade‑offs (BP vs fragmentation vs latency). Heuristics require rule surgery per change.

If we show **(a)** non‑worse tail risk and **(b)** some recovery/BP gains with tight CIs, that passes the “why bother?” test.

---

## SDN controller is emulated (no built‑in latency): how we still measure **recovery time**

Your controller is emulated and does not simulate control‑plane internals. We therefore **instrument** recovery time via explicit parameters:

- **Protection switchover (1+1):** assume a fixed **50 ms** switchover budget (typical sub‑50–100 ms protection behavior in carrier networks).
- **Controller‑based restoration:** assume a **configurable compute + signaling delay** — default **100 ms**, with a sensitivity sweep **[50, 200] ms**.
- **Measured recovery time per failure** = time from failure occurrence to the moment when all restorable connections are back in service (either via protection or restoration). We will report **mean** and **95th percentile** per scenario.

Because SDN is emulated, these are **parameters**, not micro‑benchmarks; we evaluate **policy quality** under a realistic time budget.

---

## Failure schemes (kept tight, still revealing)

- **F0:** No failure (sanity & calibration).
- **F1:** Single link cut (uniform random link & time).
- **F3:** **SRLG(2)** event (two links in the same conduit fail together).
- **F4:** **Geo‑cluster** (radius = 2 hops) — a moderate regional disaster once per run.

We keep **F1/F3/F4** in the paper (plus F0 for calibration). This is enough to expose brittleness and the value of conditioning without scope creep.

---

## What “Conservative Offline RL” means (with links) — and why it’s right for v1

**Problem:** Offline logs are heuristic‑heavy. Naïve off‑policy RL (e.g., vanilla DQN) **hallucinates** good values for actions the dataset never took → catastrophic picks online (**extrapolation error**).  
**Fix:** **Conservative offline RL** **stays near the data support** unless evidence says otherwise.

- **IQL — Implicit Q‑Learning** (recommended for v1): improves policy **only where observed advantage exists**; simple and stable for offline→small online fine‑tune.  
  Paper: https://arxiv.org/abs/2110.06169 · Code (official): https://github.com/ikostrikov/implicit_q_learning
- **CQL — Conservative Q‑Learning**: explicitly **penalizes Q** for out‑of‑distribution actions; very safe, slightly more tuning.  
  Paper: https://arxiv.org/abs/2006.04779 · Project: https://sites.google.com/view/conservative-q-learning

**BC — Behavior Cloning** is **not RL**; it’s supervised imitation of heuristic actions. We use BC as a **stable warm‑start** for the policy.  
Intro (concise): https://bair.berkeley.edu/blog/2017/12/20/imitation/

**v1 Recommendation:** **BC → IQL** offline, then **tiny safe online fine‑tune** (10–50k steps) in sim with **action masking + heuristic fallback**. This balances **stability, speed, and novelty** (offline RL + survivability is underexplored in optical).  
Surveys/tutorials on offline RL for context:  
- https://arxiv.org/abs/2005.01643 (Levine et al., 2020)  
- https://arxiv.org/abs/2303.01385 (2023 overview)

---

## “Other RL areas” — include in v1 or not? (justified)

| Area | Include v1? | Why / Why Not (justification) |
|---|---|---|
| **Deep RL (standard DQN/PPO)** | **Yes (as heads/variants)** | Use **Distributional DQN (C51/QR‑DQN)** to reduce variance/capture tails; **Recurrent PPO** as a comparator if time allows (LSTM helps during failure transients). C51: https://arxiv.org/abs/1707.06887 · QR‑DQN: https://arxiv.org/abs/1710.10044 · PPO: https://arxiv.org/abs/1707.06347 · SB3 PPO docs: https://stable-baselines3.readthedocs.io/en/stable/modules/ppo.html |
| **Transfer learning** | **Yes (core goal)** | Offline pretrain on randomized failures; **zero‑shot** to new disaster distribution; **tiny online safe fine‑tune**. This *is* the main story. |
| **Goal‑conditioned RL** | **Lightweight yes** | Add one **context bit** `is_disaster` (or severity bucket). Cheap, improves transfer; no heavy goal architectures. |
| **Hierarchical RL (options)** | **No (v1 overkill)** | Options boundaries/termination & credit assignment complicate training; not needed if a context bit is available. |
| **Meta‑RL** | **No (v1 overkill)** | Sample‑hungry; complex to evaluate credibly in 3 pages; little payoff vs tiny safe fine‑tune. |
| **Multi‑agent RL** | **No (v1 overkill)** | Coordination becomes the problem; outside a short‑paper sprint. Focus = single‑agent K‑path choice. |
| **Inverse RL (IRL)** | **No** | Brittle, slow, and unnecessary; we have a clear shaped reward (accept/block + frag penalty). |
| **Offline IL only (BC/DAgger)** | **BC yes; DAgger optional** | **BC** warm‑starts; **DAgger** needs online expert queries (we can query heuristic, but IQL already provides safe improvement). DAgger: https://proceedings.mlr.press/v15/ross11a/ross11a.pdf |

**Action masking in optical RL:** good reference from Tanaka & Shimoda (JOCN 2023): https://opg.optica.org/jocn/fulltext.cfm?uri=jocn-15-12-1019

**Distributional RL extra reference (IQN):** https://arxiv.org/abs/1806.06923

**Recurrent PPO (SB3 contrib):** https://sb3-contrib.readthedocs.io/en/master/modules/recurrent_ppo.html

**Domain randomization / sim‑to‑real:**  
- https://arxiv.org/abs/1703.06907 (Tobin et al., 2017)  
- https://arxiv.org/abs/1910.07113 (Overview)  
- https://arxiv.org/abs/1803.11329 (Randomization in dynamics)

---

## Detailed simulation setup (with your constraints)

### Network & spectrum
- **EON, flex‑grid, C+L** (start with C only for core results; add L as sensitivity if space permits).
- **Topologies:** **NSFNET (14)** and **COST239 (11)** for comparability.

### Traffic
- **Arrivals:** Poisson (exponential inter‑arrival).  
- **Holding times:** Exponential.  
- **Demand size:** 1–3 slots (uniform) to inject heterogeneity.  
- **Offered loads:** tuned so KSP‑FF yields BP ≈ **{~1%, ~4–6%, ~10–12%}**.

### Failures (one per run)
- **F0** none; **F1** single link; **F3** SRLG(2 links); **F4** geo radius = 2 hops.  
- **Failure time:** Uniform in middle third of run.  
- **Repair time:** Fixed window (e.g., 1000 arrivals) → restored. (Sensitivity: exponential.)

### SDN timing (explicit, since controller is emulated)
- **Protection switchover (1+1):** 50 ms.  
- **Restoration delay (controller compute + signaling):** default 100 ms; sweep [50, 200] ms.

### Algorithms
- **KSP‑FF** baseline, K ∈ {3,4,5}, **hops‑ordered** (strong baseline).  
- **1+1 disjoint + restoration** (fast switchover, then controller recompute).  
- **RL (ours):** **BC → IQL** (optional **Distributional head**), **action masking**, **goal bit** `is_disaster`. Spectrum = **First‑Fit**; action = **choose among K candidate paths**.

### Metrics
- **Blocking Probability (BP)** overall **and** in a **failure window** `[t_fail, t_fail+Δ]` (Δ chosen so both methods stabilize, e.g., 1000 arrivals).  
- **Bandwidth blocking**, **recovery time** (mean, 95th percentile), **fragmentation proxy** (e.g., path‑level contiguity ratio), **runtime/decision time**, **seed variance** (≥5 seeds, 95% CIs).

---

## Example state vectors (concrete)

**Global/request features**
```
src_id (one‑hot or embedding)
dst_id (one‑hot or embedding)
slots_needed ∈ {1,2,3}
est_remaining_time (scalar or bucketed)
is_disaster ∈ {0,1}       # context bit (goal‑conditioned lite)
```

**Per‑candidate path i ∈ {1..K}**
```
path_hops_i
min_residual_slots_i            # min along path
frag_indicator_i                # e.g., 1 − (largest_contig_block / total_free_slots) on path
failure_mask_i ∈ {0,1}         # 1 if any link on path failed
dist_to_disaster_centroid_i     # hops or km (0 if none)
```

**Action:** categorical over **K** paths.  
**Safety mask:** drop any action with `failure_mask_i = 1` or `min_residual_slots_i < slots_needed`. If all masked → **fallback** (KSP‑FF or 1+1 path).

---

## Offline dataset generation (realistic without real logs)

- **Domain randomization grid:**  
  - Load ∈ {low, med, high}  
  - Failure ∈ {F0, F1, F3, F4} with mix ≈ **50/20/15/15%**  
  - Restoration latency ∈ {50, 100, 200 ms}  
  - Failure time ∼ Uniform(mid‑run)
- **Behavior sources (to widen support):**  
  - **KSP‑FF** (primary)  
  - **1+1 + restoration** (secondary)  
  - **Second‑best K path** chosen **10–15%** of the time (epsilon‑mix) to modestly expand action coverage without trashing quality
- **Logged tuples:** `(s, a, r, s′, action_mask, backup_available_flag)`  
- **Feature noise:** ±5–10% jitter on residual capacity / frag to mimic telemetry errors  
- **Size:** **2–3M transitions per topology**

**Why acceptable without real data:** we **acknowledge** the limitation and **defend** with (i) broad randomization (sim‑to‑real literature), (ii) **conservative** RL (IQL/CQL) to avoid OOD actions, and (iii) **safety fallback**.

---

## RL pipeline (step‑by‑step)

1) **Behavior Cloning (BC) warm‑start**  
   - Model: 2×128 MLP on the state; softmax over K.  
   - Train to **>98% top‑2 accuracy** on a held‑out disaster slice. Fast and stable.  
   - Intro to BC: https://bair.berkeley.edu/blog/2017/12/20/imitation/

2) **Conservative Offline RL fine‑tune**  
   - **IQL (recommended):** advantage‑weighted actor; value via expectile regression (τ≈0.7–0.9).  
     Paper: https://arxiv.org/abs/2110.06169 · Code: https://github.com/ikostrikov/implicit_q_learning  
   - **Alt:** **CQL:** Q‑lower‑bounding regularizer (more conservative; slightly more tuning).  
     Paper: https://arxiv.org/abs/2006.04779 · Project: https://sites.google.com/view/conservative-q-learning  
   - Steps: **200–300k** gradient steps; early‑stop on an off‑policy eval proxy (e.g., simple importance‑weighted estimate).

3) **Optional Distributional head**  
   - Swap critic to **QR‑DQN** or **C51** to model value distributions and improve tail behavior; same discrete action space.  
     C51: https://arxiv.org/abs/1707.06887 · QR‑DQN: https://arxiv.org/abs/1710.10044 · IQN: https://arxiv.org/abs/1806.06923

4) **Online safe fine‑tune (in sim)**  
   - **10–50k** env steps, **action masking** always on, **fallback** if all actions masked.  
   - Purpose: adapt to the exact test failure distribution/latency, not to learn from scratch.

5) **Evaluation**  
   - **≥5 seeds** per (topology × load × failure), report **95% CIs** for **BP** (overall + failure window), **recovery time**, **frag proxy**, **runtime**, **std(BP)**.

6) **Interpretability (lightweight, paper‑fit)**  
   - **Distill** the final policy to a **depth‑3 decision tree** on a holdout set; report **fidelity ≥85%** and list the **top‑3 features**.  
   - **Monotonic probe:** increase `min_residual_slots` on chosen path → policy should not disprefer it; report % probes passing.

---

## Updated, falsifiable hypotheses (v1)

- **H1 (Baseline parity, nominal):** BC→IQL(+mask) **matches KSP‑FF BP within ≤1.0% abs** on NSFNET/COST239 at medium load (no failure), across ≥5 seeds.
- **H2 (During failures, stability):** For **F1** and **F3**, **95th‑percentile BP in the failure window** (with modeled latencies) for RL **≤ baseline + 2.0% abs**, and **mean recovery time** within **±10%** of baseline.
- **H3 (Disaster transfer):** Zero‑shot on **F4 (geo r=2)** degrades **≤2.0% abs BP** vs baseline; **≤1.0%** after **≤10k** safe fine‑tune steps.
- **H4 (Variance):** Across ≥5 seeds, RL **std(BP)** **≤** heuristic std(BP) **+ 0.3% abs** in all scenarios; **no variance blow‑ups** during failure windows.
- **H5 (Interpretability‑lite):** A depth‑3 distilled tree achieves **≥85% policy fidelity**; **top‑3 features** account for **≥70%** of permutation importance; monotonic check passes **≥95%**.

---

## Minimal OFC experiment slice (with recovery‑time modeling)

- **Topologies:** NSFNET(14), COST239(11).  
- **Traffic:** online, exp inter‑arrival/holding; loads tuned to BP targets above.  
- **Failures:** F0, F1, F3, F4 (one event/run).  
- **Latency modeling:** Protection **50 ms**; restoration **100 ms** default (sweep **[50, 200] ms**).  
- **Methods:**  
  - KSP‑FF (K ∈ {3,4,5}, hops‑ordered)  
  - 1+1 disjoint + restoration  
  - **BC → IQL** (+ optional Distributional head), **mask**, **fallback**, **goal bit**  
- **Reporting:** BP (overall + failure window), bandwidth blocking, **recovery time** (mean & P95), frag proxy, runtime/decision time, **seed variance** (95% CI).

**Figures (fits 3 pages):**  
1) **Mini‑schematic**: F1, F3, F4.  
2) **Main plot**: BP vs load (NSFNET), KSP‑FF vs RL (with 95% CI).  
3) **Table**: BP + P95 BP in failure window + recovery time + frag proxy + runtime + std across seeds.  
*(If space is tight, fuse #2 with seed error bars.)*

---

## Pros & cons (candid)

**Pros**  
- Safety story (mask + fallback) → **no catastrophic regressions**.  
- **Novel for optical**: **offline (IQL/CQL) + survivability + transfer**.  
- Practical and runnable in weeks, not months.

**Cons / risks**  
- If logs are too narrow, even IQL/CQL can be conservative but mediocre — **domain randomization is non‑negotiable**.  
- If we can’t show **tail‑risk bounds** (P95 in‑window) and **CIs**, reviewers won’t buy “stability.”  
- Distributional head adds one more knob — keep optional if time is tight.

---

## Appendix A — Fast link list (papers, code, docs)

- **Behavior Cloning (overview):** https://bair.berkeley.edu/blog/2017/12/20/imitation/  
- **IQL (paper):** https://arxiv.org/abs/2110.06169 · **IQL code:** https://github.com/ikostrikov/implicit_q_learning  
- **CQL (paper):** https://arxiv.org/abs/2006.04779 · **CQL project:** https://sites.google.com/view/conservative-q-learning  
- **Distributional RL:** C51 https://arxiv.org/abs/1707.06887 · QR‑DQN https://arxiv.org/abs/1710.10044 · IQN https://arxiv.org/abs/1806.06923  
- **PPO (paper):** https://arxiv.org/abs/1707.06347 · **SB3 PPO docs:** https://stable-baselines3.readthedocs.io/en/stable/modules/ppo.html · **SB3 Recurrent PPO:** https://sb3-contrib.readthedocs.io/en/master/modules/recurrent_ppo.html  
- **Action masking in optical RL (Tanaka & Shimoda, JOCN 2023):** https://opg.optica.org/jocn/fulltext.cfm?uri=jocn-15-12-1019  
- **Offline RL surveys/tutorials:** https://arxiv.org/abs/2005.01643 · https://arxiv.org/abs/2303.01385  
- **Domain randomization / sim‑to‑real:** https://arxiv.org/abs/1703.06907 · https://arxiv.org/abs/1910.07113 · https://arxiv.org/abs/1803.11329

---

## Final sanity checks & decisions to lock

- **IQL vs CQL:** pick **IQL** for v1 (simpler, stable); keep CQL as cited alternative.  
- **Include Recurrent PPO?** Optional comparator; skip if time is tight.  
- **Use the `is_disaster` goal bit?** **Yes** — cheap leverage.  
- **Latency defaults:** 50 ms (protection), 100 ms (restoration) with sweep → realistic without simulating controller internals.

