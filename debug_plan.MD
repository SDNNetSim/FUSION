# Context: FUSION v5 vs v6 grooming mismatch

## Purpose

You are helping debug a **grooming mismatch** between two branches of the same optical network simulator:

- **v6** = `chore/stabilize-v6-expected-results` (current branch, new architecture)
- **v5** = `feature/grooming-new` (older architecture, but grooming logic is considered correct)

The **core logic** of algorithms should be equivalent between v5 and v6, even if:
- File / module layout differs
- Variable names differ
- Function names differ

Your job:  
**Find and fix why grooming results in v6 do not match v5**, without changing the intended behavior from v5.

---

## Test harness

There is a comparison test:

- Script: `run_comparison.py`
- Cached fixtures: `tests/fixtures/expected_results/*`

Behavior:
- It runs simulations for several algorithms on **v6** and compares them against **cached v5 results**.
- The cached fixtures represent the **trusted v5 behavior**.

When the comparison fails, it means v6 has diverged from v5 for that algorithm, given the same random seed and configuration.

---

## What is already trusted / passing

The following algorithms already match between v5 and v6 (comparison passes):

- `baseline_spf_ff`
- `baseline_kspf_ff`
- `epsilon_greedy_bandit`
- `ext_snr_4core_cls_dy-slice`
- `xtar_slicing_pff`

Implication:  
We are already confident that the **core mechanics** are correct and consistent between v5 and v6 for:

- Path selection logic
- Slicing and dynamic slicing behavior
- Reinforcement learning plumbing (at least for epsilon-greedy bandit)
- Spectrum assignment schemes

So the mismatch is **specifically related to grooming behavior**, not the general routing/slicing/SA machinery.

---

## Domain model (minimal)

- A **request** has a unique **request ID** and a bandwidth demand (e.g. 200 Gbps).
- **Slicing**: splitting a request into multiple sub-demands (e.g. 200 → 2×100 or 4×50) for reach / spectrum reasons.
- **Dynamic slicing**: more flexible decomposition (e.g. 200 → 50 + 25 + 25 + 100).
- **Grooming**: aggregating / mapping traffic onto existing or new lightpaths according to grooming rules.

Lightpaths:

- Each allocated segment on the spectrum is a **lightpath** with a **lightpath ID**.
- A single request ID may map to multiple lightpath IDs when sliced or groomed.

Key data structures (names may differ slightly between v5 and v6, but concepts must match):

- `lp_bw_dict`: maps **lightpath IDs** to bandwidth usage / capacity info.
- `req_dict`: maps **request IDs** to their associated metadata (paths, slices, lightpaths, etc.).
- `req_status_dict`: tracks whether each **request** is blocked, partially served, fully served, etc.

When debugging, these three structures are critical for understanding how grooming is applied.

---

## Ground truth and constraints

- **v5 (feature/grooming-new)** is the **ground truth for grooming behavior**.
- **v6** must be aligned to v5’s grooming logic, not vice-versa.
- We are allowed to:
  - Inspect and instrument both branches.
  - Add temporary debug logs and assertions.
- We are **not** allowed to:
  - “Invent” new grooming behavior without evidence.
  - Change the intended algorithm semantics compared to v5.

All reasoning / code changes must be backed by **request-by-request evidence**.

---

## Debugging style I want from you

Assume we can run both v5 and v6 with the same seed, config, and traffic pattern.

Preferred method:

1. **Request-level logging**
   - Instrument both v5 and v6 with consistent debug prints/logging for each **request ID** and related **lightpath IDs**.
   - Logs should be easily diffable (e.g. structured text or JSON per request).

2. For each request (or for a subset where mismatch appears), log at least:
   - `request_id`
   - original bandwidth and any slices (slice sizes)
   - grooming decisions:
     - whether it reused an existing lightpath
     - whether it created a new lightpath
   - mapping: `request_id -> [lightpath_ids]`
   - for each `lightpath_id`:
     - chosen path (node sequence or path ID)
     - modulation format
     - start slot / end slot on spectrum
   - snapshots or summaries of:
     - `lp_bw_dict` entries touched
     - relevant `req_dict` fields
     - `req_status_dict` for that request

3. **Compare v5 vs v6 logs**:
   - If **requests differ** (different sequence, IDs, or bandwidths), that is a red flag. First ensure request generation is identical between branches.
   - Once requests are confirmed identical, compare:
     - grooming decisions
     - spectrum allocations
     - lightpath mappings
     - per-request status (served / blocked / partially served)

4. **Trace mismatches back to code**
   - When a discrepancy is found (e.g. v5 grooms onto an existing lightpath but v6 creates a new one, or uses different slicing), locate:
     - The function(s) responsible for:
       - choosing whether to groom
       - selecting target lightpaths
       - updating `lp_bw_dict`, `req_dict`, and `req_status_dict`
   - Carefully diff the relevant v5 vs v6 logic:
     - Conditions and branching
     - Loops over candidate lightpaths
     - Bandwidth accounting and residual capacity checks
     - Any ordering rules (e.g. first-fit vs best-fit for grooming)

5. **Propose minimal, targeted fixes**
   - Adjust v6 to match v5’s grooming logic exactly for the discovered discrepancy.
   - Avoid large refactors or style-only changes.
   - Keep patches narrowly scoped and explain in comments how they restore v5-equivalent behavior.

6. **Re-run `run_comparison.py`**
   - Confirm that the grooming-related tests now pass (or that the mismatch window has narrowed).
   - If still failing, repeat request-level comparison focusing on remaining discrepancies.

---

## How to interact with me (the user)

When I ask you to debug or change something related to grooming:

1. **Always assume this context is true and up to date** unless I explicitly say otherwise.
2. Ask me for:
   - The exact failing test case(s)
   - Any current logging output from both v5 and v6
   - Pointers to specific functions or files if needed
3. When suggesting instrumentation or changes:
   - Give concrete code snippets (for both branches when helpful).
   - Make logs deterministic and easy to diff.
4. Before proposing a fix, explicitly state:
   - What **v5 does** in the scenario.
   - What **v6 currently does**.
   - Why that difference is grooming-specific and not just noise.

If something is ambiguous in the code, **do not guess silently**.  
Explain the ambiguity and suggest what additional logs or tests are needed to prove the correct behavior relative to v5.

---
