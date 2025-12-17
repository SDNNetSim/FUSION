# V5 vs Legacy Debug Progress

## Overview for New Contributors

This document tracks debugging efforts to make V5 (new orchestrator architecture) match
Legacy (sdn_controller) behavior exactly. The test suite compares both implementations
request-by-request to ensure identical outcomes.

## Problem Statement

Test `spain_C_fixed_grooming_snr_recheck` had 364 metric mismatches between V5 and Legacy.
The goal is to make V5 produce identical results to Legacy.

## Test Configuration
- `fixed_grid = True` (each slot = 1 spectrum unit)
- `dynamic_lps = True` (dynamic lightpath slicing enabled)
- `is_grooming_enabled = True` (can reuse existing lightpath capacity)
- `snr_recheck = True` (validate SNR after allocation)

## Key Discovery: Request 5 Was a Red Herring

Initial investigation showed request 5 diverging:
```
V5:     req=5 | SLICED | spec=1-2/16-QAM
Legacy: req=5 | SLICED | spec=2-2/16-QAM
```

We added extensive debugging and discovered this was actually a **reporting difference**,
not a behavioral difference. Both systems allocate the SAME slots:
- Both select slot 1 for LP 5, then slot 2 for LP 6
- V5 reports the FIRST slice's spectrum in output
- Legacy reports the LAST slice's spectrum in output

The actual slot allocations were identical.

## Real Root Cause: Request 40 Partial Grooming Bug

The actual divergence started at request 40:
```
V5:     req=40 | PARTIAL_GROOM | lps_created=[50, 51] | lps_groomed=[10]
Legacy: req=40 | PARTIAL_GROOM | lps_created=[50]     | lps_groomed=[10]
```

V5 created 2 new LPs while Legacy created only 1.

### Debugging Trail

Added debug prints showing bandwidth values during dynamic slicing:

```
[V5_DYN_SLICE]     req=40 iter=1 remaining_bw=800 actual_slice_bw=500
[LEGACY_DYN_SLICE] req=40 iter=1 remaining_bw=400 bandwidth=500
```

The remaining_bw values were different:
- V5 started with 800 (original request bandwidth)
- Legacy started with 400 (actual remaining after grooming)

### The Bug

In `orchestrator.py`, for partial grooming cases:

```python
# Line 175: For spectrum allocation, use original BW (correct for slots_needed)
spectrum_bw = request.bandwidth_gbps if was_partially_groomed else remaining_bw

# Stage 4 was passing spectrum_bw (800) to dynamic slicing loop
result = self._try_allocate_on_path(..., spectrum_bw, ...)  # 800, not 400!
```

The `spectrum_bw` is correct for spectrum/modulation calculations, but the dynamic
slicing LOOP should iterate based on actual `remaining_bw` (how much bandwidth still
needs to be served), not the original request bandwidth.

Legacy correctly uses `sdn_props.remaining_bw` (400) for its slicing loop iteration.

## The Fix

Added `slicing_target_bw` parameter to separate the two concerns:

```python
# orchestrator.py Stage 4
slicing_target_bw = remaining_bw if was_partially_groomed else spectrum_bw

result = self._try_allocate_on_path(
    ..., spectrum_bw, ...,  # For spectrum allocation
    slicing_target_bw=slicing_target_bw,  # For slicing loop iteration
)

# _try_allocate_on_path now uses slicing_target_bw for _allocate_dynamic_slices
actual_remaining = slicing_target_bw if slicing_target_bw is not None else bandwidth_gbps
result = self._allocate_dynamic_slices(..., remaining_bw=actual_remaining, ...)
```

Now request 40 correctly:
- Uses remaining_bw=400 for slicing loop
- Creates 1 LP (since 500 Gbps capacity > 400 Gbps needed)
- Matches Legacy behavior

## Files Modified

| File | Changes |
|------|---------|
| `fusion/core/orchestrator.py` | Added `slicing_target_bw` parameter, fixed Stage 4 |
| `fusion/core/grooming.py` | Debug prints (to clean up) |
| `fusion/core/adapters/grooming_adapter.py` | Debug prints (to clean up) |
| `fusion/modules/spectrum/light_path_slicing.py` | Debug prints (to clean up) |
| `fusion/core/spectrum_assignment.py` | Debug prints (to clean up) |
| `fusion/modules/spectrum/utils.py` | Debug prints (to clean up) |
| `fusion/pipelines/slicing_pipeline.py` | Debug prints (to clean up) |

## Next Steps

1. **Run test to verify fix**: `python tests/run_comparison.py --test spain_C_fixed_grooming_snr_recheck`
2. **Clean up debug prints**: Remove all `[V5_DBG]`, `[LEGACY_DBG]`, `[GROOM_CORE]`, etc.
3. **Run full test suite**: Ensure no regressions in other tests
4. **Document the fix**: Update any relevant architecture docs

## Debug Commands

```bash
# Run the failing test
python tests/run_comparison.py --test spain_C_fixed_grooming_snr_recheck

# Check specific request output
grep -E "req=40" new_debug.txt legacy_debug.txt

# Compare LP counts at specific request
grep -E "req=45.*LP" new_debug.txt | head -20

# Run all comparison tests
python tests/run_comparison.py
```

## Architecture Reference

### V5 Orchestrator Stages
```
Stage 1: Grooming (GroomingAdapter) - reuse existing LP capacity
Stage 2: Routing (RoutingAdapter) - find k-shortest paths
Stage 3: Standard allocation on all k-paths (no slicing)
Stage 4: Dynamic LP slicing (if dynamic_lps=True) <- BUG WAS HERE
Stage 5: Segment slicing fallback (StandardSlicingPipeline)
Stage 6: Block request
```

### Legacy SDN Controller Flow
```
1. Grooming check
2. Standard allocation attempt
3. If fails + slicing enabled: handle_dynamic_slicing_direct()
   -> Iterates using sdn_props.remaining_bw (correct!)
```

## Lessons Learned

1. **Reported values can mislead**: Request 5's different `spec=` values looked like
   divergence but were just different reporting (first vs last slice).

2. **Add debug at multiple levels**: We needed debug in grooming core, adapter,
   orchestrator, AND slicing to trace the value transformation.

3. **Bandwidth has multiple meanings**: `spectrum_bw` (for slot calculation) vs
   `remaining_bw` (for iteration) vs `achieved_bw` (per-LP capacity) - conflating
   these caused the bug.

---

## Current Investigation: Request 46 Divergence

### Problem
After fixing request 40, a new divergence appears at request 46:
- **V5:** `ALLOCATED | lps_created=[57]` - direct Stage 3 allocation
- **Legacy:** `SLICED | lps_created=[59]` - falls back to segment_slicing

LP IDs 57 and 58 are "consumed" in Legacy but not in V5.

### Observations

Legacy request 46 debug shows:
```
[LEGACY_DBG] req=46 STANDARD path_idx=0 is_free=True start=5 end=5 mod=32-QAM
[LEGACY_DBG] req=46 STANDARD path_idx=1 is_free=True start=0 end=0 mod=32-QAM
[LEGACY_DBG] req=46 STANDARD path_idx=2 is_free=False start=5 end=5 mod=BPSK
[LEGACY_DBG] req=46 _process_single_path SLICING path_idx=0 segment_slicing=True force_slicing=True
```

Paths 0 and 1 have `is_free=True` but allocation still fails on first pass.

### Hypothesis

Legacy's `_finalize_successful_allocation` does SNR recheck AFTER LP ID is incremented:
1. First pass, Path 0: `is_free=True` → LP 57 created → SNR fails → rolled back
2. First pass, Path 1: `is_free=True` → LP 58 created → SNR fails → rolled back
3. First pass, Path 2: `is_free=False` → no LP created
4. Second pass: segment_slicing=True → LP 59 created → succeeds

V5's Stage 3 doesn't have the same SNR failure, so it allocates LP 57 directly.

### Debug Added

Added SNR recheck debug for request 46 in:
- `sdn_controller.py:602-603` (Legacy)
- `orchestrator.py:615-616` (V5)

### Key Finding: SNR Recheck Returns Different Results

Same LP 57, same spectrum slot 5, but different SNR recheck outcomes:

**Legacy LP 57:**
```
recheck_enable=False violations=[
  (14, 15.19, 13.24),  # LP 14 degraded
  (37, 12.98, 10.84),  # LP 37 degraded
  (44, 12.97, 10.84),  # LP 44 degraded
  (45, 12.97, 10.84),  # LP 45 degraded
  (46, 15.18, 13.24)   # LP 46 degraded
]
```

**V5 LP 57:**
```
all_pass=True degraded=()
```

Legacy finds 5 existing LPs would be degraded by LP 57's allocation.
V5 finds no degradation.

### Root Cause Investigation

V5's `SNRAdapter.recheck_affected()` creates a new `SnrMeasurements` instance
with proxy objects built from `NetworkState`. The proxy includes:
- `network_spectrum_dict` from NetworkState
- `lightpath_status_dict` from NetworkState

Legacy's `_check_snr_after_allocation()` uses its internal `sdn_props` which
has the full simulation state including all lightpath tracking lists.

The difference is likely in how the existing lightpaths are enumerated for
SNR impact checking. The `SDNPropsProxyForSNR` may not be providing the same
view of existing lightpaths as Legacy's actual `sdn_props`.

### Further Investigation

Added debug to `snr_recheck_after_allocation` to compare:
- `lightpath_status_dict` counts (same: 57 in both)
- `all_active_lps` counts (same: 56 in both)
- `overlapping_lps` (same: [14, 37, 44, 45, 46, 51])

Both find the same overlapping LPs, but `evaluate_lp()` returns different results:

**Legacy LP 14 evaluation:**
```
resp=False observed_snr=15.19 required_snr=13.24
```

**V5 LP 14 evaluation:**
```
resp=16-QAM observed_snr=15.19 required_snr=13.24
```

Same SNR values, but different return types! V5 returns a truthy string instead of `False`.

### Root Cause Found

The SNR adapter's `SpectrumPropsProxyForSNR` was hardcoding `slicing_flag=True`.

When `slicing_flag=True`, `check_gsnr()` skips the bandwidth validation check and
returns a modulation string (truthy) instead of `False` (boolean).

**Legacy behavior:**
- Standard allocation: `slicing_flag=False` → `check_gsnr` returns `False` on failure
- Slicing allocation: `slicing_flag=True` → `check_gsnr` returns modulation string

**V5 bug:**
- Always set `slicing_flag=True` → `check_gsnr` always returns truthy string

### Fix Applied

1. Added `slicing_flag` parameter to `SNRAdapter.recheck_affected()` (default `False`)
2. Updated `_allocate_and_validate()` to accept and pass `slicing_flag`
3. Standard allocations (Stage 3) use `slicing_flag=False` (default)
4. Dynamic slicing allocations (`_allocate_dynamic_slices`) use `slicing_flag=True`
5. **Additional fix:** When `use_dynamic_slicing=True` (Stage 4) AND spectrum can
   satisfy full request, pass `slicing_flag=use_dynamic_slicing` to match Legacy's
   segment_slicing behavior

Files modified:
- `fusion/core/adapters/snr_adapter.py`: Added `slicing_flag` param, use it in proxy
- `fusion/core/orchestrator.py`: Pass `slicing_flag` to SNR recheck in multiple places

### Next Steps

1. Run test to verify fix
2. Clean up debug prints if successful
3. Run full test suite

---

## Bug 3: demand_realization_ratio > 1.0 (Fixed)

### Problem

After the SNR recheck fix, tests showed `demand_realization_ratio` values exceeding 1.0:
```
demand_realization_ratio.overall.max expected 1.0 got 1.8571428571428572
demand_realization_ratio.600.max expected 1.0 got 1.6666666666666667
```

A ratio > 1.0 means V5 reported serving MORE bandwidth than was requested.

### Root Cause

For partial grooming, V5 uses two bandwidth values:
- `spectrum_bw` = original request bandwidth (for spectrum/modulation calculation)
- `remaining_bw` = actual bandwidth still needed (after grooming)

The bug: `_allocate_and_validate()` was using `spectrum_bw` for `total_bandwidth_allocated_gbps`
instead of `remaining_bw`.

**Example:**
- Request for 700 Gbps
- Grooming serves 300 Gbps, remaining_bw = 400
- `spectrum_bw` = 700 (for spectrum calculation)
- `_allocate_and_validate()` returns `total_bandwidth_allocated_gbps = 700` (WRONG!)
- `_combine_results()`: `total_bw = 300 + 700 = 1000` (should be 300 + 400 = 700)
- `demand_realization_ratio = 1000/700 = 1.43`

### Fix Applied

Added `actual_bw_to_allocate` parameter to separate spectrum calculation bandwidth from
allocation tracking bandwidth:

**Stage 3, 4, and 5 in `handle_arrival()`:**
```python
# For partial grooming, actual_bw_to_allocate = remaining_bw (what we need to serve)
# but spectrum_bw = original request (for spectrum calculation)
actual_bw_to_allocate = remaining_bw if was_partially_groomed else spectrum_bw

result = self._try_allocate_on_path(
    ..., spectrum_bw, ...,  # For spectrum calculation
    actual_bw_to_allocate=actual_bw_to_allocate,  # For stats tracking
)
```

**In `_try_allocate_on_path()`:**
```python
# Use actual_bw_to_allocate for allocation tracking (defaults to bandwidth_gbps)
bw_to_allocate = actual_bw_to_allocate if actual_bw_to_allocate is not None else bandwidth_gbps
alloc_result = self._allocate_and_validate(..., bw_to_allocate, ...)
```

### Files Modified

| File | Changes |
|------|---------|
| `fusion/core/orchestrator.py` | Added `actual_bw_to_allocate` param to Stages 3, 4, 5 |
| `fusion/core/orchestrator.py` | Updated `_try_allocate_on_path()` signature and logic |
| `fusion/core/orchestrator.py` | Updated slicing fallback to use correct bandwidth |

### Result

- `spain_C_fixed_grooming` now **PASSES** (was failing with demand_realization_ratio > 1.0)
- `spain_C_fixed_grooming_snr_recheck` has only minor SNR stat differences remaining

---

## Current Status: Minor SNR Stat Differences

### Remaining Errors (spain_C_fixed_grooming_snr_recheck)

```
mods_used_dict.16-QAM.snr.overall.max expected 18.1 got 15.99
mods_used_dict.32-QAM.snr.overall.max expected 19.95 got 18.21
snr_mean expected 15.29 got 15.23
```

These are small differences in SNR statistics:
- Mean SNR is very close (15.29 vs 15.23 = 0.06 difference)
- Max SNR values differ more significantly

### Hypothesis

V5's SNR recheck with the `slicing_flag` fix causes slightly different allocation decisions:
- Some high-SNR allocations that Legacy accepts are rejected by V5's stricter recheck
- Or vice versa - different allocation outcomes lead to different SNR samples

Since `spain_C_fixed_grooming` (without snr_recheck) passes perfectly, the difference
is specifically related to SNR recheck behavior.

### Next Steps

1. Investigate if V5's SNR recheck is making different accept/reject decisions
2. Check if `slicing_flag` is being set consistently in all paths
3. Consider if small statistical differences are acceptable
