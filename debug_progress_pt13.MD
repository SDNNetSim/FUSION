# Debug Progress - Part 13: Root Cause - Dynamic Slicing Retry Logic

## Session Overview
Added instrumentation to both v5 and v6 SNR bandwidth selection logic. Discovered that v6 retries failed paths with dynamic slicing enabled, creating larger capacity lightpaths that persist across iterations.

## Root Cause Identified

### The Key Difference: req_id=86 in Iteration 0

**v5 Behavior:**
```
[V5-SNR-COND] req_id=86, slicing_flag=False, fixed_grid=True, dynamic_lps=True, remaining_bw=200, orig_bw=400
[V5-SNR-STANDARD] req_id=86, NOT entering dynamic modulation selection
[V5-DEBUG-LP-CREATED] req_id=86, lp_id=97, light_id=('14', '24'), lp_bw=400, remaining_bw=200
```
- Grooms 200 onto lp_id=94
- Tries remaining 200 with slicing_flag=False
- Creates lp_id=97 with capacity=**400**
- Never enters dynamic modulation selection

**v6 Behavior:**
```
[V6-SNR-COND] req_id=86, slicing_flag=False, ... remaining_bw=200.0
[V6-GSNR-CALC] req_id=86 mod=64-QAM gsnr_db=12.893295 req_snr=19.010000 decision=FAIL
[V6-PATH-RESULT] req_id=86 path_index=0 success=False
[V6-PATH-TRY] req_id=86 src=14 dst=24 path_index=0 path=['24', '6', '19', '17', '16', '14'] path_len=5
[V6-SNR-COND] req_id=86, slicing_flag=True, ... remaining_bw=200.0
[DEBUG-GSNR-MOD-SELECT] SELECTED mod=8-QAM, bw=500
[V6-SLICING] req_id=86, orig_bw=400, slice_bw=500, dedicated_bw=200.0, mod=8-QAM
[DEBUG-LP-CREATED] req_id=86, lp_id=97, light_id=('14', '24'), lp_bw=500, remaining_bw=300.00
```
- Grooms 200 onto lp_id=94
- Tries remaining 200 with 64-QAM → **SNR FAILS**
- **Retries same path** with slicing_flag=True (segment_slicing retry)
- Dynamic slicing selects 8-QAM with bw=**500** (not 200!)
- Creates lp_id=97 with capacity=**500**, using only 200, leaving **300 remaining**

### The Cascade Effect

**Iteration 0:**
- v6's lp_id=97 has 300 remaining capacity
- v5's lp_id=97 has 200 remaining capacity

**Iteration 1:**
- v5: lp_id=97 gets fully released before iteration 1 starts
- v6: lp_id=97 persists with 300 remaining
- req_id=4 (src=14 dst=24 bw=100) in iteration 1:
  - v5: Creates NEW lightpath (lp_id=97 is gone)
  - v6: **Grooms onto existing lp_id=97** from iteration 0!

This grooming difference cascades through iteration 1, causing:
- Different request blocking patterns
- Different path selections
- Different modulation statistics
- Test failures

## The Mechanism

### Retry Logic (Both v5 and v6 have this)
In `fusion/core/sdn_controller.py` lines 1108-1115:
```python
# Try segment slicing if not already tried
if (
    self.engine_props["max_segments"] > 1
    and self.sdn_props.bandwidth != "25"
    and not segment_slicing
):
    segment_slicing = True
    continue  # Retries all paths with segment_slicing=True
```

### When segment_slicing=True Triggers Dynamic Slicing
1. `segment_slicing=True` is passed to `_process_single_path`
2. This eventually calls `get_spectrum_dynamic_slicing()`
3. Which sets `self.spectrum_props.slicing_flag = True` (line 954)
4. SNR check with `slicing_flag=True` AND `dynamic_lps=True` triggers:

### Dynamic Modulation Selection (snr_measurements.py:835-855)
```python
if self.spectrum_props.slicing_flag and self.engine_props_dict['fixed_grid'] and self.engine_props_dict.get('dynamic_lps', False):
    # Selects modulation based on SNR
    for mod in force_mod_format:
        if gsnr_db >= req_snr_val:
            resp = mod
            bw_resp = bw_mapping[mod]  # e.g., 8-QAM → 500!
            break
```

**The Issue:** Returns `bw_mapping[mod]` (standard capacity for that modulation) instead of actual remaining bandwidth needed.

## Why v5 and v6 Diverge

**The Open Question:** Why does v5's initial attempt (with slicing_flag=False) succeed while v6's fails?

Possible causes:
1. Different modulation selection on first attempt
2. Different SNR calculation or thresholds
3. Different path feasibility checks
4. v5 might succeed with a lower modulation format on first try

**Evidence:** v5 has 4 SNR-COND checks for req_id=86, all with slicing_flag=False, suggesting it tries multiple paths but never needs the segment_slicing retry.

## Files Modified
- `fusion/core/snr_measurements.py`: Added V6-SNR-COND, V6-SNR-STANDARD instrumentation
- `src/snr_measurements.py`: Added V5-SNR-COND, V5-SNR-STANDARD, V5-SNR-DYNAMIC instrumentation

## Potential Fixes

### Option 1: Fix Bandwidth Mapping
In `snr_measurements.py` line 850:
```python
# Current:
bw_resp = bw_mapping[mod]

# Fixed:
bw_resp = min(bw_mapping[mod], remaining_bw)
```

### Option 2: Prevent Retry for Partial Grooming
Skip segment_slicing retry when `was_partially_groomed=True`

### Option 3: Match v5's Initial Behavior
Investigate why v5's first attempt succeeds and make v6 do the same

## Success Metrics
- ✅ Identified exact divergence point (req_id=86, iteration 0)
- ✅ Traced mechanism (segment_slicing retry → dynamic slicing → bw_mapping)
- ✅ Explained cascade effect into iteration 1
- ❌ Root cause of initial path failure difference (v5 vs v6)
- **Next:** Investigate why v5's initial path attempt succeeds while v6's fails
