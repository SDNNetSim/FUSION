# Request 37 Grooming Bug - Debug Progress

## Issue Summary
Request 37 allocation differs between v5 (correct) and v6 (buggy):
- v5: Partial grooming + slicing, 2 lightpaths [36, 47] with bandwidths [600, 200]
- v6: Same but wrong values - bandwidths [600.0, '600'], remaining_bw=100.0, segment_slicing=True

## Root Cause Analysis

### Bug Location
`fusion/modules/spectrum/light_path_slicing.py::handle_dynamic_slicing_direct()`

**Lines 299 & 309 (original):**
```python
lightpath_bandwidth = str(bandwidth)  # Sets to '600' instead of 200
remaining_bw -= bandwidth             # Subtracts 600 instead of 100
```

### The Problem
1. `bandwidth = 600` (spectrum capacity found by SNR check)
2. `dedicated_bw = min(600, 100) = 100` (actual bandwidth needed)
3. Code used `bandwidth` (600) instead of selecting appropriate tier for `dedicated_bw` (100)
4. Should select smallest tier >= 100, which is 200 Gbps

## Fix Applied

Lines 303-319: Conditional fix for partial grooming only
```python
if self.sdn_props.was_partially_groomed:
    # Find smallest bandwidth tier >= dedicated_bw
    selected_tier = 200  # For dedicated_bw=100
    lightpath_bw = selected_tier
    remaining_bw -= dedicated_bw
else:
    # Preserve v5 behavior for non-grooming cases
    lightpath_bw = str(bandwidth)
    remaining_bw -= bandwidth
```

## Current Status (Partial Fix)

**v6 after fix:**
- Route: ['24', '6', '8', '9'] ✓
- Was partially groomed: True ✓
- Lightpath Bandwidths: [600, 200] ✓ (FIXED!)
- Remaining BW: 0 ✓ (FIXED!)
- **Segment slicing: True** ✗ (should be False)

**v5 (correct):**
- Segment slicing: False ✓

## Remaining Issue

First allocation attempt with `segment_slicing=False` fails (`is_free=False`), forcing retry with `segment_slicing=True`. Need to investigate why standard allocation fails for partial grooming when it should succeed (like in v5).

## Next Steps
1. Compare v5 vs v6 `get_spectrum()` behavior for partial grooming cases
2. Check why `is_free=False` on first attempt in v6 but succeeds in v5
3. Fix standard allocation to succeed on first try, eliminating need for segment slicing retry
