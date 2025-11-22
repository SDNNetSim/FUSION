# Debug Progress - Part 14: Partial Grooming Fixes

## Session Overview
Fixed two critical bugs in v6's partial grooming logic that were causing mismatches with v5. Both fixes are now in place, but test failures persist with different characteristics.

## Bugs Fixed

### Bug #1: Restricted Modulation List for Partial Grooming
**Location:** `fusion/core/sdn_controller.py` lines 985-1010

**Problem:**
When a request was partially groomed, v6 only provided `['64-QAM']` instead of the full modulation list.

**Root Cause:**
```python
# OLD CODE (lines 997-999)
force_mod_format = list(mod_formats_dict.keys())[0]  # Only first modulation!
```

For partially groomed requests, code took only the first (best) modulation from sorted list, then `_setup_routing` created `formats_matrix = [[force_mod_format]]`.

**Consequence:**
- v6 tried only 64-QAM, which failed SNR check
- Triggered segment_slicing retry with dynamic slicing
- Dynamic slicing selected 8-QAM with bw=500 (from bw_mapping) instead of 400

**Fix:**
Override `modulation_formats_matrix` AFTER `_setup_routing` to provide full list:
```python
# After routing setup, override for partially groomed requests
if getattr(self.sdn_props, "was_partially_groomed", False):
    full_mod_list = list(mod_formats_dict.keys())
    self.route_obj.route_props.modulation_formats_matrix = [full_mod_list]
```

**Impact:** All 86 partially groomed requests now get full modulation list and avoid dynamic slicing retry.

### Bug #2: Wrong Lightpath Bandwidth for Partial Grooming
**Location:** `fusion/core/sdn_controller.py` lines 855-874

**Problem:**
v6 used `remaining_bw` for lightpath capacity instead of original request bandwidth.

**Example (req_id=86):**
- Original request: 400 Gbps
- Groomed: 200 Gbps
- Remaining: 200 Gbps
- v5: Created lightpath with capacity=400, allocated 200, left 200 remaining
- v6: Created lightpath with capacity=200, allocated 200, left 0 remaining

**Root Cause:**
```python
# OLD CODE
if self.sdn_props.was_partially_groomed:
    allocate_bandwidth = str(self.sdn_props.remaining_bw)  # Wrong!
```

**Consequence:**
- Zero remaining capacity prevented future grooming
- Cascading differences in iteration 1 and beyond

**Fix:**
Split lightpath capacity from statistics bandwidth:
```python
# Lightpath capacity: Always use original request bandwidth (matches v5)
lightpath_capacity = self.sdn_props.bandwidth

# Statistics: Use remaining_bw for partial grooming
if self.sdn_props.was_partially_groomed:
    stats_bandwidth = str(self.sdn_props.remaining_bw)
else:
    stats_bandwidth = self.sdn_props.bandwidth

self.spectrum_obj.spectrum_props.lightpath_bandwidth = lightpath_capacity
self._update_request_statistics(bandwidth=stats_bandwidth)
```

**Impact:** Partially groomed requests now create lightpaths with correct capacity and remaining bandwidth.

## Verification

### req_id=86 Comparison (Iteration 0)

**v5:**
```
[V5-DEBUG-LP-CREATED] req_id=86, lp_id=97, lp_bw=400, remaining_bw=200
[CMP-ALLOC] req_id=86 lp_id=97 mod=8-QAM
```

**v6 (after fixes):**
```
[DEBUG-LP-CREATED] req_id=86, lp_id=97, lp_bw=400, remaining_bw=200.00
[CMP-ALLOC] req_id=86 lp_id=97 mod=8-QAM
```

✅ **Perfect match!**

### Statistics
- ✅ 86 partially groomed requests fixed
- ✅ All get full modulation lists
- ✅ None go through dynamic slicing incorrectly
- ✅ First 20 allocations identical between v5 and v6

## Remaining Issues

### Test Failures Persist
```
FAIL: iter_stats.1.sim_block_list expected [0.025, 0.025] got [0.05, 0.05]
FAIL: iter_stats.1.demand_realization_ratio.800.num_full_served expected 25 got 21
FAIL: iter_stats.1.mods_used_dict.32-QAM.c expected 38 got 40
```

### Allocation Count Mismatch
- v5: 806 total allocations (CMP-ALLOC lines)
- v6: 832 total allocations (+26)

Despite higher allocations, v6 has:
- **Higher blocking** (0.05 vs 0.025 in iteration 1)
- **Fewer 800 Gbps requests served** (21 vs 25)
- Different modulation distribution

### Request ID Discrepancy
- Both process up to req_id=200
- v5 missing: req_id=152, 166
- These requests exist in v6 but were blocked (no CMP-ALLOC)
- Suggests possible request generation difference OR v5.txt/v6.txt from different runs

## Open Questions

1. **Why higher blocking with more allocations?**
   - More allocations might be slices that don't fully serve requests
   - Or allocations that get released/fail differently

2. **Are v5.txt and v6.txt from same configuration?**
   - req_id=152, 166 missing in v5 is suspicious
   - Need to verify both use same seed, erlang load, config

3. **Is there another grooming-related bug?**
   - Fixes address partial grooming AFTER first grooming
   - But initial grooming candidate selection might differ
   - Or bandwidth tracking/utilization might have issues

## Next Steps

1. **Run fresh comparison test** with fixes in place
2. **Verify v5.txt and v6.txt are from same config/seed**
3. **If still failing:**
   - Compare grooming decisions for same request between v5 and v6
   - Check bandwidth tracking and utilization updates
   - Investigate why v6 has more allocations but higher blocking

## Files Modified
- `fusion/core/sdn_controller.py`: Lines 985-1010 (modulation list fix), lines 855-874 (bandwidth fix)
