# V5 vs Legacy Debug Progress

## Current Issue: SNR Stat Mismatch

**Test:** `spain_C_fixed_grooming_snr_recheck`

**Errors:**
```
mods_used_dict.16-QAM.snr.overall.max expected 18.1 got 15.99
mods_used_dict.32-QAM.snr.overall.max expected 19.95 got 18.21
snr_mean expected 15.29 got 15.23
```

## Root Cause Found

Legacy grooming code appends SNR values to `snr_list` **before** grooming completes.
These values remain even when `lps_groomed=[]` (grooming didn't actually happen).

### Example: Request 100

**Legacy snr_list accumulation:**
```
[UPDATE_PARAMS_DEBUG] req=100 key=snr_list value=18.10  # grooming attempt 1
[UPDATE_PARAMS_DEBUG] req=100 key=snr_list value=16.85  # grooming attempt 2
[UPDATE_PARAMS_DEBUG] req=100 key=snr_list value=15.95  # new LP 126
```

**Legacy final result:**
```
req=100 | lps_created=[126] | lps_groomed=[]
snr_list=[18.10, 16.85, 15.95]  # 3 values despite only 1 LP created
```

**V5 snr_list:**
```
snr_list=[15.95]  # Only the actual LP's SNR
```

### The Problem

When metrics reads `snr_list[i]` for LP 126's modulation tracking:
- **Legacy:** reads index 0 → 18.10 (from failed grooming attempt)
- **V5:** reads index 0 → 15.95 (correct LP SNR)

This causes 16-QAM.snr.max to be 18.10 in Legacy (incorrect index) vs 15.95 in V5 (correct).

### Source Code

The problematic line is `grooming.py:155`:
```python
self.sdn_props.snr_list.append(lp_info["snr_cost"])
```

This appends to `snr_list` during grooming iteration, before knowing if grooming will succeed.

## Fix Options

1. **Make V5 match Legacy (preserve bug):** Add same grooming SNR accumulation to V5
2. **Fix Legacy behavior:** Clear snr_list values from failed grooming in Legacy
3. **Fix metrics indexing:** Use lightpath_id_list indices to access correct snr values

## Previous Bugs Fixed

| Bug | Issue | Fix |
|-----|-------|-----|
| Request 40 | Wrong bandwidth for partial grooming | Added `slicing_target_bw` parameter |
| Request 46 | SNR recheck `slicing_flag` hardcoded True | Pass actual slicing_flag to SNR adapter |
| demand_ratio > 1.0 | Wrong bandwidth tracked for allocation | Added `actual_bw_to_allocate` parameter |

## Test Commands

```bash
# Run the failing test
python tests/run_comparison.py --test spain_C_fixed_grooming_snr_recheck

# Compare request 100 output
grep "req=100" new_debug.txt legacy_debug.txt

# Check snr_list updates
grep "UPDATE_PARAMS_DEBUG.*snr_list" legacy_debug.txt | head -20
```

## Files with Debug Prints (to clean up)

- `fusion/core/properties.py` - UPDATE_PARAMS_DEBUG
- `fusion/core/grooming.py` - GROOM_LP126_DEBUG, GROOM_CORE
- `fusion/core/metrics.py` - V5_SNR_TRACK, LEGACY_SNR_TRACK
- `fusion/core/orchestrator.py` - various debug prints
