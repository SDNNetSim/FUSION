# FUSION v5→v6 Grooming Debug: Part 2 - Modulation Override Issue

## Context

Continuation of grooming mismatch investigation. Part 1 fixed crashes and integration issues. Part 2 addresses result discrepancies.

**Test Target**: `spain_C_fixed_grooming` via `tests/run_comparison.py`

**Current Failure**: `weights_dict.200.32-QAM` returns None instead of expected values (mean=224.75, std=8.5, min=213.0, max=233.0)

---

## Investigation Summary

### Phase 1: Initial Hypothesis - Grooming Logic Issues

Applied 3 fixes based on v5 comparison:

**Fix 1: Remove Integer Truncation** (grooming.py:89)
```python
# Before: key=lambda group: int(group["total_remaining_bandwidth"])
# After:  key=lambda group: group["total_remaining_bandwidth"]
```

**Fix 2: Remove Duplicate SNR Cost Append** (grooming.py:150)
```python
# Before: Appended to both crosstalk_list AND snr_list
# After:  Only appends to snr_list (matching v5)
```

**Fix 3: Add Early Returns for Groomed Requests** (metrics.py:473-490)
```python
# Skip stats tracking for fully groomed requests (v5 behavior)
if sdn_data.was_groomed:
    self.bit_rate_request += int(sdn_data.bandwidth)
    return

# Skip if partially groomed with no new lightpaths
if remaining_bw != "0" and not was_new_lps:
    return
```

**Result**: Test runs to completion, but 32-QAM statistics still None.

---

### Phase 2: Modulation Selection Analysis

Added debug instrumentation to `check_gsnr` (snr_measurements.py:834-852):
```python
print(f"[DEBUG-GSNR-MOD-ORDER] req_id={...}, gsnr_db={...}, mod_order={...}")
print(f"[DEBUG-GSNR-MOD-CHECK] mod={...}, req_snr={...}, gsnr={...}, meets={...}")
print(f"[DEBUG-GSNR-MOD-SELECT] SELECTED mod={...}, bw={...}")
```

**Findings**:
1. Modulation order is correct: `['64-QAM', '32-QAM', '16-QAM', '8-QAM', 'QPSK', 'BPSK']`
2. SNR thresholds working correctly (32-QAM requires 17.8 dB)
3. **32-QAM is selected 210 times** when GSNR is in range 17.8-20.8 dB
4. Example: `req_id=6, gsnr=18.14 → SELECTED mod=32-QAM, bw=700`

---

### Phase 3: Weight Tracking Analysis

Checked if 32-QAM selections reach weights_dict tracking:

**Command**: `grep "DEBUG-WEIGHT.*32-QAM" v6.txt | wc -l`
**Result**: **0** (zero 32-QAM weights tracked)

**Comparison**:
- 8-QAM: Tracked in weights_dict ✓
- 16-QAM: Tracked in weights_dict ✓
- **32-QAM: NOT tracked** ✗
- 64-QAM: Tracked in weights_dict ✓

**For bandwidth=200 lightpaths specifically**:
```
v6 actual:
- 64-QAM: 147 lightpaths
- 16-QAM: 3 lightpaths
- 8-QAM: 3 lightpaths
- 32-QAM: 0 lightpaths  ← Missing!

v5 expected (from fixtures):
- 64-QAM: mean path weight 115.0 km
- 32-QAM: mean path weight 224.75 km  ← Expected
- 16-QAM: mean path weight 444.7 km
- 8-QAM: mean path weight 742.5 km
```

---

### Phase 4: Root Cause Discovery - Modulation Override Bug

**Critical Evidence** (Request 6 trace):
```
[DEBUG-GSNR-MOD-SELECT] SELECTED mod=32-QAM, bw=700
[DEBUG-WEIGHT] req_id=6, lp_id=12, bw_key=800, mod=64-QAM, path_weight=242.00, was_new_lp=True
```

**Analysis**:
1. `check_gsnr()` correctly selects **32-QAM** based on GSNR=18.14 dB
2. Lightpath lp_id=12 is actually created with **64-QAM** (not 32-QAM!)
3. Bandwidth also changes: check_gsnr returns bw=700, but lightpath created with bw=800

**Conclusion**: **The modulation format returned by `check_gsnr()` is being OVERRIDDEN** somewhere between the SNR check and actual lightpath creation.

---

## Technical Details

### check_gsnr Return Values

From `snr_measurements.py:743-871`, `check_gsnr()` returns:
```python
return (resp, gsnr_db, bw_resp)
# resp: modulation format (e.g., "32-QAM") or False
# gsnr_db: calculated GSNR in dB
# bw_resp: supported bitrate from bw_mapping
```

The `bw_mapping` in check_gsnr (lines 754-761):
```python
bw_mapping = {
    "64-QAM": 800,
    "32-QAM": 700,  # ← Returned when 32-QAM selected
    "16-QAM": 600,
    "8-QAM": 500,
    "QPSK": 400,
    "BPSK": 200
}
```

**Important**: This `bw_mapping` represents **supported bitrate capacity**, NOT the actual lightpath bandwidth. The actual lightpath_bandwidth (used as weights_dict key) is determined by slicing logic and can be 100, 200, 300, ..., 800.

### Expected Behavior

From v5 expected results, 32-QAM should appear with multiple bandwidths:
```python
v5_32qam_bandwidths = ['200', '300', '400', '500', '600', '700']
```

This means:
1. Dynamic slicing creates lightpaths with various bandwidths (e.g., 200 Gbps)
2. `check_gsnr()` is called to determine modulation based on path SNR
3. If GSNR is 17.8-20.8 dB, 32-QAM should be selected
4. Lightpath is created with bandwidth=200 (from slicing) and modulation=32-QAM (from check_gsnr)

### Actual Behavior in v6

1. `check_gsnr()` correctly selects 32-QAM (210 times) ✓
2. **Something overrides the modulation before lightpath creation** ✗
3. Lightpaths end up with different modulations (e.g., 64-QAM instead of 32-QAM)
4. Result: No 32-QAM lightpaths created → weights_dict.*.32-QAM remains empty

---

## Impact Assessment

### What Works Correctly

1. Grooming fixes (int() cast, crosstalk_list) are correct
2. Early return logic for stats tracking is correct (matching v5)
3. SNR calculation and modulation selection logic in check_gsnr works correctly
4. Modulation order and thresholds are correct

### What Is Broken

1. **Modulation override**: check_gsnr's return value is ignored/overridden
2. This is a **spectrum assignment bug**, not a grooming bug
3. Likely introduced during v5→v6 architectural refactoring
4. Affects ALL dynamic slicing scenarios using GSNR, not just grooming

---

## Next Steps

### Investigation Targets

Need to trace the call path from check_gsnr to lightpath creation:

1. **Entry Point**: `handle_snr_dynamic_slicing()` in snr_measurements.py
   - Calls `check_gsnr()` and gets (modulation, gsnr, bw_resp)
   - How is this modulation passed forward?

2. **Spectrum Assignment**: `get_spectrum_dynamic_slicing()` in spectrum_assignment.py
   - Receives SNR check results
   - Creates lightpaths with allocated slots
   - **Where is modulation format set?**

3. **Lightpath Creation**: SDN controller lightpath allocation
   - `sdn_props.modulation_list` should contain check_gsnr's modulation
   - **Is it being overwritten?**

4. **Possible Override Locations**:
   - After SNR check but before spectrum assignment
   - During spectrum slot allocation
   - During lightpath status dict population
   - In grooming reuse logic (but we fixed those paths)

### Debug Strategy

1. Add instrumentation at every point where modulation is read/written
2. Trace request 6 specifically (known to have 32-QAM→64-QAM override)
3. Find the exact line where 32-QAM is replaced with 64-QAM
4. Compare with v5 behavior at that point
5. Apply minimal fix to preserve check_gsnr selection

---

## Files Modified So Far

### Core Logic
- `fusion/core/grooming.py` - Grooming bandwidth selection and SNR tracking
- `fusion/core/metrics.py` - Statistics tracking and early returns
- `fusion/core/snr_measurements.py` - Debug instrumentation in check_gsnr

### Data Structures (from Part 1)
- `fusion/core/properties.py` - SNRProps, SpectrumProps (req_snr, nsp, slicing_flag)
- `fusion/io/generate.py` - Fiber optical band frequencies

---

## Open Questions

1. **Where does the override happen?**
   - Need to trace spectrum_assignment.py and snr_measurements.py interaction

2. **Why only 32-QAM?**
   - Other modulations (8-QAM, 16-QAM, 64-QAM) work fine
   - Is there something special about 32-QAM's SNR range or bw_mapping value?

3. **Was this working in v5?**
   - v5 fixtures show 32-QAM was used extensively
   - What changed in v6's architecture that broke this?

4. **Is this test-specific or systemic?**
   - Affects spain_C_fixed_grooming test
   - Likely affects ALL tests using dynamic_lps + GSNR + slicing
   - Could impact production simulations

---

## References

- Part 1: debug_progress.md (initial crash fixes and integration)
- v5 branch: feature/grooming-new (ground truth)
- v6 branch: chore/stabilize-v6-expected-results (current work)
- Test: tests/run_comparison.py --test-case spain_C_fixed_grooming
- Expected results: tests/fixtures/expected_results/spain_C_fixed_grooming/

---

## Debugging Commands Used

```bash
# Check 32-QAM selections
grep "SELECTED mod=32-QAM" v6.txt | wc -l  # 210

# Check 32-QAM weight tracking
grep "DEBUG-WEIGHT.*32-QAM" v6.txt | wc -l  # 0

# Trace specific request
grep -A 20 "req_id=6, gsnr_db=18.14" v6.txt | grep -E "(SELECTED|DEBUG-WEIGHT)"

# Check bandwidth=200 modulation distribution
grep "DEBUG-WEIGHT.*bw_key=200" v6.txt | cut -d',' -f4 | sort | uniq -c
```

---

## Status

**Current State**: Grooming logic fixed, but modulation override bug discovered during investigation.

**Blocking Issue**: 32-QAM selections from check_gsnr are being overridden before lightpath creation.

**Next Action**: Investigate spectrum assignment code to find override location and fix.
