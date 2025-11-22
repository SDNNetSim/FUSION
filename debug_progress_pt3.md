# FUSION v5→v6 Grooming Debug: Part 3 - Modulation Selection Investigation

## Current Status

**Test**: `spain_C_fixed_grooming` via `tests/run_comparison.py`

**Issue**: Modulation distribution mismatch between v5 and v6
- v5 expects: `mods_used_dict.100.32-QAM = 0`
- v6 produces: `mods_used_dict.100.32-QAM = 4`
- Similar discrepancies for other bandwidth/modulation combinations

## Root Cause Analysis Progress

### What We Fixed (Parts 1-2)

1. **12 integration fixes** (Part 1): v5→v6 naming changes, missing data structures, type issues
2. **3 grooming logic fixes** (Part 2): Integer truncation, duplicate SNR appends, stats tracking
3. **slicing_flag persistence bug**: Added reset in `_initialize_spectrum_information()` (spectrum_assignment.py:468)

**Result**: 32-QAM lightpaths are now being created (was 0, now 222), but distribution still doesn't match v5.

### Current Investigation: Modulation Selection for Non-Sliced Lightpaths

#### Key Findings

**Problem**: v6 creates bw=100 lightpaths with 32-QAM that v5 doesn't create.

**Evidence from logs**:
```
Request 10:
[MOD-LIST] req_id=10, mod_list=['64-QAM', '32-QAM', '16-QAM', '8-QAM', 'QPSK', 'BPSK'], slice_bw=None
[SET-MOD] req_id=10, mod=64-QAM, slicing_flag=False, slice_bw=None
[SET-MOD] req_id=10, mod=32-QAM, slicing_flag=False, slice_bw=None  ← Succeeds!
[DEBUG-LP-CREATED] req_id=10, lp_id=20, lp_bw=100, remaining_bw=0.00
[DEBUG-LP-STORE-MOD] req_id=10, lp_id=20, mod_format=32-QAM, lp_bw=100
```

#### Observations

1. **Full modulation list**: All requests receive `['64-QAM', '32-QAM', '16-QAM', '8-QAM', 'QPSK', 'BPSK']`
   - No path-length filtering (no `False` entries)
   - Caused by extremely large `max_length` values in mod_formats.json (40,000+ km)

2. **Multiple allocation attempts**: Request 10 calls `get_spectrum()` 4 times
   - 1st call: Tries 64-QAM→32-QAM, **succeeds with 32-QAM** → creates bw=100 lightpath
   - 2nd-4th calls: All modulations fail
   - Finally: Dynamic slicing with GSNR → 8-QAM, bw=500

3. **Mysterious bw=100**: Request is 800 Gbps, but creates 100 Gbps lightpath first
   - Not from dynamic slicing (no GSNR logs before creation)
   - Not from grooming (no grooming logs)
   - Not from static slicing (no slice_bw parameter)
   - Possibly from 1+1 protection backup path?

4. **slicing_flag behavior**: Correctly `False` during these allocations (fix working)

#### Hypothesis

**Modulation selection difference**: v6 iterates through full mod_format_list and succeeds with higher-order modulations (32-QAM) when GSNR check passes. v5 might use different logic or fail the GSNR check.

**Possible causes**:
1. **Data structure mismatch**: `modulation_formats_dict` might have wrong structure
   - Config has per-bandwidth structure: `{"100": {"32-QAM": {...}}, "200": {...}}`
   - Routing code accesses: `modulation_formats_dict[mod_format]` (no bandwidth key!)
   - This could be reading wrong max_length values or failing to find entries

2. **GSNR calculation difference**: v5 and v6 calculate GSNR differently (unlikely, we ported check_gsnr)

3. **Different code path**: These bw=100 lightpaths might use a v6-specific path that doesn't exist in v5

## Next Steps

### Immediate Investigation

1. **Debug modulation_formats_dict structure**:
   - Added instrumentation in k_shortest_path.py:148
   - Will show actual max_length values being used
   - Will reveal if dict structure is incorrect

2. **Identify bw=100 lightpath source**:
   - Determine which code path creates these 100 Gbps lightpaths
   - Check if it's 1+1 protection, grooming attempts, or other mechanism

3. **Compare v5 vs v6 for same request**:
   - Already have v5.txt and v6.txt outputs
   - Need deeper comparison of request 10's full flow

### Potential Fixes

**If dict structure is wrong**:
- Fix how modulation_formats_dict is populated/accessed
- Ensure routing module uses correct per-bandwidth max_length values

**If code path is different**:
- Align v6's allocation logic with v5's
- May need to port additional v5 logic for non-sliced allocations

**If GSNR check is wrong**:
- Compare GSNR calculations between v5 and v6 for same paths
- Check if req_snr thresholds are correct

## Files Modified (This Session)

### Instrumentation Added
- `fusion/core/spectrum_assignment.py:807` - Log mod_format_list at get_spectrum entry
- `fusion/core/spectrum_assignment.py:895` - Log when modulation is set
- `fusion/modules/routing/k_shortest_path.py:148` - Log modulation dict access

### Previous Fixes (Still Active)
- `fusion/core/spectrum_assignment.py:468` - Reset slicing_flag
- `fusion/core/snr_measurements.py:903-909` - Comment about check_gsnr return types
- Parts 1-2 fixes (see debug_progress.MD and debug_progress_pt2.md)

## Test Command

```bash
python tests/run_comparison.py --test-case spain_C_fixed_grooming > v6.txt 2>&1
```

## Key Questions to Answer

1. What is the actual structure of `modulation_formats_dict` at runtime?
2. Why are bw=100 lightpaths being created for 800 Gbps requests?
3. Why does get_spectrum get called 4 times for request 10?
4. What is the GSNR value for request 10's path, and why does 32-QAM pass?
5. Does v5 also create these bw=100 lightpaths but with different modulations?

## References

- Part 1: `debug_progress.md` - Initial 12 fixes, GSNR integration
- Part 2: `debug_progress_pt2.md` - Modulation override discovery, grooming fixes
- Plan: `debug_plan.MD` - Overall debugging methodology
- v5 branch: `feature/grooming-new` (ground truth)
- v6 branch: `chore/stabilize-v6-expected-results` (current work)
