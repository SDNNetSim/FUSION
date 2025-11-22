# Debug Progress - Part 8: GSNR Threshold Investigation

## Session Overview
Added detailed SNR/GSNR instrumentation to identify why v6 selects lower modulation formats than v5.

## What We Did

### 1. Initial Analysis
- Analyzed v5.txt and v6.txt outputs from Part 7
- Confirmed modulation format mismatch pattern:
  - Request 1: v5 selects 16-QAM, v6 selects 8-QAM
  - Request 2: v5 selects 32-QAM, v6 selects 16-QAM
  - Pattern: v6 consistently requires 1 step lower modulation

### 2. First Instrumentation Attempt (Wrong Path)
- Added `[V5-XT-CALC]` and `[V6-XT-CALC]` instrumentation to `check_xt()` function
- Expected to see crosstalk threshold comparisons
- **Result**: No output - check_xt() was never called!

### 3. Root Cause Discovery
- Checked config: `snr_type = gsnr` (NOT `xt_calculation`)
- The system uses `check_gsnr()`, not `check_xt()`
- This explains why XT-CALC instrumentation never appeared

### 4. Correct Instrumentation Added
**v5 (`src/snr_measurements.py` line 537-540):**
```python
req_snr_threshold = self.snr_props.req_snr[self.spectrum_props.modulation]
resp = gsnr_db >= req_snr_threshold
print(f"[V5-GSNR-CALC] req_id={...} mod={...} gsnr_db={...:.6f} req_snr={...:.6f} decision={PASS/FAIL}")
```

**v6 (`fusion/core/snr_measurements.py` line 858-861):**
```python
req_snr_threshold = self.snr_props.req_snr[self.spectrum_props.modulation]
resp = gsnr_db >= req_snr_threshold
print(f"[V6-GSNR-CALC] req_id={...} mod={...} gsnr_db={...:.6f} req_snr={...:.6f} decision={PASS/FAIL}")
```

## Key Discoveries

### 1. SNR Method Confusion
- Config has `requested_xt` thresholds (e.g., 16-QAM: -36.69 dB)
- **These are NOT used** when `snr_type = gsnr`
- GSNR uses different thresholds: `self.snr_props.req_snr[modulation]`

### 2. Threshold Comparison Logic
**GSNR decision rule:**
```python
gsnr_db >= req_snr[modulation]  # Must meet or exceed threshold
```

**Observation from existing output:**
- Request 1: xt_cost ≈ 14.5 dB
  - v5: 16-QAM PASSES (xt_cost=14.532)
  - v6: 16-QAM FAILS (xt_cost=14.463)
- Lower xt_cost should be BETTER (less crosstalk)
- Yet v6 with lower xt_cost FAILS where v5 PASSES

### 3. Potential Root Causes
1. **Different GSNR calculation between v5/v6** → producing different gsnr_db values
2. **Different req_snr thresholds** → threshold values differ between versions
3. **Different lp_bw behavior** → affects GSNR calculation:
   - v5: lp_bw=0 for failed, actual BW for passed
   - v6: lp_bw always shows modulation-specific bandwidth

## Current Status

### Instrumentation Complete
**v5 (feature/grooming-new):**
- ✓ V5-MOD-TRY, V5-SLOTS, V5-SPEC-RESULT
- ✓ V5-SNR, V5-MOD-SKIP
- ✓ V5-ALLOC, V5-ITER-START
- ✓ **V5-GSNR-CALC** (NEW!)

**v6 (chore/stabilize-v6-expected-results):**
- ✓ V6-MOD-TRY, V6-SLOTS, V6-SPEC-RESULT
- ✓ V6-SNR, V6-MOD-SKIP
- ✓ V6-ALLOC, V6-ITER-START
- ✓ **V6-GSNR-CALC** (NEW!)

### Test Files
- `v5.txt` - Generated with GSNR instrumentation
- `v6.txt` - Ready to regenerate with GSNR instrumentation

## Next Steps

### 1. Run v6 with GSNR Instrumentation
```bash
# On branch: chore/stabilize-v6-expected-results
# Run tests and capture output to v6.txt
```

### 2. Compare GSNR-CALC Outputs
Extract and compare for same requests:
```bash
grep "GSNR-CALC" v5.txt | head -20
grep "GSNR-CALC" v6.txt | head -20
```

### 3. Analyze Differences
Look for:
- **gsnr_db values**: Are they different for same request/path?
- **req_snr thresholds**: Are threshold values different between v5/v6?
- **Decision patterns**: Why does same gsnr_db get different decisions?

### 4. Hypothesis Testing
Once we have GSNR-CALC data:

**If gsnr_db differs:**
→ GSNR calculation formula changed between v5/v6
→ Investigate GSNR calculation code differences

**If req_snr thresholds differ:**
→ Threshold initialization changed
→ Check where `snr_props.req_snr` is populated

**If both are same but decision differs:**
→ Comparison logic bug
→ Check `>=` vs `>` or other logic changes

## Technical Notes

### GSNR vs XT Calculation
- **XT (Crosstalk)**: Measures interference between channels
- **GSNR (Generalized SNR)**: Comprehensive path quality metric
  - Includes ASE (Amplified Spontaneous Emission) noise
  - Includes NLI (Nonlinear Interference) from other channels
  - More accurate for real optical systems

### Config Files
- Test config: `tests/fixtures/expected_results/spain_C_fixed_grooming/spain_c_fixed_grooming_config.ini`
- Modulation formats: `tests/fixtures/expected_results/spain_C_fixed_grooming/mod_formats.json`
- SNR type: `gsnr` (line 59 of config)

### File Locations
**v5:**
- SNR: `src/snr_measurements.py` (line 456-563)
- Spectrum: `src/spectrum_assignment.py`
- SDN: `src/sdn_controller.py`

**v6:**
- SNR: `fusion/core/snr_measurements.py` (line 743-876)
- Spectrum: `fusion/core/spectrum_assignment.py`
- SDN: `fusion/core/sdn_controller.py`

## Expected Outcome
GSNR-CALC output will show exact threshold values and calculated GSNR for each modulation attempt, revealing why v6 requires lower modulation formats than v5.
