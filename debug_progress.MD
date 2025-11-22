# FUSION v5→v6 Grooming Migration: Progress Report

## Context

This document tracks fixes applied to resolve grooming test failures between v5 (`feature/grooming-new`) and v6 (`chore/stabilize-v6-expected-results`). See `debug_plan.MD` for methodology.

**Test Target**: `spain_C_fixed_grooming` via `tests/run_comparison.py`

**Root Cause Pattern**: The `check_gsnr()` method was ported from v5 to v6, but integration was incomplete—missing attributes, renamed properties, and disconnected code paths.

---

## Fixes Applied (12 Total)

### Category 1: v5→v6 Naming Changes

**Fix 1: `net_spec_dict` → `network_spectrum_dict`**
- **Files**: `fusion/core/snr_measurements.py` (lines 770, 788)
- **Issue**: v6 renamed attribute in SDNProps
- **Fix**: Global rename in check_gsnr method

**Fix 2: `num_slots` → `number_of_slots`**
- **Files**: `fusion/core/snr_measurements.py` (lines 774, 775, 802)
- **Issue**: v6 uses more descriptive naming
- **Fix**: Update all references in check_gsnr

**Fix 3: `plank` → `planck_constant`**
- **Files**: `fusion/core/snr_measurements.py` (line 779)
- **Issue**: v6 corrected spelling and improved naming
- **Fix**: Update reference in check_gsnr

---

### Category 2: Missing Data Structures

**Fix 4: Optical Band Frequencies**
- **Files**: `fusion/io/generate.py` (lines 31-37)
- **Issue**: Fiber properties missing frequency ranges for C/L/S bands
- **Fix**: Added standard ITU-T frequency specifications
  ```python
  "frequency_start_c": 191.5e12,  # C-band: 191.5-196.1 THz
  "frequency_end_c": 196.1e12,
  "frequency_start_l": 186.5e12,  # L-band: 186.5-191.5 THz
  "frequency_end_l": 191.5e12,
  "frequency_start_s": 185.0e12,  # S-band: 185.0-190.0 THz
  "frequency_end_s": 190.0e12,
  ```

**Fix 5: Noise Spontaneous Parameter (NSP) Dictionary**
- **Files**: `fusion/core/properties.py` (lines 141-148)
- **Issue**: SNRProps missing band-specific amplifier noise figures
- **Fix**: Added `nsp` dict with values per optical band (C: 1.8, L/S/O/E: 2.0)

**Fix 6: Required SNR Thresholds Dictionary**
- **Files**: `fusion/core/properties.py` (lines 149-157)
- **Issue**: SNRProps missing modulation-specific SNR requirements
- **Fix**: Added `req_snr` dict mapping modulation formats to dB thresholds
  ```python
  "BPSK": 6.8, "QPSK": 8.5, "8-QAM": 12.6,
  "16-QAM": 14.8, "32-QAM": 17.8, "64-QAM": 20.8
  ```

**Fix 7: Slicing Flag Attribute**
- **Files**: `fusion/core/properties.py` (line 114)
- **Issue**: SpectrumProps missing slicing mode indicator
- **Fix**: Added `slicing_flag: bool = False`

---

### Category 3: Missing Integration Points

**Fix 8: GSNR Support in Dynamic Slicing**
- **Files**: `fusion/core/snr_measurements.py` (lines 1193-1204)
- **Issue**: `handle_snr_dynamic_slicing()` only supported `snr_e2e_external_resources`
- **Fix**: Added `gsnr` case that calls `check_gsnr()` for C-band
- **Note**: Multi-band GSNR (`check_gsnr_mb`) not yet ported to v6

**Fix 9: Dynamic Modulation Selection Logic**
- **Files**: `fusion/core/snr_measurements.py` (lines 834-869)
- **Issue**: `check_gsnr()` tried to access `req_snr[None]` during dynamic slicing
- **Fix**: Restructured to handle two modes:
  - **Dynamic mode** (`slicing_flag=True` + `fixed_grid=True` + `dynamic_lps=True`): Determines modulation based on GSNR threshold
  - **Standard mode**: Requires pre-set modulation, validates GSNR

**Fix 10: Set Slicing Flag in Dynamic Path**
- **Files**: `fusion/core/spectrum_assignment.py` (line 953)
- **Issue**: `get_spectrum_dynamic_slicing()` never set `slicing_flag=True`
- **Fix**: Added flag assignment at method entry

---

### Category 4: Type Consistency Issues

**Fix 11: Lightpath Bandwidth Type Conversion**
- **Files**: `fusion/core/spectrum_assignment.py` (line 768)
- **Issue**: Stored raw `lp_bandwidth` (potentially string from dict) instead of numeric
- **Fix**: Store `lp_bandwidth_float` (explicitly converted at line 732)
- **Impact**: Prevents `TypeError` in grooming bandwidth division

**Fix 12: None Value Filtering in Statistics**
- **Files**: `fusion/core/metrics.py` (line 650)
- **Issue**: Metric lists contained `None` values, breaking `stdev()` calculations
- **Fix**: Filter with `filtered_value = [v for v in value if v is not None]`

---

## Current Status

**Test Execution**: All crashes resolved. Test now completes simulation and reaches statistics finalization.

**Next Steps** (per `debug_plan.MD` methodology):
1. ✅ **Phase 1 Complete**: Baseline assessment and crash fixes
2. **Phase 2 Pending**: Request-level instrumentation and comparison
3. **Phase 3 Pending**: Root cause analysis of result differences
4. **Phase 4 Pending**: Targeted grooming logic fixes
5. **Phase 5 Pending**: Full validation

---

## Known Gaps / Future Work

### 1. Multi-Band GSNR (Low Priority)
- **Location**: `snr_measurements.py` line 1197
- **Issue**: `check_gsnr_mb()` not ported from v5
- **Impact**: C+L band configurations will fail
- **Current Test**: Uses C-band only, so not blocking

### 2. Potential Grooming Logic Discrepancies
- **Evidence**: From initial research (see original Plan agent report)
  - **Crosstalk list appending**: v6 appends to both `crosstalk_list` and `snr_list` (lines 150-151 in grooming.py), v5 only to `snr_list`
  - **Integer casting**: v6 uses `int()` cast in `_find_path_max_bw` (line 89), v5 doesn't
- **Status**: Not yet validated with request-level comparison
- **Action Required**: Run instrumented test, compare logs per debug_plan.MD Phase 2-3

### 3. Debug Output Cleanup
- Multiple `[DEBUG-*]` print statements added during development
- Should be converted to proper logging or removed before merge

---

## Test Configuration Reference

**File**: `tests/fixtures/expected_results/spain_C_fixed_grooming/spain_c_fixed_grooming_config.ini`

**Key Settings**:
- `network = Spainbackbone30`
- `snr_type = gsnr` (why all GSNR fixes were needed)
- `is_grooming_enabled = True`
- `can_partially_serve = True`
- `dynamic_lps = True` (dynamic lightpath slicing)
- `fixed_grid = True`
- `route_method = k_shortest_path` (k=3)
- `allocation_method = first_fit`

---

## Key Files Modified

### Core Logic
- `fusion/core/snr_measurements.py` - GSNR calculation and SNR handling
- `fusion/core/spectrum_assignment.py` - Lightpath creation and dynamic slicing
- `fusion/core/grooming.py` - Grooming logic (unchanged, but identified for future review)

### Data Structures
- `fusion/core/properties.py` - SNRProps, SpectrumProps, SDNProps
- `fusion/io/generate.py` - Fiber physical properties

### Statistics
- `fusion/core/metrics.py` - Metrics aggregation and finalization

---

## Validation Checklist

Before declaring grooming tests passing:

- [ ] `spain_C_fixed_grooming` comparison passes
- [ ] All non-grooming tests still pass (baseline_spf_ff, baseline_kspf_ff, epsilon_greedy_bandit, ext_snr_4core_cls_dy-slice, xtar_slicing_pff)
- [ ] Additional grooming fixtures pass incrementally:
  - [ ] `spain_C_fixed_grooming_snr_recheck`
  - [ ] `spain_mb_CL_fixed_grooming`
  - [ ] `spain_mb_CL_fixed_grooming_snr_recheck`
  - [ ] `usbackbone_C_fixed_grooming`
  - [ ] `usbackbone_C_fixed_grooming_snr_recheck`
  - [ ] `usbackbone_mb_CL_fixed_grooming`
  - [ ] `usbackbone_mb_CL_fixed_grooming_snr_recheck`

---

## Debug Pattern Recognition

**Lesson Learned**: When porting complex methods (like `check_gsnr`) from v5→v6:
1. Check all property/attribute names against v6 conventions
2. Ensure all referenced data structures exist and are initialized
3. Verify all code paths that call the method set required flags/state
4. Check type consistency (especially dict values that might be strings vs numbers)
5. Add None-safety to aggregation/statistics calculations

**Recommendation for Future Ports**: Create a checklist based on these 12 fixes to validate before declaring a port complete.
