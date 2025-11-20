# FUSION v5→v6 Grooming Debug: Part 7 - Modulation/SNR Instrumentation

## Session Summary

Added comprehensive instrumentation for debugging modulation format selection and SNR calculation differences between v5 and v6.

---

## Completed Work

### 1. Request Generation Fix (Committed & Pushed)

**File**: `fusion/core/request.py`, `fusion/core/simulation.py`

**Changes**:
- Implemented tuple key structure: `(request_id, time)` instead of `float`
- Removed collision check/retry logic (no longer needed)
- Updated sorting: `sorted(..., key=lambda x: x[0][1])`
- Updated type signatures: `dict[tuple[int, float], dict[str, Any]]`

**Result**: ✓ Request generation now matches v5 perfectly (800/800 requests identical per iteration)

### 2. Request Generation Verification

**Status**: PASS
- Both v5 and v6 generate 200 requests × 4 iterations = 800 total
- Iterations are in different order (v6 iter 2 = v5 iter 3, etc.) but content matches
- All request attributes identical: req_id, src, dst, bw, arrival, departure

### 3. Allocation Analysis

**Key Finding**: Modulation format mismatch between v5 and v6

**Observations**:
- Routes match ✓
- Spectrum slots (start, end) appear to match at first glance
- **Modulation formats differ** ✗ - v6 consistently selects lower formats than v5
- Allocation count: v6=902, v5=808 (94 extra allocations in v6)

**Examples**:
- Position 1: v6=8-QAM vs v5=16-QAM
- Position 2: v6=16-QAM vs v5=32-QAM
- Position 11: v6=32-QAM vs v5=64-QAM

### 4. V6 Modulation/SNR Instrumentation Added

**File**: `fusion/core/spectrum_assignment.py`

**New Print Statements**:
- `[V6-MOD-TRY]` - When trying each modulation format
- `[V6-MOD-SKIP]` - When skipping a format (with reason)
- `[V6-SLOTS]` - Slots needed calculation + path length
- `[V6-SPEC-RESULT]` - Spectrum allocation result (is_free, start, end)
- `[V6-SNR]` - SNR check results (snr_ok, xt_cost, lp_bw)

**File**: `fusion/core/sdn_controller.py`

**Updated Print Statements**:
- `[V6-ALLOC]` - Added path_len and slots_needed fields

**File**: `fusion/core/simulation.py`

**New Print Statements**:
- `[V6-ITER-START]` - Iteration number, erlang value, thread

---

## Analysis Questions

**Critical Question from User**:
> "If we use different mod formats, that would mean we'd use different slots, so how on earth would start and end slot assignments be the same?"

**Answer**: Need to verify this with detailed instrumentation. The initial comparison may have been misleading or comparing wrong data. The new instrumentation will show:
1. Exact slots_needed for each modulation format attempt
2. Which modulation format is tried first
3. Whether spectrum is found for each format
4. The actual start/end slots allocated

---

## Next Steps

### Immediate (Part 7 continuation):

1. **Add v5 instrumentation** (same as v6):
   - Switch to `feature/grooming-new` branch
   - Add matching print statements in:
     - `src/spectrum_assignment.py` (or equivalent)
     - `src/sdn_controller.py`
     - `src/engine.py`
   - Use `[V5-*]` prefixes to match v6 format

2. **Run comparison tests**:
   - Execute on both branches with same erlang values
   - Capture output to v5.txt and v6.txt

3. **Analyze modulation selection**:
   - Compare `[V5-MOD-TRY]` vs `[V6-MOD-TRY]` for same requests
   - Identify first divergence point
   - Check if mod_format_list order differs
   - Verify path length calculations match
   - Compare SNR calculations

### Investigation Areas:

1. **Modulation Format List Order**
   - Is the mod_format_list ordered differently in v5 vs v6?
   - Are they trying higher formats first vs lower formats first?

2. **Path Length / Distance Calculation**
   - Does v5/v6 calculate path distance differently?
   - Are reach limits different for each modulation format?

3. **SNR Calculation**
   - Are SNR thresholds the same?
   - Is crosstalk calculation identical?
   - Are bandwidth mappings consistent?

4. **Slots Needed Calculation**
   - Does `_calculate_slots_needed` differ between v5/v6?
   - Are guard slots handled the same way?

---

## Files Modified (V6 Only - Not Committed)

1. `fusion/core/spectrum_assignment.py` - Modulation/SNR instrumentation
2. `fusion/core/sdn_controller.py` - Enhanced allocation tracking
3. `fusion/core/simulation.py` - Iteration/erlang tracking

---

## Success Criteria

- [ ] v5 instrumentation added (matching v6 format)
- [ ] Comparison tests run on both branches
- [ ] Modulation format selection logic differences identified
- [ ] Root cause of modulation mismatch determined
- [ ] Fix applied to align v6 with v5 grooming behavior
- [ ] All allocations match between v5 and v6
- [ ] Comparison test passes

---

## Notes

- Keep all existing print statements (REQ-GEN, ALLOC, RELEASE)
- User will run tests, not automated
- Focus on request-level comparison to trace exact divergence point
