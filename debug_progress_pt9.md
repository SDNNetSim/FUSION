# Debug Progress - Part 9: Deep Divergence Investigation

## Session Overview
After achieving 97.9% match rate (190/194 requests), we recognized that **v5 is ground truth** and ANY difference indicates a bug. Even seemingly "better" results (e.g., v6 selecting 32-QAM where v5 selects 16-QAM) are problems requiring investigation.

## What We Accomplished

### 1. Applied Critical Fixes (Committed: 50a0894e)
**Fix 13: Corrected SNR Thresholds** (fusion/core/properties.py)
- Updated req_snr to match v5 exactly:
  - BPSK: 6.8 → 3.71 dB
  - QPSK: 8.5 → 6.72 dB
  - 8-QAM: 12.6 → 10.84 dB
  - 16-QAM: 14.8 → 13.24 dB
  - 32-QAM: 17.8 → 16.16 dB
  - 64-QAM: 20.8 → 19.01 dB

**Fix 14: Corrected NSP Values** (fusion/core/properties.py)
- C-band: 1.8 → 1.77
- L-band: 2.0 → 1.99
- Eliminates ~0.07 dB GSNR calculation differences

**Fix 15: Congestion Handling Cleanup** (fusion/core/sdn_controller.py)
- Before releasing lightpath during congestion:
  1. Remove current request from requests_dict
  2. Restore allocated bandwidth
  3. Then safely release lightpath

### 2. Analyzed 4 Remaining Mismatches

**Request 86**: v5=8-QAM, v6=BLOCKED
- V6 fully groomed (no new allocation needed)
- Comparison script error (only counted GSNR-CALC, not grooming)

**Request 132**: v5=16-QAM, v6=32-QAM
- NOT a v6 improvement - indicates divergence
- V6: 64-QAM failed (gsnr=18.07 < 19.01), 32-QAM succeeded
- Different network state causing different GSNR values

**Request 171**: v5=16-QAM, v6=BLOCKED
- Part 1 groomed successfully
- Part 2 exhausted all modulations due to congestion
- Network state difference

**Request 195**: v5=16-QAM, v6=8-QAM
- Part 1: gsnr=12.906, requires 8-QAM (16-QAM needs ≥13.24)
- Part 2: All modulations failed
- Network state difference

### 3. Key Insight: Cascading Differences
User correction: "There should be NO network state difference"
- If v6 gets different modulation than v5, something is fundamentally wrong
- "Early differences compound" suggests systematic issue
- Need to find EXACT point of first divergence

## Instrumentation Added

### V5 Branch (commit: cc50ed25)
- `[CMP-ALLOC]`: Tracks every allocation with full details (req_id, lp_id, path, slots, mod)
- `[CMP-RELEASE]`: Tracks every release (req_id, lp_id, path, slicing flag)
- `[V5-GSNR-CALC]`: Already present from previous debugging

### V6 Branch (commit: d54f9c65)
- `[CMP-ALLOC]`: Matches v5 format exactly
- `[CMP-RELEASE]`: Matches v5 format exactly
- `[V6-GSNR-CALC]`: Already present from previous debugging

## Next Steps

### 1. Run Both Branches with Comparison Instrumentation
```bash
# On v5 (feature/grooming-new)
python tests/run_comparison.py --fixture spain_C_fixed_grooming --verbose > v5_cmp.txt 2>&1

# On v6 (chore/stabilize-v6-expected-results)
python tests/run_comparison.py --fixture spain_C_fixed_grooming --verbose > v6_cmp.txt 2>&1
```

### 2. Compare Line-by-Line to Find First Divergence
```bash
# Extract CMP lines for comparison
grep "CMP-" v5_cmp.txt > v5_cmp_only.txt
grep "CMP-" v6_cmp.txt > v6_cmp_only.txt

# Compare side-by-side
diff -y v5_cmp_only.txt v6_cmp_only.txt | head -100
```

### 3. Investigation Strategy
- Find the FIRST CMP-ALLOC or CMP-RELEASE that differs
- Check if it's request 1, or later
- If request 1 differs: allocation logic bug
- If later: cascading from previous difference
- Compare GSNR-CALC values for same request/path between v5/v6
- Check spectrum state before first divergence

### 4. Areas to Investigate
Based on potential root causes:
- **Allocation logic**: Slot assignment, guard bands, spectrum search
- **Release logic**: Spectrum cleanup, lightpath tracking
- **Statistical calculations**: Aggregation, averaging, finalization
- **Grooming logic**: Bandwidth allocation, lightpath reuse
- **GSNR precision**: Floating point differences, rounding

## Expected Outcome
Find the exact line/operation where v5 and v6 first diverge, identify the root cause bug, fix it, and achieve 100% match rate.

## Files Modified
- `fusion/core/properties.py`: SNR thresholds and NSP values
- `fusion/core/sdn_controller.py`: Congestion handling + comparison instrumentation
- `fusion/core/spectrum_assignment.py`: Mismatch-specific instrumentation
- `fusion/core/grooming.py`: Grooming attempt tracking
- `src/sdn_controller.py` (v5): Comparison instrumentation

## Key Commits
- `50a0894e`: fix: correct SNR thresholds and NSP values to match v5
- `cc50ed25`: Add CMP-ALLOC and CMP-RELEASE instrumentation to v5
- `d54f9c65`: Add CMP-ALLOC and CMP-RELEASE instrumentation to v6
