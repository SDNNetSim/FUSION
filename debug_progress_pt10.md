# Debug Progress - Part 10: Lightpath ID and Slicing Fixes

## Session Overview
Found and fixed two critical bugs causing v5/v6 divergence. First iteration now matches perfectly (100%), but second iteration shows complete divergence from request 1.

## What We Accomplished

### 1. Fixed Duplicate Lightpath ID Generation (Commit: c7325639)

**Problem**: Lightpath IDs incrementing by 2 instead of 1
- `spectrum_assignment.py:926` was calling `get_lightpath_id()`
- `sdn_controller.py:850` was ALSO calling `get_lightpath_id()`
- Result: Every allocation consumed 2 IDs (lp_id: 2, 4, 6, 8...)

**Solution**: Removed duplicate call from `spectrum_assignment.py`
- Only `sdn_controller` should generate lightpath IDs (matches v5)
- Lightpath IDs now sequential: 1, 2, 3, 4...

**Evidence**:
- Before: v6 line 50 had `req_id=40 lp_id=50` AND `lp_id=51`
- After: Properly sequential allocation

### 2. Fixed Slicing After Grooming (Commit: c7325639)

**Problem**: Slicing used original bandwidth instead of remaining bandwidth after grooming
- Request 40: 800 Gbps total
- v5: Groomed 400 Gbps, then sliced remaining 400 Gbps
- v6: Groomed 400 Gbps, then sliced FULL 800 Gbps (wrong!)

**Root Cause**: `light_path_slicing.py:252`
```python
# Before (wrong):
remaining_bw = int(self.sdn_props.bandwidth)

# After (correct):
remaining_bw = (
    self.sdn_props.remaining_bw
    if self.sdn_props.was_partially_groomed
    else int(self.sdn_props.bandwidth)
)
```

**Solution**: Use `remaining_bw` when `was_partially_groomed` is True (matches v5 logic)

### 3. Verified First Iteration Match

**Result**: Lines 1-254 are IDENTICAL between v5 and v6
- 100% match for iteration 0
- All allocations, releases, paths, slots, modulations match perfectly
- Proves both fixes are correct

## What We Discovered

### Critical Finding: Second Iteration Diverges Immediately

**Iteration 1 (v5) vs Iteration 1 (v6)**:
```
v5 line 425: req_id=1 lp_id=1 path=['4', '7', '8', '5', '11'] (same as iter 0)
v6 line 420: req_id=1 lp_id=1 path=['15', '16', '14'] (completely different!)
```

**Implications**:
- Random seed NOT being reset identically between v5 and v6
- OR network state NOT being reset properly
- Divergence happens from FIRST request of iteration 1

### File Structure Analysis
- v5.txt: 1657 CMP lines, 7 instances of req_id=1
- v6.txt: 1658 CMP lines, 8 instances of req_id=1
- v6 has 1 extra line (83 more lines total in diff)

### Seed Behavior
- Config file: No explicit seed specified
- Default behavior: `request_seed = iteration + 1`
  - Iteration 0: seed = 1
  - Iteration 1: seed = 2
- Both v5 and v6 should use this default

## What Still Needs To Be Done

### 1. Understand Test Setup
**Questions**:
- How were v5.txt and v6.txt generated?
  - Both from manual test runs on respective branches?
  - Or v5.txt from cached fixtures?
- Should iterations use SAME seed or DIFFERENT seeds?
  - Current: seed varies per iteration (1, 2, 3...)
  - v5 behavior: Need to verify

### 2. Investigate Iteration Reset Logic
**Areas to check**:
- `simulation.py:662` - `reset_iteration()` method
  - Resets lightpath_counter ✓
  - Resets network_spectrum_dict ✓
  - Resets reqs_status_dict ✓
  - Resets grooming structures ✓
  - Does NOT explicitly show seed reset (handled elsewhere)
- `simulation.py:804-849` - Seed setting per iteration
  - Looks correct (uses iteration + 1 default)
- Compare with v5's reset logic

### 3. Possible Root Causes for Iteration 1 Divergence
1. **Different seed calculation** between v5 and v6
2. **Network state not fully reset** (spectrum, lightpaths, grooming data)
3. **RNG state contamination** from iteration 0
4. **v5 uses constant seed** across iterations (need to verify)

### 4. Next Steps
1. Verify how v5.txt and v6.txt were generated
2. Check if v5 uses same seed for all iterations vs varying seed
3. Add logging to show actual seeds being used per iteration
4. Compare network state reset between v5 and v6
5. Test with explicit constant seed to isolate the issue

## Files Modified
- `fusion/core/spectrum_assignment.py`: Removed duplicate lightpath ID generation
- `fusion/modules/spectrum/light_path_slicing.py`: Fixed remaining_bw calculation

## Key Commits
- `c7325639`: fix: remove duplicate lightpath ID generation and fix slicing after grooming

## Success Metrics
- ✅ First iteration: 100% match (254/254 lines identical)
- ❌ Second iteration: Diverges from line 1
- **Overall**: Need to fix iteration reset to achieve 100% match across all iterations
