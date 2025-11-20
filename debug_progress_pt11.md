# Debug Progress - Part 11: Partial Grooming Fix and Iteration End Discovery

## Session Overview
Fixed partial grooming tracking for blocked requests. First 364 CMP events now match v5 perfectly. Discovered v5 releases all active requests at iteration end (60 extra releases) while v6 leaves them active.

## What We Accomplished

### 1. Added Comprehensive Debug Instrumentation
- Seed tracking: `[V5-SEED]` / `[V6-SEED]` shows request_seed per iteration
- Request generation: `[V5-REQ-GEN-DETAIL]` / `[V6-REQ-GEN-DETAIL]` shows first 10 requests
- Bandwidth accounting: `[DEBUG-LP-RELEASE-BW]` shows req_bw and remaining_bw before/after
- Release checks: `[DEBUG-LP-RELEASE-CHECK]` shows why lightpaths are/aren't released

### 2. Verified Request Generation is IDENTICAL
**Result**: Both v5 and v6 generate identical requests with same seeds
- Iteration 0: seed=1, req_id=1 → src=4 dst=11 bw=200 (both match)
- Iteration 1: seed=2, req_id=1 → src=15 dst=14 bw=400 (both match)
- Seeds properly vary per iteration (iteration + 1)

### 3. Discovered Root Cause: Partially Groomed Blocked Requests
**Problem**: When requests partially groom but can't allocate remaining bandwidth:
- v5: Marks as "partially served", schedules departure, releases at departure time
- v6: Was immediately blocking, orphaning groomed bandwidth

**Example** - req_id=158 (src=14 dst=24 bw=800):
- Groomed 300 Gbps onto lp_id=97
- Couldn't allocate remaining 500 Gbps → blocked
- v5: Kept 300 Gbps allocated until departure time, then released
- v6 (before fix): Orphaned 300 Gbps, never cleaned up

### 4. Implemented Fix for Partial Grooming
**Solution** (sdn_controller.py:1109-1117):
```python
# When all paths exhausted after partial grooming
if was_partially_groomed and lightpath_id_list:
    # Mark as partially served to schedule departure event
    was_routed = True
    was_partially_routed = True
```

**Result**: Partially groomed blocked requests now:
- Keep groomed bandwidth allocated (matches v5)
- Get departure events scheduled
- Release groomed lightpaths at departure time
- Generate proper CMP-RELEASE events

### 5. Verified Release Flow Works Correctly
**Example** - req_id=86 with lp_id=97:
- lp_id=97: capacity=500, req_id=86 uses 200, req_id=158 uses 300
- req_id=158 releases first (depart=2569.9): frees 300 → remaining_bw=300 (not empty)
- req_id=86 releases later (depart=2850.0): frees 200 → remaining_bw=500 (empty, released!)
- Result: `[CMP-RELEASE] req_id=86 lp_id=97` ✓

## What We Discovered

### Critical Finding: Iteration End Behavior Differs
**v5 Iteration 0:**
- 212 CMP-ALLOC events
- 212 CMP-RELEASE events
- All allocated requests released by iteration end
- Last releases at time >13,000 (e.g., req_id=67 departs at 13813.6)

**v6 Iteration 0:**
- 212 CMP-ALLOC events
- 152 CMP-RELEASE events
- **60 requests still active** at iteration end
- Last release at time ~4,445 (req_id=110 departs at 4445.3)

**Implication**: v5 either:
1. Runs iteration until ALL requests depart (no requests left active), OR
2. Does final cleanup at iteration end to release all active requests

v6 appears to end iteration after processing all 200 arrivals, regardless of active requests.

### Event Comparison Summary
- Lines 1-364: **Perfect match** between v5 and v6 ✓
- Lines 365-424: v5 has 60 extra CMP-RELEASE events (the active requests)
- Total: v5=424 events, v6=364 events, difference=60 events

## What Needs To Be Done

### 1. Investigate Iteration End Condition
**Questions**:
- Does v5 wait until all requests depart before ending iteration?
- Or does v5 do final cleanup to release active requests?
- Where is the iteration end condition in v5 vs v6?

**Files to check**:
- v5: `src/engine.py` - iteration/simulation loop
- v6: `fusion/core/simulation.py` - iteration loop around line 662 (reset_iteration)

### 2. Compare Simulation Loop Logic
Look for:
- Event queue processing termination condition
- Whether loop continues until queue is empty vs until N arrivals processed
- Any final cleanup code that releases active requests

### 3. Implement Matching Behavior
Once we understand v5's approach, either:
- **Option A**: Make v6 run until all requests depart (change iteration end condition)
- **Option B**: Add final cleanup in v6 to release all active requests at iteration end

### 4. Expected Outcome
After fix:
- Both iterations should have 212 ALLOC, 212 RELEASE
- CMP event counts should match exactly (424 events)
- 100% match for both iterations 0 and 1

## Files Modified
- `fusion/core/simulation.py`: Added seed and request generation logging
- `fusion/core/grooming.py`: Added bandwidth accounting and release check logging
- `fusion/core/sdn_controller.py`: Fixed partial grooming to mark as partially served

## Key Commits
- `9943abd2`: debug: add seed and request generation logging for v6
- `461d7307`: debug: add seed and request generation logging for v5
- Latest: fix: treat partially groomed blocked requests as partially served

## Success Metrics
- ✅ Request generation: 100% identical between v5 and v6
- ✅ First 364 CMP events: Perfect match
- ✅ Partial grooming: Now properly tracked and released
- ✅ Release flow: Working correctly (multi-request lightpaths)
- ❌ Total CMP events: 364 vs 424 (60 missing due to iteration end difference)
- **Next goal**: Match iteration end behavior to achieve 424 vs 424
