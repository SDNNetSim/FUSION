# Debug Progress - Part 12: Modulation Statistics Divergence in Iteration 1

## Session Overview
After fixing partial grooming, CMP-RELEASE events now match perfectly (212 in both v5 and v6). However, comparison tests still fail on modulation statistics. Enhanced tracking revealed iteration 1 has fundamental routing differences.

## What We Accomplished

### 1. Verified Iteration 0 is Perfect
**Result**: Both v5 and v6 match exactly in iteration 0
- 212 CMP-ALLOC events, identical
- 212 CMP-RELEASE events, identical
- Last allocation: `req_id=200 lp_id=212` (same in both)
- All paths, modulations, spectrum assignments match

### 2. Added Enhanced Request Tracking
**New CMP-ALLOC fields** (both v5 and v6):
```
[CMP-ALLOC] req_id=X lp_id=Y src=A dst=B bw=Z groomed=T/F
            path=[...] path_len=N core=C band=B
            start=S end=E mod=MOD
```

Now tracks:
- Source and destination nodes
- Bandwidth requirement
- Whether request was groomed
- Full path and path length
- Modulation format selected

### 3. Discovered Critical Divergence in Iteration 1

**Issue #1: req_id=4 is BLOCKED in v6 but ALLOCATED in v5**

v5 iteration 1:
```
req_id=4 lp_id=4 src=14 dst=24 bw=100
path=['14', '16', '17', '19', '6', '24'] path_len=5
mod=8-QAM
```

v6 iteration 1:
- req_id=4 is completely missing (blocked)
- Sequence goes: 1, 2, 3, (skip 4), 5, 6, 7...

**Issue #2: req_id=9 has REVERSED path direction**

v5:
```
req_id=9 src=29 dst=11
path=['29', '27', '24', '6', '8', '5', '11']  ← src to dst
```

v6:
```
req_id=9 src=29 dst=11 groomed=False
path=['11', '5', '8', '6', '24', '27', '29']  ← REVERSED! dst to src
```

### 4. Event Count Mismatch
- v5 total: 745 CMP-ALLOC events
- v6 total: 838 CMP-ALLOC events
- Difference: +93 allocations in v6

This explains the modulation statistics failures:
- Different requests allocated/blocked
- Paths in wrong direction → wrong distance calculations
- Wrong distances → wrong modulation formats selected
- Wrong modulations → statistics don't match

## What Needs To Be Done

### 1. Investigate Why req_id=4 is Blocked in v6
**Questions**:
- Why does v5 find a feasible path but v6 doesn't?
- Is v6 checking paths in a different order?
- Is spectrum already allocated differently in v6?

**Added instrumentation**:
```python
[V6-PATH-TRY] req_id=X path_index=N path=[...] path_len=N
[V6-PATH-RESULT] req_id=X path_index=N success=T/F
[V6-BLOCKED] req_id=X src=A dst=B bw=Z reason=R tried_paths=N
```

### 2. Investigate Why Paths are Reversed
**Questions**:
- Where are paths being reversed?
- Is this in routing, spectrum assignment, or grooming?
- Is the reversal intentional or a bug?

**Need to check**:
- Routing algorithm output
- Where `self.sdn_props.path_list` is set
- Whether grooming changes path direction

### 3. Compare Spectrum State Before req_id=4
**Hypothesis**: Perhaps v6 has different spectrum allocations from requests 1-3, making req_id=4's path unavailable

**To verify**:
- Compare spectrum state after req_id=3
- Check if slot assignments differ
- Look at grooming decisions for requests 1-3

### 4. Next Steps
1. Run v5 and v6 with new instrumentation
2. Extract logs for req_id=4 in iteration 1:
   - Which paths were tried
   - Why each path failed
   - Final blocking reason
3. Extract logs for req_id=9 in iteration 1:
   - Check initial path from routing
   - Check if path is reversed during processing
4. Fix the root cause (likely in routing or path storage)

## Files Modified
- `fusion/core/sdn_controller.py`: Added path attempt tracking, blocking details
- `src/sdn_controller.py`: Added enhanced CMP-ALLOC tracking

## Key Insight
The partial grooming fix resolved the CMP-RELEASE mismatch, but revealed a deeper issue: **v6's routing/path selection differs from v5 in iteration 1**. This is not a grooming issue—it's a fundamental routing or spectrum assignment difference.

## Success Metrics
- ✅ Iteration 0: 100% match (212 ALLOC, 212 RELEASE)
- ✅ Partial grooming: Fixed and working
- ❌ Iteration 1: Fundamental routing differences
- ❌ Modulation statistics: Fail due to routing differences
- **Goal**: Make iteration 1 routing match v5 exactly
