# FUSION v5→v6 Grooming Debug: Part 6 - Request Dictionary Key Structure Fix

## Session Summary

Identified and fixed fundamental architectural difference in request event storage between v5 and v6.

---

## Root Cause Identified

**Issue**: v5 and v6 use different key structures for `reqs_dict`, causing request loss and divergent behavior.

### Evidence

**v5 Architecture:**
- Dict keys: `(request_id, time)` tuples
- Example: `(10, 21.952...)` for request 10 at time 21.952
- Sorting: `sorted(..., key=lambda x: x[0][1])` - sorts by time (second element)
- **Allows multiple events at same simulation time**

**v6 Architecture (before fix):**
- Dict keys: `float` (time only)
- Example: `21.952...` for event at time 21.952
- Sorting: `sorted(...)` - sorts by time directly
- **Cannot handle multiple events at same time - later events OVERWRITE earlier ones**

### Proof of Request Loss

From analysis of generated requests:

**v5:**
- 800 unique arrival times
- 831 total requests
- **31 requests share arrival times** (from different iterations)

**v6 (before fix):**
- 1200 unique times
- 1200 total requests
- **NO duplicates** - each overwritten event is lost

Example: Request at time `4.650655818775675`
- v5: Shows 2 instances (different iterations)
- v6: Shows 1 instance (second overwrites first)

---

## Fix Implemented

### 1. Request Generation (`fusion/core/request.py`)

**Changed:**
```python
# OLD - float keys
requests_dict[arrival_time] = {...}
requests_dict[departure_time] = {...}

# NEW - tuple keys
requests_dict[(request_id, arrival_time)] = {...}
requests_dict[(request_id, departure_time)] = {...}
```

**Type hints updated:**
```python
# OLD
def generate_simulation_requests(...) -> dict[float, dict[str, Any]]:

# NEW
def generate_simulation_requests(...) -> dict[tuple[int, float], dict[str, Any]]:
```

**Collision check removed:**
- No longer needed since tuple keys allow simultaneous events
- Removed retry logic that was consuming extra random numbers

### 2. Simulation (`fusion/core/simulation.py`)

**Sorting updated:**
```python
# OLD
self.reqs_dict = dict(sorted(self.reqs_dict.items()))

# NEW - match v5 behavior
self.reqs_dict = dict(sorted(self.reqs_dict.items(), key=lambda x: x[0][1]))
```

**Type signatures updated:**
```python
# Method signatures changed from:
def handle_request(self, current_time: float, ...) -> None:
def handle_arrival(self, current_time: float, ...) -> None:
def handle_release(self, current_time: float) -> None:

# To:
def handle_request(self, current_time: tuple[int, float], ...) -> None:
def handle_arrival(self, current_time: tuple[int, float], ...) -> None:
def handle_release(self, current_time: tuple[int, float]) -> None:
```

---

## Test Results After Fix

**Status**: Tests still failing, but with different errors

### Previous Failures (Part 5):
- Demand realization: 90% mean, some requests only 25% served
- Request loss due to time collisions

### Current Failures (Part 6):
- Modulation format mismatches:
  - 64-QAM: expected 3, got 0
  - 32-QAM: expected 3, got 6
  - 16-QAM: expected 55, got 60
- Demand realization: 94% mean (improved from 90%)
- Overall min: 0.2 (some requests still poorly served)

**Analysis**: Tuple key fix prevented request loss, but modulation selection differs. This suggests:
- Events now processed in same quantity
- But event ordering or grooming decisions may still differ
- Need to verify event processing order matches v5

---

## Current Investigation

### Hypothesis

Event processing order may differ due to:
1. Different secondary sort order when times are equal
2. Different grooming state at time of allocation
3. Subtle differences in how tuple keys are iterated

### Debug Instrumentation Added

**Both branches now print ALL events:**

v5 (`src/engine.py`):
```python
print(f"[V5-EVENT-ORDER] req_num={req_num}, event_key={curr_time},
      req_id={...}, type={...}, bw={...}")
```

v6 (`fusion/core/simulation.py`):
```python
print(f"[V6-EVENT-ORDER] req_num={request_number}, event_key={current_time},
      req_id={...}, type={...}, bw={...}")
```

---

## Next Steps

1. **Compare event ordering** between v5 and v6:
   - Extract `[V5-EVENT-ORDER]` and `[V6-EVENT-ORDER]` from logs
   - Diff to find first divergence point
   - Identify why events are processed differently

2. **If ordering matches**, investigate:
   - Grooming decision differences
   - Spectrum availability at allocation time
   - Modulation format selection logic

3. **If ordering differs**, investigate:
   - Tuple sorting behavior
   - Iterator behavior over dict with tuple keys
   - Any remaining type mismatches

---

## Files Modified

1. `fusion/core/request.py`:
   - Lines 285-317: Changed to tuple keys, removed collision check
   - Line 159: Updated return type hint
   - Line 324: Updated backward-compat function type hint

2. `fusion/core/simulation.py`:
   - Line 607-608: Updated sorting with key function
   - Line 610: Updated `handle_request` signature
   - Line 339: Updated `handle_arrival` signature
   - Line 394: Updated `handle_release` signature
   - Line 623: Added debug event ordering print

---

## Important Notes

**Changes Lost in Stash**:
- Tuple key changes were accidentally reverted during git stash operation
- Need to re-apply fixes before testing

**Priority**:
- Re-apply tuple key fixes
- Run comparison test with event ordering debug enabled
- Analyze event order differences

---

## Success Criteria

- [ ] All events processed in same order as v5
- [ ] No request loss (all 831 requests from v5 present in v6)
- [ ] Modulation format distributions match v5
- [ ] Demand realization ratio 100% (1.0)
- [ ] Comparison test passes
