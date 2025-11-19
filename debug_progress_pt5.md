# FUSION v5→v6 Grooming Debug: Part 5 - Bandwidth Corruption Fix

## Session Summary

Fixed critical bandwidth corruption bug causing phantom 100 Gbps lightpaths for 800 Gbps requests.

---

## Investigation Process

### 1. Added Bandwidth Tracking Instrumentation

**File**: `fusion/core/properties.py:465`

```python
if key == "bandwidth" and hasattr(self, "request_id"):
    print(f"[V6-BW-SET] req_id={self.request_id}, setting bandwidth={value}")
```

### 2. Analyzed Bandwidth Timeline

**Discovery**: Request 10 bandwidth set MULTIPLE times:
- `[V6-BW-SET] req_id=10, setting bandwidth=100` ← WRONG (first)
- `[V6-BW-SET] req_id=10, setting bandwidth=800` ← CORRECT (later)

### 3. Identified Corruption Source

**Pattern Found**:
```
[V6-BW-SET] req_id=10, setting bandwidth=100
[DEBUG-LP-RELEASE-START] req_id=10...  ← Happens during RELEASE!
```

**Root Cause**: `fusion/core/simulation.py:1329`
```python
self.sdn.sdn_props.bandwidth = request_info.get("bandwidth")  # BUG!
```

During request RELEASE, bandwidth was set from stored `request_info`, which contained:
- ALLOCATED bandwidth (100) from lightpath
- NOT original REQUEST bandwidth (800)

This corrupted `sdn_props.bandwidth` for subsequent operations.

---

## Fix Applied

**File**: `fusion/core/simulation.py:1329-1332`

**Before**:
```python
self.sdn.sdn_props.bandwidth = request_info.get("bandwidth")
```

**After**:
```python
# NOTE: Do NOT set bandwidth from request_info during release - it may contain
# allocated bandwidth (e.g., 100) instead of original request bandwidth (e.g., 800),
# which corrupts sdn_props.bandwidth for subsequent operations.
# self.sdn.sdn_props.bandwidth = request_info.get("bandwidth")
```

**Rationale**: Release operations only need to free spectrum resources, not modify request bandwidth.

---

## Expected Impact

1. ✓ No more phantom 100 Gbps lightpaths for 800 Gbps requests
2. ✓ Request bandwidth stays at original value throughout processing
3. ✓ Correct modulation selections (no more 32-QAM override)
4. ✓ v5/v6 statistics alignment

---

## Files Modified

1. `fusion/core/properties.py:465` - Added bandwidth tracking debug print
2. `fusion/core/simulation.py:1329-1332` - Commented out bandwidth corruption line

---

## Next Steps

1. **Re-run comparison test**:
   ```bash
   python tests/run_comparison.py --test-case spain_C_fixed_grooming > v6.txt 2>&1
   ```

2. **Verify fixes**:
   - Check no `[V6-BW-SET]` during release operations
   - Confirm request 10 only gets `bandwidth=800`
   - Verify no phantom 100 Gbps lightpaths created
   - Check test comparison passes

3. **If test passes**:
   - Remove debug instrumentation from properties.py
   - Commit final fix
   - Run full test suite

4. **If test still fails**:
   - Analyze remaining discrepancies
   - Check for other corruption points

---

## Success Criteria

- `spain_C_fixed_grooming` comparison test passes
- No phantom lightpaths in statistics
- `weights_dict.*.32-QAM` populated correctly
- All modulation distributions match v5
