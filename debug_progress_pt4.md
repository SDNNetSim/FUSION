# FUSION v5→v6 Grooming Debug: Part 4 - Bandwidth Corruption Investigation

## Context

Continuation from Part 3. Investigating why v6 creates bw=100 lightpaths that v5 doesn't.

## Key Finding

**Root Cause Identified**: `sdn_props.bandwidth` is being set to 100 instead of 800 BEFORE first allocation attempt.

## Evidence

Request 10 (800 Gbps):
- First allocation: `allocate_bw=100` (WRONG!)
- Creates phantom lp_id=20 with bw=100
- Later dynamic slicing: `orig_bw=800` (CORRECT!)
- Creates proper lp_id=19,20 with bw=500+300

## Hypothesis

Both v5 and v6 create phantom lightpaths, BUT:
- v5 doesn't record stats for them ✓
- v6 records stats for them ✗

## Investigation Plan

Need to find WHERE `sdn_props.bandwidth` gets changed from 800 to 100.

**Action**: Add instrumentation to track bandwidth modifications:

```python
# In fusion/core/properties.py, update_params method
if key == "bandwidth" and hasattr(self, "request_id"):
    print(f"[V6-BW-SET] req_id={self.request_id}, setting bandwidth={value}")
```

## Status

**Next Step**: Run test with instrumentation to trace bandwidth changes.
