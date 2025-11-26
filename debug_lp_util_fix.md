# LP Utilization Dict Fix for V6

## Problem
V6 records rolled-back lightpaths in `lp_bw_utilization_dict`, but V5 does not. This causes test failures when comparing expected results.

## Root Cause
The timing of when lightpaths are added to `lightpath_status_dict` differs between versions:

| Version | Where LP added to dict | When |
|---------|----------------------|------|
| V5 | `engine.py` | AFTER allocation succeeds |
| V6 | `spectrum_assignment.py` | BEFORE SNR recheck |

When SNR recheck fails and triggers a rollback via `release()`:
- **V5**: Lightpath not in dict yet → `release()` finds nothing → no LP_UTIL recorded
- **V6**: Lightpath already in dict → `release()` finds it → LP_UTIL recorded (wrong!)

## Evidence (Request 46)

**V6 output:**
```
[LP_CREATE] req_id=46 lp_id=57    <- Added to lightpath_status_dict
[SNR_ROLLBACK] req_id=46 lp_id=57 <- SNR fails
[LP_UTIL] req_id=46 lp_id=57      <- INCORRECTLY recorded in lp_bw_utilization_dict
[LP_CREATE] req_id=46 lp_id=58
[SNR_ROLLBACK] req_id=46 lp_id=58
[LP_UTIL] req_id=46 lp_id=58      <- INCORRECTLY recorded
[LP_CREATE] req_id=46 lp_id=59    <- Success
```

**V5 output:**
```
[SNR_ROLLBACK] req_id=46 lp_id=57 <- SNR fails (no LP_CREATE before, no LP_UTIL after)
[SNR_ROLLBACK] req_id=46 lp_id=58 <- SNR fails
[LP_CREATE] req_id=46 lp_id=59    <- Success
```

## Fix
In V6's `release()` function, skip recording to `lp_bw_utilization_dict` when `skip_validation=True` (which is set during SNR rollback).

**Location:** `fusion/core/sdn_controller.py` in `_release_lightpath_resources()`

**Change:** Wrap the `lp_bw_utilization_dict.update()` call in a condition that checks `skip_validation`.

## Why This Fix Is Correct
1. `skip_validation=True` is only passed during SNR rollback (line 613)
2. Rolled-back lightpaths never actually served traffic - they shouldn't have utilization stats
3. This makes V6 behavior match V5: only successfully allocated lightpaths get recorded
4. The fix is minimal and targeted - doesn't affect normal release operations
