# Component 5: Spectrum Assignment

**File:** `fusion/core/spectrum_assignment.py`
**Estimated Time:** 1 hour
**Dependencies:** Components 1, 2 (Properties, Grooming)

## Overview

Modify spectrum assignment to track lightpath IDs and handle partial grooming bandwidth calculations.

## Changes Required

### 1. Add Lightpath ID Generation

In the `get_spectrum()` method, generate and assign lightpath IDs:

```python
def get_spectrum(self, mod_format_list: list, slice_bandwidth: str | None = None) -> None:
    """
    Attempt to find available spectrum for the request.

    :param mod_format_list: List of modulation formats to try
    :type mod_format_list: list
    :param slice_bandwidth: Specific bandwidth for slicing (optional)
    :type slice_bandwidth: str | None
    """
    # ... existing spectrum search logic ...

    if self.spectrum_props.is_free:
        # NEW: Generate lightpath ID
        lp_id = self.sdn_props.get_lightpath_id()
        self.spectrum_props.lightpath_id = lp_id

        # Mark that a new lightpath was established
        if self.engine_props.get('is_grooming_enabled', False):
            self.sdn_props.was_new_lp_established.append(lp_id)

        logger.debug("Assigned lightpath ID %d to request %d", lp_id, self.sdn_props.request_id)
```

### 2. Handle Partial Grooming Bandwidth

When calculating slots needed for partially groomed requests:

```python
def _calculate_slots_needed(self, modulation: str, slice_bandwidth: str | None = None) -> int:
    """
    Calculate slots needed for modulation and bandwidth.

    Handles special case for partial grooming where remaining bandwidth
    needs to be rounded up to the next available bandwidth tier.

    :param modulation: Modulation format
    :type modulation: str
    :param slice_bandwidth: Bandwidth for slicing
    :type slice_bandwidth: str | None
    :return: Number of slots needed
    :rtype: int
    """
    if self.engine_props['fixed_grid']:
        return 1

    # NEW: Handle partial grooming
    if self.sdn_props.was_partially_groomed:
        remaining_bw = int(self.sdn_props.remaining_bw)

        # Find next higher bandwidth tier
        available_bw_tiers = [
            int(k) for k in self.engine_props['mod_per_bw'].keys()
            if int(k) >= remaining_bw
        ]

        if not available_bw_tiers:
            logger.warning(
                "No bandwidth tier available for remaining bandwidth %d",
                remaining_bw
            )
            return None

        bw_tmp = min(available_bw_tiers)
        slots_needed = self.engine_props['mod_per_bw'][str(bw_tmp)][modulation]['slots_needed']

        logger.debug(
            "Partial grooming: remaining_bw=%d, using bw_tier=%d, slots=%d",
            remaining_bw, bw_tmp, slots_needed
        )

        return slots_needed

    # Standard case
    if slice_bandwidth:
        return self.engine_props['mod_per_bw'][slice_bandwidth][modulation]['slots_needed']

    return self.sdn_props.modulation_formats_dict[modulation]['slots_needed']
```

### 3. Update SNR Handling

Modify SNR checks to return lightpath bandwidth:

```python
def get_spectrum(self, mod_format_list: list, slice_bandwidth: str | None = None) -> None:
    """Find available spectrum."""
    # ... existing code ...

    for modulation in mod_format_list:
        # Calculate slots
        self.spectrum_props.slots_needed = self._calculate_slots_needed(
            modulation, slice_bandwidth
        )

        if self.spectrum_props.slots_needed is None:
            continue

        # Check for free spectrum
        self._check_spectrum_availability()

        if self.spectrum_props.is_free:
            self.spectrum_props.modulation = modulation

            # NEW: Get lightpath bandwidth from SNR check
            if self.engine_props['snr_type'] not in ['None', None]:
                snr_check, xt_cost, lp_bw = self.snr_obj.handle_snr(
                    self.sdn_props.path_index
                )

                self.spectrum_props.crosstalk_cost = xt_cost
                self.spectrum_props.lightpath_bandwidth = lp_bw  # NEW

                if not snr_check:
                    self.spectrum_props.is_free = False
                    self.sdn_props.block_reason = 'xt_threshold'
                    continue

                self.spectrum_props.is_free = True
                self.sdn_props.block_reason = None

            return
```

### 4. Update Lightpath Status Dictionary

After successful allocation, populate the lightpath status dictionary:

```python
def _update_lightpath_status(self) -> None:
    """
    Update lightpath status dictionary after allocation.

    Called after spectrum is allocated to track the new lightpath
    for future grooming operations.
    """
    if not self.engine_props.get('is_grooming_enabled', False):
        return

    light_id = tuple(sorted([self.sdn_props.source, self.sdn_props.destination]))
    lp_id = self.spectrum_props.lightpath_id

    # Initialize light_id entry if needed
    if light_id not in self.sdn_props.lightpath_status_dict:
        self.sdn_props.lightpath_status_dict[light_id] = {}

    # Get lightpath bandwidth
    lp_bandwidth = self.spectrum_props.lightpath_bandwidth
    if lp_bandwidth is None:
        # Calculate from modulation and slots if not set by SNR
        mod = self.spectrum_props.modulation
        lp_bandwidth = self.sdn_props.modulation_formats_dict[mod].get('bandwidth', 0)

    # Create lightpath entry
    self.sdn_props.lightpath_status_dict[light_id][lp_id] = {
        "path": self.sdn_props.path_list,
        "path_weight": self.sdn_props.path_weight,
        "core": self.spectrum_props.core_number,
        "band": self.spectrum_props.current_band,
        "start_slot": self.spectrum_props.start_slot,
        "end_slot": self.spectrum_props.end_slot,
        "mod_format": self.spectrum_props.modulation,
        "lightpath_bandwidth": lp_bandwidth,
        "remaining_bandwidth": lp_bandwidth,  # Initially all available
        "snr_cost": self.spectrum_props.crosstalk_cost,
        "is_degraded": False,
        "requests_dict": {},  # Will be populated when requests are groomed
        "time_bw_usage": {self.sdn_props.arrive: 0.0}  # Track utilization over time
    }

    logger.debug(
        "Created lightpath status entry: light_id=%s, lp_id=%d, bw=%s",
        light_id, lp_id, lp_bandwidth
    )
```

Call this method after allocation:

```python
def get_spectrum(self, mod_format_list: list, slice_bandwidth: str | None = None) -> None:
    """Find available spectrum."""
    # ... allocation logic ...

    if self.spectrum_props.is_free:
        # Generate lightpath ID
        lp_id = self.sdn_props.get_lightpath_id()
        self.spectrum_props.lightpath_id = lp_id

        # NEW: Update lightpath status for grooming
        self._update_lightpath_status()

        return
```

## Testing

Create tests in `fusion/core/tests/test_spectrum_assignment.py`:

```python
def test_lightpath_id_generation(engine_props, sdn_props):
    """Test lightpath ID is generated for allocations."""
    spec = SpectrumAssignment(engine_props, sdn_props, route_props)

    # Setup for allocation
    sdn_props.path_list = [1, 2, 3]
    sdn_props.path_index = 0

    spec.get_spectrum(mod_format_list=['QPSK'])

    if spec.spectrum_props.is_free:
        assert spec.spectrum_props.lightpath_id is not None
        assert spec.spectrum_props.lightpath_id > 0


def test_partial_grooming_bandwidth_calculation(engine_props, sdn_props):
    """Test bandwidth calculation for partial grooming."""
    engine_props['is_grooming_enabled'] = True
    sdn_props.was_partially_groomed = True
    sdn_props.remaining_bw = 150  # Needs rounding up

    spec = SpectrumAssignment(engine_props, sdn_props, route_props)
    slots = spec._calculate_slots_needed('QPSK')

    assert slots is not None
    assert slots > 0


def test_lightpath_status_update(engine_props, sdn_props):
    """Test lightpath status dict is populated."""
    engine_props['is_grooming_enabled'] = True
    sdn_props.lightpath_status_dict = {}
    sdn_props.source = 'A'
    sdn_props.destination = 'B'

    spec = SpectrumAssignment(engine_props, sdn_props, route_props)
    spec.spectrum_props.lightpath_id = 1
    spec.spectrum_props.lightpath_bandwidth = 200

    spec._update_lightpath_status()

    light_id = ('A', 'B')
    assert light_id in sdn_props.lightpath_status_dict
    assert 1 in sdn_props.lightpath_status_dict[light_id]
```

Run tests:

```bash
python -m pylint fusion/core/spectrum_assignment.py
python -m mypy fusion/core/spectrum_assignment.py
python -m pytest fusion/core/tests/test_spectrum_assignment.py -v -k grooming
```

## Validation Checklist

- [ ] Lightpath ID generation added
- [ ] `_calculate_slots_needed()` handles partial grooming
- [ ] SNR methods return lightpath bandwidth
- [ ] `_update_lightpath_status()` method added
- [ ] Lightpath status dict populated after allocation
- [ ] was_new_lp_established list updated
- [ ] All methods have proper type hints
- [ ] Logging added for debugging
- [ ] Code passes pylint and mypy
- [ ] Unit tests created and passing

## Next Component

After completing this component, proceed to: [Component 6: SNR Measurements](06-snr.md)
