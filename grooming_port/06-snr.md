# Component 6: SNR Measurements

**File:** `fusion/core/snr_measurements.py`
**Estimated Time:** 1 hour
**Dependencies:** Component 1 (Properties)

## Overview

Modify SNR measurement methods to:
1. Return lightpath bandwidth alongside SNR values
2. Add SNR rechecking after spectrum allocation (for grooming)
3. Support adjacent core and cross-band interference checks

## Changes Required

### 1. Modify handle_snr() Return Value

Update the method signature to return bandwidth:

```python
def handle_snr(self, path_index: int | None = None) -> tuple[bool, float, float]:
    """
    Perform SNR calculation and modulation validation.

    :param path_index: Index of path to check
    :type path_index: int | None
    :return: Tuple of (snr_acceptable, crosstalk_cost, lightpath_bandwidth)
    :rtype: tuple[bool, float, float]
    """
    # ... existing SNR calculation ...

    # Calculate lightpath bandwidth based on modulation
    modulation = self.spectrum_props.modulation
    if modulation in self.snr_props.bandwidth_mapping_dict:
        lightpath_bandwidth = self.snr_props.bandwidth_mapping_dict[modulation]
    else:
        # Fall back to request bandwidth if mapping not found
        lightpath_bandwidth = float(self.sdn_props.bandwidth)

    # Return SNR result, crosstalk cost, and bandwidth
    return snr_acceptable, crosstalk_cost, lightpath_bandwidth
```

### 2. Add SNR Rechecking Method

Add new method for rechecking SNR after allocation:

```python
def recheck_snr_after_allocation(self, lightpath_id: int) -> tuple[bool, float]:
    """
    Recheck SNR after spectrum allocation.

    When grooming is enabled, validates that newly allocated spectrum
    still meets SNR requirements after considering all interference.

    :param lightpath_id: ID of newly allocated lightpath
    :type lightpath_id: int
    :return: Tuple of (snr_acceptable, crosstalk_cost)
    :rtype: tuple[bool, float]
    """
    if not self.engine_props.get('snr_recheck', False):
        # Rechecking disabled, assume OK
        return True, 0.0

    logger.debug("Rechecking SNR for lightpath %d", lightpath_id)

    # Get allocated spectrum details
    start_slot = self.spectrum_props.start_slot
    end_slot = self.spectrum_props.end_slot
    core_num = self.spectrum_props.core_number
    band = self.spectrum_props.current_band
    path_list = self.sdn_props.path_list

    # Calculate interference from adjacent cores
    adjacent_core_interference = 0.0
    if self.engine_props.get('recheck_adjacent_cores', False):
        adjacent_core_interference = self._calculate_adjacent_core_interference(
            path_list, band, core_num, start_slot, end_slot
        )

    # Calculate cross-band interference
    crossband_interference = 0.0
    if self.engine_props.get('recheck_crossband', False):
        crossband_interference = self._calculate_crossband_interference(
            path_list, core_num, start_slot, end_slot
        )

    # Total interference
    total_interference = adjacent_core_interference + crossband_interference

    # Re-calculate SNR with interference
    snr_margin = self._calculate_snr_with_interference(total_interference)

    # Check if SNR is still acceptable
    required_snr = self.snr_props.request_snr
    snr_acceptable = snr_margin >= required_snr

    if not snr_acceptable:
        logger.warning(
            "SNR recheck failed for lightpath %d: margin=%.2f < required=%.2f",
            lightpath_id, snr_margin, required_snr
        )

    return snr_acceptable, total_interference


def _calculate_adjacent_core_interference(
    self,
    path_list: list,
    band: str,
    core_num: int,
    start_slot: int,
    end_slot: int
) -> float:
    """
    Calculate interference from adjacent cores.

    :param path_list: Path nodes
    :type path_list: list
    :param band: Spectral band
    :type band: str
    :param core_num: Core number
    :type core_num: int
    :param start_slot: Start slot index
    :type start_slot: int
    :param end_slot: End slot index
    :type end_slot: int
    :return: Adjacent core interference value
    :rtype: float
    """
    interference = 0.0

    # Get adjacent cores (depends on core layout)
    adjacent_cores = self._get_adjacent_cores(core_num)

    for source, dest in zip(path_list, path_list[1:], strict=False):
        link_dict = self.sdn_props.network_spectrum_dict[(source, dest)]

        for adj_core in adjacent_cores:
            if adj_core >= self.engine_props['cores_per_link']:
                continue

            core_array = link_dict['cores_matrix'][band][adj_core]

            # Check for occupied slots in adjacent core
            occupied_slots = core_array[start_slot:end_slot]
            if np.any(occupied_slots != 0):
                # Add interference (simplified - use actual crosstalk model)
                interference += self.engine_props.get('adjacent_core_xt_coefficient', 0.01)

    return interference


def _calculate_crossband_interference(
    self,
    path_list: list,
    core_num: int,
    start_slot: int,
    end_slot: int
) -> float:
    """
    Calculate interference from other spectral bands.

    :param path_list: Path nodes
    :type path_list: list
    :param core_num: Core number
    :type core_num: int
    :param start_slot: Start slot index
    :type start_slot: int
    :param end_slot: End slot index
    :type end_slot: int
    :return: Cross-band interference value
    :rtype: float
    """
    interference = 0.0

    current_band = self.spectrum_props.current_band
    other_bands = [b for b in self.engine_props['band_list'] if b != current_band]

    for source, dest in zip(path_list, path_list[1:], strict=False):
        link_dict = self.sdn_props.network_spectrum_dict[(source, dest)]

        for band in other_bands:
            core_array = link_dict['cores_matrix'][band][core_num]

            # Check for occupied slots in other band
            occupied_slots = core_array[start_slot:end_slot]
            if np.any(occupied_slots != 0):
                # Add interference (simplified - use actual crosstalk model)
                interference += self.engine_props.get('crossband_xt_coefficient', 0.005)

    return interference


def _get_adjacent_cores(self, core_num: int) -> list[int]:
    """
    Get list of cores adjacent to the given core.

    :param core_num: Core number
    :type core_num: int
    :return: List of adjacent core numbers
    :rtype: list[int]
    """
    # Simplified adjacency - actual adjacency depends on fiber geometry
    total_cores = self.engine_props['cores_per_link']

    if total_cores == 1:
        return []

    # Simple linear adjacency model
    adjacent = []
    if core_num > 0:
        adjacent.append(core_num - 1)
    if core_num < total_cores - 1:
        adjacent.append(core_num + 1)

    return adjacent


def _calculate_snr_with_interference(self, interference: float) -> float:
    """
    Calculate SNR including additional interference.

    :param interference: Additional interference value
    :type interference: float
    :return: SNR margin in dB
    :rtype: float
    """
    # Get base SNR calculation
    # ... existing SNR calculation logic ...

    # Adjust for additional interference
    # (simplified - actual calculation depends on SNR model)
    base_snr = 20.0  # Example base SNR in dB
    interference_db = 10 * np.log10(1 + interference) if interference > 0 else 0

    adjusted_snr = base_snr - interference_db

    return adjusted_snr
```

### 3. Update Existing SNR Methods

Ensure all SNR calculation methods return the bandwidth:

```python
def handle_snr_dynamic_slicing(self, path_index: int) -> tuple[str | None, float, float]:
    """
    Handle SNR for dynamic slicing scenarios.

    :param path_index: Path index
    :type path_index: int
    :return: Tuple of (modulation_format, bandwidth, snr_cost)
    :rtype: tuple[str | None, float, float]
    """
    # ... existing logic ...

    if snr_acceptable:
        return modulation, bandwidth, snr_cost
    else:
        return None, 0.0, 0.0
```

## Testing

Create tests in `fusion/core/tests/test_snr_measurements.py`:

```python
def test_handle_snr_returns_bandwidth(engine_props, sdn_props, route_props):
    """Test that handle_snr returns bandwidth."""
    snr = SnrMeasurements(engine_props, sdn_props, route_props)

    # Setup
    snr.spectrum_props.modulation = 'QPSK'

    snr_ok, xt_cost, lp_bw = snr.handle_snr(path_index=0)

    assert isinstance(snr_ok, bool)
    assert isinstance(xt_cost, float)
    assert isinstance(lp_bw, float)
    assert lp_bw > 0


def test_snr_recheck_after_allocation(engine_props, sdn_props, route_props):
    """Test SNR rechecking."""
    engine_props['snr_recheck'] = True
    snr = SnrMeasurements(engine_props, sdn_props, route_props)

    # Setup allocated lightpath
    snr.spectrum_props.lightpath_id = 1
    snr.spectrum_props.start_slot = 10
    snr.spectrum_props.end_slot = 20

    snr_ok, interference = snr.recheck_snr_after_allocation(lightpath_id=1)

    assert isinstance(snr_ok, bool)
    assert isinstance(interference, float)


def test_adjacent_core_interference(engine_props, sdn_props, route_props):
    """Test adjacent core interference calculation."""
    engine_props['recheck_adjacent_cores'] = True
    engine_props['cores_per_link'] = 7

    snr = SnrMeasurements(engine_props, sdn_props, route_props)

    interference = snr._calculate_adjacent_core_interference(
        path_list=[1, 2, 3],
        band='C',
        core_num=3,
        start_slot=10,
        end_slot=20
    )

    assert isinstance(interference, float)
    assert interference >= 0
```

Run tests:

```bash
python -m pylint fusion/core/snr_measurements.py
python -m mypy fusion/core/snr_measurements.py
python -m pytest fusion/core/tests/test_snr_measurements.py -v
```

## Validation Checklist

- [ ] `handle_snr()` returns tuple with bandwidth
- [ ] `recheck_snr_after_allocation()` method added
- [ ] Adjacent core interference calculation implemented
- [ ] Cross-band interference calculation implemented
- [ ] SNR rechecking respects config flags
- [ ] All methods have proper type hints
- [ ] Logging added for debugging
- [ ] Code passes pylint and mypy
- [ ] Unit tests created and passing

## Next Component

After completing this component, proceed to: [Component 7: Helper Functions](07-helpers.md)
