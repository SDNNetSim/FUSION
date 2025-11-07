# Component 4: SDN Controller

**File:** `fusion/core/sdn_controller.py`
**Estimated Time:** 2-3 hours
**Dependencies:** Components 1, 2, 5, 6 (Properties, Grooming, Spectrum, SNR)

## Overview

Integrate grooming functionality into the SDN controller. This is the **most complex component** as it requires modifying the core request handling logic and coordinating between grooming, routing, and spectrum assignment.

## Major Changes Required

### 1. Add Grooming Object to Initialization

In `SDNController.__init__()`:

```python
from fusion.core.grooming import Grooming

def __init__(self, engine_props: dict[str, Any]) -> None:
    self.engine_props = engine_props
    self.sdn_props = SDNProps()

    # Existing objects
    self.ai_obj = None
    self.route_obj = Routing(engine_props=self.engine_props, sdn_props=self.sdn_props)
    self.spectrum_obj = SpectrumAssignment(
        engine_props=self.engine_props,
        sdn_props=self.sdn_props,
        route_props=self.route_obj.route_props,
    )
    self.slicing_manager = LightPathSlicingManager(...)

    # NEW: Add grooming object
    self.grooming_obj = Grooming(
        engine_props=self.engine_props,
        sdn_props=self.sdn_props
    )
```

### 2. Modify release() Method

Update signature and implementation to support lightpath-based release:

```python
def release(self, lightpath_id: int | None = None, slicing_flag: bool = False) -> None:
    """
    Remove a previously allocated request from the network.

    :param lightpath_id: Specific lightpath ID to release (for grooming)
    :type lightpath_id: int | None
    :param slicing_flag: If True, only release spectrum, not transponders
    :type slicing_flag: bool
    """
    if self.sdn_props.path_list is None:
        return

    # Use provided lightpath_id or fall back to request_id
    release_id = lightpath_id if lightpath_id is not None else self.sdn_props.request_id

    for source, dest in zip(
        self.sdn_props.path_list, self.sdn_props.path_list[1:], strict=False
    ):
        for band in self.engine_props["band_list"]:
            for core_num in range(self.engine_props["cores_per_link"]):
                if self.sdn_props.network_spectrum_dict is None:
                    continue

                core_array = self.sdn_props.network_spectrum_dict[(source, dest)][
                    "cores_matrix"
                ][band][core_num]

                # Release using lightpath_id instead of request_id
                request_id_indices = np.where(core_array == release_id)
                guard_band_indices = np.where(core_array == (release_id * -1))

                for request_index in request_id_indices[0]:
                    self.sdn_props.network_spectrum_dict[(source, dest)][
                        "cores_matrix"
                    ][band][core_num][request_index] = 0
                    self.sdn_props.network_spectrum_dict[(dest, source)][
                        "cores_matrix"
                    ][band][core_num][request_index] = 0

                for guard_band_index in guard_band_indices[0]:
                    self.sdn_props.network_spectrum_dict[(source, dest)][
                        "cores_matrix"
                    ][band][core_num][guard_band_index] = 0
                    self.sdn_props.network_spectrum_dict[(dest, source)][
                        "cores_matrix"
                    ][band][core_num][guard_band_index] = 0

    # Throughput calculation (existing code)
    try:
        # ... existing throughput calculation ...
        pass
    except (TypeError, ValueError) as e:
        logger.warning("Throughput update skipped: %s", e)

    # NEW: Handle grooming-specific cleanup
    if not slicing_flag and lightpath_id is not None:
        self._release_lightpath_resources(lightpath_id)


def _release_lightpath_resources(self, lightpath_id: int) -> None:
    """
    Release transponders and update lightpath status dict.

    :param lightpath_id: ID of lightpath to release
    :type lightpath_id: int
    """
    # Always update transponders
    for node in [self.sdn_props.source, self.sdn_props.destination]:
        if node not in self.sdn_props.transponder_usage_dict:
            logger.warning("Node %s not in transponder usage dict", node)
            continue
        self.sdn_props.transponder_usage_dict[node]["available_transponder"] += 1

    light_id = tuple(sorted([self.sdn_props.path_list[0], self.sdn_props.path_list[-1]]))

    # Handle lightpath status dict
    if (light_id in self.sdn_props.lightpath_status_dict and
            lightpath_id in self.sdn_props.lightpath_status_dict[light_id]):

        # Calculate bandwidth utilization stats
        try:
            from fusion.utils.network import average_bandwidth_usage

            average_bw_usage = average_bandwidth_usage(
                bw_dict=self.sdn_props.lightpath_status_dict[light_id][lightpath_id]['time_bw_usage'],
                departure_time=self.sdn_props.depart
            )

            self.sdn_props.lp_bw_utilization_dict.update({
                lightpath_id: {
                    "band": self.sdn_props.lightpath_status_dict[light_id][lightpath_id]['band'],
                    "core": self.sdn_props.lightpath_status_dict[light_id][lightpath_id]["core"],
                    "bit_rate": self.sdn_props.lightpath_status_dict[light_id][lightpath_id]['lightpath_bandwidth'],
                    "utilization": average_bw_usage
                }
            })
        except (TypeError, ValueError, KeyError) as e:
            logger.warning("Average BW update skipped: %s", e)

        # Grooming validation - ensure no active requests
        if (self.sdn_props.lightpath_status_dict[light_id][lightpath_id]['requests_dict'] and
                self.engine_props['is_grooming_enabled']):
            raise ValueError(f'Lightpath {lightpath_id} still has active requests')

        # Remove from status dict
        self.sdn_props.lightpath_status_dict[light_id].pop(lightpath_id)
        logger.debug("Released lightpath %d", lightpath_id)
```

### 3. Modify allocate() Method

Update to use lightpath IDs from spectrum assignment:

```python
def allocate(self) -> None:
    """
    Allocate spectrum resources for the current request.
    """
    # Get spectrum assignment results
    start_slot = self.spectrum_obj.spectrum_props.start_slot
    end_slot = self.spectrum_obj.spectrum_props.end_slot
    core_num = self.spectrum_obj.spectrum_props.core_number
    band = self.spectrum_obj.spectrum_props.current_band
    lightpath_id = self.spectrum_obj.spectrum_props.lightpath_id  # NEW

    if self.engine_props["guard_slots"] != 0:
        end_slot = end_slot - 1

    if (
        self.sdn_props.network_spectrum_dict is None
        or self.sdn_props.path_list is None
    ):
        return

    for source, dest in zip(
        self.sdn_props.path_list, self.sdn_props.path_list[1:], strict=False
    ):
        link_dict = self.sdn_props.network_spectrum_dict[(source, dest)]
        rev_link_dict = self.sdn_props.network_spectrum_dict[(dest, source)]

        core_matrix = link_dict["cores_matrix"]
        rev_core_matrix = rev_link_dict["cores_matrix"]

        # Use lightpath_id instead of request_id
        core_matrix[band][core_num][start_slot:end_slot] = lightpath_id
        rev_core_matrix[band][core_num][start_slot:end_slot] = lightpath_id

        if self.engine_props["guard_slots"]:
            self._allocate_guard_band(
                core_matrix=core_matrix,
                rev_core_matrix=rev_core_matrix,
                end_slot=end_slot,
                core_num=core_num,
                band=band,
                lightpath_id=lightpath_id  # NEW parameter
            )
```

### 4. Modify _allocate_guard_band() Method

Add lightpath_id parameter:

```python
def _allocate_guard_band(
    self,
    band: str,
    core_matrix: dict[str, Any],
    reverse_core_matrix: dict[str, Any],
    core_num: int,
    end_slot: int,
    lightpath_id: int,  # NEW
) -> None:
    """
    Allocate guard band slots for spectrum isolation.

    :param lightpath_id: Lightpath ID to use for guard band marking
    :type lightpath_id: int
    """
    if (core_matrix[band][core_num][end_slot] != 0.0 or
        reverse_core_matrix[band][core_num][end_slot] != 0.0):
        raise BufferError("Attempted to allocate a taken spectrum.")

    # Use lightpath_id with negative sign for guard bands
    core_matrix[band][core_num][end_slot] = lightpath_id * -1
    reverse_core_matrix[band][core_num][end_slot] = lightpath_id * -1
```

### 5. Add Grooming to Request Handling

Modify the main allocation method to check grooming first:

```python
def allocate_request(
    self,
    force_route_matrix: list | None = None,
    force_mod_format: list | None = None,
) -> None:
    """
    Main entry point for request allocation.

    Attempts grooming first if enabled, then falls back to
    standard routing and spectrum assignment.
    """
    request_type = self.sdn_props.request_type

    # Handle release requests
    if request_type == "release":
        if self.engine_props['is_grooming_enabled']:
            lightpath_id_list = self.grooming_obj.handle_grooming(request_type)
        else:
            lightpath_id_list = self.sdn_props.lightpath_id_list if hasattr(self.sdn_props, 'lightpath_id_list') else [None]

        for lightpath_id in lightpath_id_list:
            self.release(lightpath_id=lightpath_id)
        return

    # Initialize request statistics
    self._init_req_stats()
    self.sdn_props.number_of_transponders = 1

    # NEW: Try grooming first if enabled
    if self.engine_props['is_grooming_enabled']:
        self.grooming_obj.lightpath_status_dict = self.sdn_props.lightpath_status_dict
        groom_result = self.grooming_obj.handle_grooming(request_type)

        if groom_result:
            # Fully groomed - done!
            return
        else:
            # Not groomed or partially groomed
            self.sdn_props.was_new_lp_established = []

            if self.sdn_props.was_partially_groomed:
                # Force route on same path as groomed portion
                force_route_matrix = [self.sdn_props.path_list]

                # Get modulation formats for remaining bandwidth
                from fusion.utils.data import sort_nested_dict_values
                mod_formats_dict = sort_nested_dict_values(
                    original_dict=self.sdn_props.modulation_formats_dict,
                    nested_key='max_length'
                )
                force_mod_format = [list(mod_formats_dict.keys())]

    # Continue with normal routing/spectrum assignment
    start_time = time.time()

    if force_route_matrix is None:
        self.route_obj.get_route()
        route_matrix = self.route_obj.route_props.paths_matrix
    else:
        route_matrix = force_route_matrix

    # ... rest of existing allocation logic ...
```

### 6. Add SNR Rechecking

```python
def _check_snr_after_allocation(self, lightpath_id: int) -> bool:
    """
    Recheck SNR after spectrum allocation (for grooming).

    :param lightpath_id: ID of newly allocated lightpath
    :type lightpath_id: int
    :return: True if SNR is acceptable, False otherwise
    :rtype: bool
    """
    if not self.engine_props.get('snr_recheck', False):
        return True

    from fusion.core.snr_measurements import SnrMeasurements

    snr_obj = SnrMeasurements(
        engine_props=self.engine_props,
        sdn_props=self.sdn_props,
        route_props=self.route_obj.route_props
    )

    # Recheck SNR with allocated spectrum
    snr_acceptable, _ = snr_obj.recheck_snr_after_allocation(lightpath_id)

    return snr_acceptable
```

### 7. Handle Partial Allocation Failures

```python
def _handle_congestion_with_grooming(self, remaining_bw: int) -> None:
    """
    Handle allocation failure with grooming rollback.

    If partial grooming occurred but remaining bandwidth cannot be allocated,
    rollback the newly created lightpaths but keep the groomed portion.

    :param remaining_bw: Remaining bandwidth that could not be allocated
    :type remaining_bw: int
    """
    if remaining_bw != int(self.sdn_props.bandwidth):
        # Rollback newly established lightpaths
        for lpid in list(self.sdn_props.was_new_lp_established):
            self.release(lightpath_id=lpid, slicing_flag=True)
            self.sdn_props.was_new_lp_established.remove(lpid)

            # Remove from tracking lists
            lp_idx = self.sdn_props.lightpath_id_list.index(lpid)
            self.sdn_props.lightpath_id_list.pop(lp_idx)
            self.sdn_props.lightpath_bandwidth_list.pop(lp_idx)
            self.sdn_props.start_slot_list.pop(lp_idx)
            self.sdn_props.band_list.pop(lp_idx)
            self.sdn_props.core_list.pop(lp_idx)
            self.sdn_props.end_slot_list.pop(lp_idx)
            self.sdn_props.crosstalk_list.pop(lp_idx)
            self.sdn_props.bandwidth_list.pop(lp_idx)
            self.sdn_props.modulation_list.pop(lp_idx)

    self.sdn_props.number_of_transponders = 1
    self.sdn_props.is_sliced = False
    self.sdn_props.was_partially_routed = False

    if self.sdn_props.was_partially_groomed:
        self.sdn_props.remaining_bw = self.sdn_props.lightpath_bandwidth_list
    else:
        self.sdn_props.remaining_bw = int(self.sdn_props.bandwidth)

    self.sdn_props.was_new_lp_established = []
```

## Testing

Create comprehensive tests in `fusion/core/tests/test_sdn_controller.py`:

```python
def test_sdn_controller_grooming_init(engine_props):
    """Test SDN controller has grooming object."""
    sdn = SDNController(engine_props)
    assert hasattr(sdn, 'grooming_obj')
    assert sdn.grooming_obj is not None


def test_release_with_lightpath_id(engine_props, setup_network):
    """Test release with specific lightpath ID."""
    sdn = SDNController(engine_props)
    sdn.sdn_props.network_spectrum_dict = setup_network
    # ... test release logic ...
```

Run tests:

```bash
python -m pylint fusion/core/sdn_controller.py
python -m mypy fusion/core/sdn_controller.py
python -m pytest fusion/core/tests/test_sdn_controller.py -v -k grooming
```

## Validation Checklist

- [ ] Grooming object added to __init__
- [ ] release() method accepts lightpath_id and slicing_flag
- [ ] _release_lightpath_resources() method added
- [ ] allocate() uses lightpath_id from spectrum props
- [ ] _allocate_guard_band() accepts lightpath_id
- [ ] Grooming check added to request handling
- [ ] SNR rechecking implemented
- [ ] Partial allocation rollback implemented
- [ ] All methods have proper type hints
- [ ] Code passes pylint and mypy
- [ ] Unit tests created and passing
- [ ] Integration tests pass

## Next Component

After completing this component, proceed to: [Component 5: Spectrum Assignment](05-spectrum.md)
