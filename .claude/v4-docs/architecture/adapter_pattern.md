# Adapter Pattern in V4 Architecture

This document describes the adapter pattern used to bridge legacy FUSION code with the new V4 pipeline architecture.

## Overview

Adapters wrap legacy classes to implement new pipeline protocols, enabling gradual migration without breaking existing functionality.

```
Legacy Code                    Adapter                       New Interface
+---------------+         +-------------------+         +------------------+
| Routing class |  <---   | RoutingAdapter    |  --->   | RoutingPipeline  |
| (dict-based)  |         | (translates)      |         | (typed protocol) |
+---------------+         +-------------------+         +------------------+
```

## Purpose

### Why Use Adapters?

1. **Gradual Migration**: Migrate one component at a time without rewriting everything

2. **Risk Reduction**: Legacy behavior preserved; new interface tested independently

3. **Parallel Development**: Teams can work on new pipelines while adapters keep system running

4. **Validation**: Compare adapter output with native pipeline output for verification

### Adapter Lifecycle

```
Phase 2: Create adapters wrapping legacy code
    |
Phase 3: Use adapters in SDNOrchestrator
    |
Phase 4: Develop native pipeline implementations
    |
Phase 5: Replace adapters with native pipelines
    |
Phase 6: Remove adapters and legacy code
```

---

## Adapter Types

| Adapter | Wraps | Implements |
|---------|-------|------------|
| `RoutingAdapter` | `Routing` class | `RoutingPipeline` protocol |
| `SpectrumAdapter` | Spectrum assignment functions | `SpectrumPipeline` protocol |
| `GroomingAdapter` | Grooming module | `GroomingPipeline` protocol |
| `SNRAdapter` | SNR calculation classes | `SNRPipeline` protocol |

---

## RoutingAdapter

### File: `fusion/core/adapters/routing_adapter.py`

```python
"""Adapter wrapping legacy Routing class."""

from __future__ import annotations

from fusion.domain.config import SimulationConfig
from fusion.domain.network_state import NetworkState
from fusion.domain.results import RouteResult
from fusion.routing.routing import Routing  # Legacy class


class RoutingAdapter:
    """
    Adapts legacy Routing class to RoutingPipeline protocol.

    This adapter:
    1. Receives typed parameters (NetworkState, bandwidth_gbps)
    2. Translates to legacy format (engine_props dict)
    3. Calls legacy Routing methods
    4. Translates results to RouteResult
    """

    def __init__(self, config: SimulationConfig):
        """
        Initialize adapter with configuration.

        Args:
            config: Simulation configuration
        """
        self.config = config
        # Legacy routing needs engine_props format
        self._legacy_routing = Routing(self._build_legacy_props())

    def find_routes(
        self,
        source: str,
        destination: str,
        bandwidth_gbps: int,
        network_state: NetworkState,
        forced_path: list[str] | None = None,
    ) -> RouteResult:
        """
        Find routes using legacy routing class.

        Args:
            source: Source node ID
            destination: Destination node ID
            bandwidth_gbps: Required bandwidth
            network_state: Current network state
            forced_path: Optional forced path

        Returns:
            RouteResult with candidate paths
        """
        if forced_path is not None:
            return self._handle_forced_path(
                forced_path, bandwidth_gbps, network_state
            )

        # Call legacy routing
        legacy_result = self._legacy_routing.get_route(
            source=source,
            destination=destination,
        )

        # Translate to new format
        return self._translate_result(legacy_result, bandwidth_gbps)

    def _build_legacy_props(self) -> dict:
        """Build legacy engine_props dict from config."""
        return {
            "topology": self.config.topology,
            "k_paths": self.config.k_paths,
            "route_method": self.config.route_method,
            "mod_formats": self.config.modulation_formats,
            "mod_per_bw": self.config.mod_per_bw,
            # ... other legacy properties
        }

    def _translate_result(
        self,
        legacy_result: dict,
        bandwidth_gbps: int,
    ) -> RouteResult:
        """
        Translate legacy result dict to RouteResult.

        Legacy format:
        {
            "paths_matrix": [[node1, node2, ...], ...],
            "weights_list": [weight1, weight2, ...],
            "mod_formats_list": [[mod1, mod2], ...],
        }
        """
        paths = legacy_result.get("paths_matrix", [])
        weights = legacy_result.get("weights_list", [])
        mods = legacy_result.get("mod_formats_list", [])

        # Filter modulations by bandwidth and path length
        filtered_mods = []
        for i, path_mods in enumerate(mods):
            valid_mods = self._filter_modulations(
                path_mods, weights[i], bandwidth_gbps
            )
            filtered_mods.append(valid_mods)

        return RouteResult(
            paths=paths,
            weights_km=weights,
            modulations=filtered_mods,
            strategy_name=self.config.route_method,
        )

    def _filter_modulations(
        self,
        modulations: list[str],
        path_length_km: float,
        bandwidth_gbps: int,
    ) -> list[str | None]:
        """Filter modulations by reach and bandwidth constraints."""
        valid = []
        for mod in modulations:
            mod_info = self.config.modulation_formats.get(mod, {})
            max_reach = mod_info.get("max_reach_km", float("inf"))

            if path_length_km <= max_reach:
                valid.append(mod)

        return valid if valid else [None]

    def _handle_forced_path(
        self,
        path: list[str],
        bandwidth_gbps: int,
        network_state: NetworkState,
    ) -> RouteResult:
        """Handle forced path case (from grooming)."""
        weight = self._compute_path_weight(path, network_state)
        mods = list(self.config.modulation_formats.keys())
        filtered_mods = self._filter_modulations(mods, weight, bandwidth_gbps)

        return RouteResult(
            paths=[path],
            weights_km=[weight],
            modulations=[filtered_mods],
            strategy_name="forced",
        )

    def _compute_path_weight(
        self,
        path: list[str],
        network_state: NetworkState,
    ) -> float:
        """Compute total path length in km."""
        total = 0.0
        topology = network_state.topology
        for i in range(len(path) - 1):
            edge_data = topology[path[i]][path[i + 1]]
            total += edge_data.get("weight", 0.0)
        return total
```

---

## SpectrumAdapter

### File: `fusion/core/adapters/spectrum_adapter.py`

```python
"""Adapter wrapping legacy spectrum assignment."""

from __future__ import annotations

from fusion.domain.config import SimulationConfig
from fusion.domain.network_state import NetworkState
from fusion.domain.results import SpectrumResult
from fusion.spectrum.spectrum_assignment import SpectrumAssignment  # Legacy


class SpectrumAdapter:
    """
    Adapts legacy spectrum assignment to SpectrumPipeline protocol.

    Handles translation between:
    - New: NetworkState with typed access
    - Legacy: network_spectrum_dict with numpy arrays
    """

    def __init__(self, config: SimulationConfig):
        """Initialize adapter with configuration."""
        self.config = config
        self._legacy_assigner = SpectrumAssignment(
            allocation_method=config.allocation_method,
            guard_slots=config.guard_slots,
        )

    def find_spectrum(
        self,
        path: list[str],
        modulations: list[str | None],
        bandwidth_gbps: int,
        network_state: NetworkState,
    ) -> SpectrumResult:
        """
        Find available spectrum using legacy assignment.

        Args:
            path: Node sequence
            modulations: Valid modulation formats
            bandwidth_gbps: Required bandwidth
            network_state: Current network state

        Returns:
            SpectrumResult with slot assignment
        """
        # Get legacy format for spectrum checking
        legacy_spectrum = network_state.network_spectrum_dict

        # Try each modulation in order
        for mod in modulations:
            if mod is None:
                continue

            slots_needed = self._compute_slots_needed(bandwidth_gbps, mod)

            # Try each core
            for core in range(self.config.cores_per_link):
                # Try each band
                for band in self.config.band_list:
                    result = self._legacy_assigner.find_free_slots(
                        path=path,
                        slots_needed=slots_needed,
                        core=core,
                        band=band,
                        spectrum_dict=legacy_spectrum,
                    )

                    if result is not None:
                        start_slot, end_slot = result
                        return SpectrumResult(
                            is_free=True,
                            start_slot=start_slot,
                            end_slot=end_slot,
                            core=core,
                            band=band,
                            modulation=mod,
                            slots_needed=slots_needed,
                        )

        return SpectrumResult(is_free=False)

    def find_protected_spectrum(
        self,
        primary_path: list[str],
        backup_path: list[str],
        modulations: list[str | None],
        bandwidth_gbps: int,
        network_state: NetworkState,
    ) -> SpectrumResult:
        """
        Find spectrum available on both primary and backup paths.

        For 1+1 protection, same slots must be free on both paths.
        """
        legacy_spectrum = network_state.network_spectrum_dict

        for mod in modulations:
            if mod is None:
                continue

            slots_needed = self._compute_slots_needed(bandwidth_gbps, mod)

            for core in range(self.config.cores_per_link):
                for band in self.config.band_list:
                    # Check primary
                    primary_result = self._legacy_assigner.find_free_slots(
                        path=primary_path,
                        slots_needed=slots_needed,
                        core=core,
                        band=band,
                        spectrum_dict=legacy_spectrum,
                    )

                    if primary_result is None:
                        continue

                    start_slot, end_slot = primary_result

                    # Check backup at same slots
                    backup_free = self._is_range_free_on_path(
                        backup_path, start_slot, end_slot, core, band,
                        legacy_spectrum
                    )

                    if backup_free:
                        return SpectrumResult(
                            is_free=True,
                            start_slot=start_slot,
                            end_slot=end_slot,
                            core=core,
                            band=band,
                            modulation=mod,
                            slots_needed=slots_needed,
                            backup_start_slot=start_slot,
                            backup_end_slot=end_slot,
                            backup_core=core,
                            backup_band=band,
                        )

        return SpectrumResult(is_free=False)

    def _compute_slots_needed(self, bandwidth_gbps: int, modulation: str) -> int:
        """Compute slots needed for bandwidth and modulation."""
        mod_info = self.config.modulation_formats.get(modulation, {})
        bits_per_symbol = mod_info.get("bits_per_symbol", 2)
        slot_bandwidth_ghz = self.config.slot_bandwidth_ghz

        # Simple calculation (actual formula may be more complex)
        spectral_efficiency = bits_per_symbol * 2  # Simplified
        slots = int(bandwidth_gbps / (slot_bandwidth_ghz * spectral_efficiency))
        return max(slots, 1) + self.config.guard_slots

    def _is_range_free_on_path(
        self,
        path: list[str],
        start_slot: int,
        end_slot: int,
        core: int,
        band: str,
        spectrum_dict: dict,
    ) -> bool:
        """Check if slot range is free on all path links."""
        for i in range(len(path) - 1):
            link = (path[i], path[i + 1])
            if link not in spectrum_dict:
                return False

            matrix = spectrum_dict[link]["cores_matrix"][band][core]
            if any(matrix[start_slot:end_slot] != 0):
                return False

        return True
```

---

## GroomingAdapter

### File: `fusion/core/adapters/grooming_adapter.py`

```python
"""Adapter wrapping legacy grooming module."""

from __future__ import annotations

from fusion.domain.config import SimulationConfig
from fusion.domain.network_state import NetworkState
from fusion.domain.request import Request
from fusion.domain.results import GroomingResult
from fusion.grooming.grooming import Grooming  # Legacy


class GroomingAdapter:
    """
    Adapts legacy grooming to GroomingPipeline protocol.

    Translates between:
    - New: Request objects, Lightpath objects
    - Legacy: request dicts, lightpath_status_dict
    """

    def __init__(self, config: SimulationConfig):
        """Initialize adapter with configuration."""
        self.config = config
        self._legacy_grooming = Grooming(self._build_legacy_props())

    def try_groom(
        self,
        request: Request,
        network_state: NetworkState,
    ) -> GroomingResult:
        """
        Attempt to groom request using legacy logic.

        Args:
            request: Incoming request
            network_state: Current network state

        Returns:
            GroomingResult indicating grooming outcome
        """
        # Build legacy request dict
        legacy_request = self._request_to_dict(request)

        # Get legacy lightpath dict
        legacy_lp_dict = network_state.lightpath_status_dict

        # Call legacy grooming
        result = self._legacy_grooming.try_groom(
            request=legacy_request,
            lightpath_status_dict=legacy_lp_dict,
        )

        # Translate result
        return self._translate_result(result, request.bandwidth_gbps)

    def rollback(
        self,
        request: Request,
        lightpath_ids: list[int],
        network_state: NetworkState,
    ) -> None:
        """
        Rollback grooming allocations.

        Called when request cannot be fully served after partial grooming.
        """
        for lp_id in lightpath_ids:
            lightpath = network_state.get_lightpath(lp_id)
            if lightpath is None:
                continue

            # Get allocated bandwidth for this request
            allocated = lightpath.request_allocations.get(request.request_id, 0)

            # Restore bandwidth
            lightpath.remaining_bandwidth_gbps += allocated

            # Remove allocation record
            lightpath.request_allocations.pop(request.request_id, None)

    def _request_to_dict(self, request: Request) -> dict:
        """Convert Request to legacy dict format."""
        return {
            "request_id": request.request_id,
            "source": request.source,
            "destination": request.destination,
            "bandwidth": request.bandwidth_gbps,
            "arrive": request.arrival_time,
            "depart": request.arrival_time + request.holding_time,
        }

    def _translate_result(
        self,
        legacy_result: dict,
        original_bw: int,
    ) -> GroomingResult:
        """
        Translate legacy grooming result to GroomingResult.

        Legacy format:
        {
            "groomed": True/False,
            "fully_groomed": True/False,
            "bandwidth_groomed": int,
            "lightpaths_used": [lp_id, ...],
            "forced_path": [node, ...] or None,
        }
        """
        if not legacy_result.get("groomed", False):
            return GroomingResult(
                fully_groomed=False,
                partially_groomed=False,
                bandwidth_groomed_gbps=0,
                remaining_bandwidth_gbps=original_bw,
                lightpaths_used=[],
            )

        bw_groomed = legacy_result.get("bandwidth_groomed", 0)
        remaining = original_bw - bw_groomed
        fully = legacy_result.get("fully_groomed", False)

        return GroomingResult(
            fully_groomed=fully,
            partially_groomed=not fully and bw_groomed > 0,
            bandwidth_groomed_gbps=bw_groomed,
            remaining_bandwidth_gbps=remaining,
            lightpaths_used=legacy_result.get("lightpaths_used", []),
            forced_path=legacy_result.get("forced_path"),
        )

    def _build_legacy_props(self) -> dict:
        """Build legacy props dict from config."""
        return {
            "grooming_policy": self.config.grooming_policy,
            "can_partially_serve": self.config.can_partially_serve,
        }
```

---

## SNRAdapter

### File: `fusion/core/adapters/snr_adapter.py`

```python
"""Adapter wrapping legacy SNR calculation."""

from __future__ import annotations

from fusion.domain.config import SimulationConfig
from fusion.domain.lightpath import Lightpath
from fusion.domain.network_state import NetworkState
from fusion.domain.results import SNRResult
from fusion.snr.snr_calculator import SNRCalculator  # Legacy


class SNRAdapter:
    """
    Adapts legacy SNR calculation to SNRPipeline protocol.
    """

    def __init__(self, config: SimulationConfig):
        """Initialize adapter with configuration."""
        self.config = config
        self._legacy_snr = SNRCalculator(
            snr_type=config.snr_type,
            snr_thresholds=config.snr_thresholds,
        )

    def validate(
        self,
        lightpath: Lightpath,
        network_state: NetworkState,
    ) -> SNRResult:
        """
        Validate SNR for a lightpath.

        Args:
            lightpath: Lightpath to validate
            network_state: Current network state (for interference)

        Returns:
            SNRResult with pass/fail and SNR value
        """
        # Get legacy format data
        legacy_spectrum = network_state.network_spectrum_dict
        legacy_lp_dict = network_state.lightpath_status_dict

        # Build lightpath info dict
        lp_info = self._lightpath_to_dict(lightpath)

        # Call legacy SNR calculation
        snr_db = self._legacy_snr.calculate(
            lightpath_info=lp_info,
            spectrum_dict=legacy_spectrum,
            lightpath_dict=legacy_lp_dict,
            topology=network_state.topology,
        )

        # Get required SNR for modulation
        required_snr = self.config.snr_thresholds.get(
            lightpath.modulation, 0.0
        )

        return SNRResult(
            passed=snr_db >= required_snr,
            snr_db=snr_db,
            required_snr_db=required_snr,
        )

    def validate_protected(
        self,
        lightpath: Lightpath,
        network_state: NetworkState,
    ) -> SNRResult:
        """
        Validate SNR on both primary and backup paths.

        For 1+1 protection, both paths must pass SNR check.
        """
        # Validate primary
        primary_result = self.validate(lightpath, network_state)

        if not primary_result.passed:
            return primary_result

        if lightpath.backup_path is None:
            return primary_result

        # Create temporary lightpath for backup path validation
        backup_lp = Lightpath(
            lightpath_id=lightpath.lightpath_id,
            path=lightpath.backup_path,
            start_slot=lightpath.start_slot,
            end_slot=lightpath.end_slot,
            core=lightpath.core,
            band=lightpath.band,
            modulation=lightpath.modulation,
            bandwidth_gbps=lightpath.bandwidth_gbps,
            remaining_bandwidth_gbps=lightpath.remaining_bandwidth_gbps,
            path_weight_km=lightpath.backup_path_weight_km or 0.0,
        )

        backup_result = self.validate(backup_lp, network_state)

        # Both must pass
        if not backup_result.passed:
            return SNRResult(
                passed=False,
                snr_db=backup_result.snr_db,
                required_snr_db=backup_result.required_snr_db,
                metadata={"failed_on": "backup"},
            )

        return SNRResult(
            passed=True,
            snr_db=primary_result.snr_db,
            required_snr_db=primary_result.required_snr_db,
            metadata={
                "backup_snr_db": backup_result.snr_db,
            },
        )

    def _lightpath_to_dict(self, lightpath: Lightpath) -> dict:
        """Convert Lightpath to legacy dict format."""
        return {
            "path": lightpath.path,
            "core": lightpath.core,
            "band": lightpath.band,
            "start_slot": lightpath.start_slot,
            "end_slot": lightpath.end_slot,
            "mod_format": lightpath.modulation,
            "lightpath_bandwidth": lightpath.bandwidth_gbps,
        }
```

---

## Design Rules for Adapters

### 1. No Behavior Modification

Adapters must produce identical results to legacy code:

```python
# Test for parity
def test_adapter_matches_legacy():
    legacy_result = legacy_routing.get_route(src, dst)
    adapter_result = routing_adapter.find_routes(src, dst, bw, network_state)

    assert adapter_result.paths == legacy_result["paths_matrix"]
```

### 2. Receive NetworkState Per Call

```python
# GOOD
def find_routes(self, ..., network_state: NetworkState):
    legacy_spectrum = network_state.network_spectrum_dict
    ...

# BAD
def __init__(self, network_state: NetworkState):
    self._state = network_state  # NEVER cache
```

### 3. Translate at Boundaries

```python
# Input: Translate new types to legacy format
legacy_request = self._request_to_dict(request)

# Call legacy code
legacy_result = self._legacy_method(legacy_request)

# Output: Translate legacy format to new types
return RouteResult(...)
```

### 4. Document Legacy Dependencies

```python
class RoutingAdapter:
    """
    Adapts legacy Routing class.

    Legacy Dependencies:
    - Routing.get_route() returns dict with paths_matrix, weights_list
    - Requires engine_props format for initialization

    Will Be Replaced By:
    - KShortestPathStrategy (Phase 4)
    """
```

---

## Testing Adapters

### Parity Tests

```python
class TestRoutingAdapterParity:
    """Verify adapter produces same results as legacy."""

    def test_paths_match_legacy(self, network_state, config):
        # Setup
        legacy = Routing(build_legacy_props(config))
        adapter = RoutingAdapter(config)

        # Act
        legacy_result = legacy.get_route("A", "D")
        adapter_result = adapter.find_routes("A", "D", 100, network_state)

        # Assert
        assert adapter_result.paths == legacy_result["paths_matrix"]

    def test_weights_match_legacy(self, network_state, config):
        legacy = Routing(build_legacy_props(config))
        adapter = RoutingAdapter(config)

        legacy_result = legacy.get_route("A", "D")
        adapter_result = adapter.find_routes("A", "D", 100, network_state)

        assert adapter_result.weights_km == legacy_result["weights_list"]
```

### Protocol Compliance Tests

```python
def test_adapter_implements_protocol():
    from fusion.interfaces.pipelines import RoutingPipeline

    adapter = RoutingAdapter(config)

    # Protocol check
    assert hasattr(adapter, "find_routes")
    assert callable(adapter.find_routes)

    # Duck typing check
    result = adapter.find_routes("A", "D", 100, network_state)
    assert isinstance(result, RouteResult)
```

---

## Migration Path

### Step 1: Create Adapter

```python
# fusion/core/adapters/routing_adapter.py
class RoutingAdapter:
    def find_routes(self, ...):
        # Wrap legacy Routing class
        ...
```

### Step 2: Use in Orchestrator

```python
# PipelineFactory creates adapter
pipelines = PipelineSet(
    routing=RoutingAdapter(config),
    spectrum=SpectrumAdapter(config),
)
orchestrator = SDNOrchestrator(config, pipelines)
```

### Step 3: Develop Native Pipeline

```python
# fusion/routing/strategies/ksp.py
class KShortestPathStrategy:
    def find_routes(self, ...):
        # Native implementation (no legacy dependency)
        ...
```

### Step 4: Swap in Factory

```python
@staticmethod
def create_routing(config: SimulationConfig) -> RoutingPipeline:
    # Before: return RoutingAdapter(config)
    return KShortestPathStrategy(config)  # Native!
```

### Step 5: Remove Adapter

```bash
git rm fusion/core/adapters/routing_adapter.py
```

---

## Related Documentation

- [Migration: Phase 2 Checklist](../migration/phase_2_checklist.md) - Adapter creation tasks
- [Architecture: Pipeline Interfaces](./pipeline_interfaces.md) - Protocols adapters implement
- [Testing: Phase 2 Testing](../testing/phase_2_testing.md) - Adapter test patterns
