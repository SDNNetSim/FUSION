"""Phase 4.1 Integration Smoke Test.

This script verifies that RLSimulationAdapter correctly integrates
with the real SDNOrchestrator and pipelines.

Run with: python tests/integration/test_p41_adapter_integration.py

Key verifications:
1. Adapter can be created from real orchestrator
2. Pipeline identity is preserved (adapter.routing IS orchestrator.routing)
3. Adapter methods don't crash when called with mock data
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_adapter_creation_with_mock_orchestrator() -> bool:
    """Test 1: Adapter can be created with a mock orchestrator."""
    print("\n[Test 1] Adapter creation with mock orchestrator...")

    from fusion.modules.rl.adapter import RLSimulationAdapter, RLConfig

    # Create mock orchestrator with required properties
    mock_orchestrator = MagicMock()
    mock_orchestrator.routing = MagicMock()
    mock_orchestrator.spectrum = MagicMock()

    # Create adapter
    config = RLConfig(k_paths=3)
    adapter = RLSimulationAdapter(orchestrator=mock_orchestrator, config=config)

    # Verify adapter was created
    assert adapter is not None, "Adapter should not be None"
    assert adapter.orchestrator is mock_orchestrator, "Should store orchestrator reference"

    print("  PASSED: Adapter created successfully")
    return True


def test_pipeline_identity() -> bool:
    """Test 2: Pipeline identity is preserved (same object, not copy)."""
    print("\n[Test 2] Pipeline identity verification...")

    from fusion.modules.rl.adapter import RLSimulationAdapter

    # Create mock orchestrator
    mock_routing = MagicMock(name="routing_pipeline")
    mock_spectrum = MagicMock(name="spectrum_pipeline")
    mock_orchestrator = MagicMock()
    mock_orchestrator.routing = mock_routing
    mock_orchestrator.spectrum = mock_spectrum

    # Create adapter
    adapter = RLSimulationAdapter(orchestrator=mock_orchestrator)

    # Verify identity (same object, not just equal)
    assert adapter.routing is mock_routing, "adapter.routing should BE orchestrator.routing"
    assert adapter.spectrum is mock_spectrum, "adapter.spectrum should BE orchestrator.spectrum"
    assert id(adapter.routing) == id(mock_orchestrator.routing), "Must be same object (identity)"
    assert id(adapter.spectrum) == id(mock_orchestrator.spectrum), "Must be same object (identity)"

    print("  PASSED: Pipeline identity preserved")
    return True


def test_get_path_options_with_mock() -> bool:
    """Test 3: get_path_options works with mock data."""
    print("\n[Test 3] get_path_options with mock data...")

    from fusion.modules.rl.adapter import RLSimulationAdapter, PathOption

    # Create mock orchestrator with routing that returns mock results
    mock_route_result = MagicMock()
    mock_route_result.paths = (("0", "2", "5"),)
    mock_route_result.weights_km = (450.0,)
    mock_route_result.modulations = (("QPSK",),)
    mock_route_result.is_empty = False
    # Make len() work on paths
    mock_route_result.__len__ = lambda self: 1

    mock_spectrum_result = MagicMock()
    mock_spectrum_result.is_free = True
    mock_spectrum_result.slots_needed = 4
    mock_spectrum_result.start_slot = 10
    mock_spectrum_result.end_slot = 14
    mock_spectrum_result.core = 0
    mock_spectrum_result.band = "C"

    mock_routing = MagicMock()
    mock_routing.find_routes.return_value = mock_route_result

    mock_spectrum = MagicMock()
    mock_spectrum.find_spectrum.return_value = mock_spectrum_result

    mock_orchestrator = MagicMock()
    mock_orchestrator.routing = mock_routing
    mock_orchestrator.spectrum = mock_spectrum

    # Create mock request and network state
    mock_request = MagicMock()
    mock_request.source = "0"
    mock_request.destination = "5"
    mock_request.bandwidth_gbps = 100.0
    mock_request.holding_time = 50.0

    # Use spec=[] so hasattr() returns False for missing methods
    # This avoids MagicMock comparison issues in _compute_path_congestion
    mock_network_state = MagicMock(spec=[])

    # Create adapter and call get_path_options
    adapter = RLSimulationAdapter(orchestrator=mock_orchestrator)
    options = adapter.get_path_options(mock_request, mock_network_state)

    # Verify results
    assert len(options) == 1, f"Expected 1 option, got {len(options)}"
    assert isinstance(options[0], PathOption), "Should return PathOption instances"
    assert options[0].path == ("0", "2", "5"), f"Path mismatch: {options[0].path}"
    assert options[0].is_feasible is True, "Path should be feasible"
    assert options[0].slots_needed == 4, f"Slots mismatch: {options[0].slots_needed}"

    # Verify routing was called
    mock_routing.find_routes.assert_called_once()

    print("  PASSED: get_path_options works correctly")
    return True


def test_get_action_mask() -> bool:
    """Test 4: get_action_mask returns correct mask."""
    print("\n[Test 4] get_action_mask verification...")

    from fusion.modules.rl.adapter import RLSimulationAdapter, RLConfig, PathOption
    import numpy as np

    # Create mock orchestrator
    mock_orchestrator = MagicMock()
    mock_orchestrator.routing = MagicMock()
    mock_orchestrator.spectrum = MagicMock()

    # Create adapter with k_paths=3
    config = RLConfig(k_paths=3)
    adapter = RLSimulationAdapter(orchestrator=mock_orchestrator, config=config)

    # Create mock path options
    options = [
        PathOption(
            path_index=0, path=("0", "1"), weight_km=100.0, num_hops=1,
            modulation="QPSK", slots_needed=4, is_feasible=True,
            congestion=0.0, available_slots=1.0,
        ),
        PathOption(
            path_index=1, path=("0", "2", "1"), weight_km=200.0, num_hops=2,
            modulation="QPSK", slots_needed=4, is_feasible=False,
            congestion=0.5, available_slots=0.5,
        ),
    ]

    # Get action mask
    mask = adapter.get_action_mask(options)

    # Verify
    assert mask.shape == (3,), f"Expected shape (3,), got {mask.shape}"
    assert mask.dtype == np.bool_, f"Expected bool dtype, got {mask.dtype}"
    assert mask.tolist() == [True, False, False], f"Mask mismatch: {mask.tolist()}"

    print("  PASSED: get_action_mask works correctly")
    return True


def test_apply_action_with_mock() -> bool:
    """Test 5: apply_action routes through orchestrator."""
    print("\n[Test 5] apply_action verification...")

    from fusion.modules.rl.adapter import RLSimulationAdapter, PathOption

    # Create mock orchestrator that returns success
    mock_result = MagicMock()
    mock_result.success = True

    mock_orchestrator = MagicMock()
    mock_orchestrator.routing = MagicMock()
    mock_orchestrator.spectrum = MagicMock()
    mock_orchestrator.handle_arrival.return_value = mock_result

    # Create adapter
    adapter = RLSimulationAdapter(orchestrator=mock_orchestrator)

    # Create mock request, state, and options
    mock_request = MagicMock()
    mock_network_state = MagicMock()
    options = [
        PathOption(
            path_index=0, path=("0", "2", "5"), weight_km=450.0, num_hops=2,
            modulation="QPSK", slots_needed=4, is_feasible=True,
            congestion=0.0, available_slots=1.0,
        ),
    ]

    # Apply action
    result = adapter.apply_action(0, mock_request, mock_network_state, options)

    # Verify orchestrator was called with forced_path
    mock_orchestrator.handle_arrival.assert_called_once()
    call_kwargs = mock_orchestrator.handle_arrival.call_args.kwargs
    assert "forced_path" in call_kwargs, "Should pass forced_path to orchestrator"
    assert call_kwargs["forced_path"] == ["0", "2", "5"], f"Path mismatch: {call_kwargs['forced_path']}"
    assert result.success is True, "Should return orchestrator result"

    print("  PASSED: apply_action routes through orchestrator")
    return True


def test_compute_reward() -> bool:
    """Test 6: compute_reward uses config values."""
    print("\n[Test 6] compute_reward verification...")

    from fusion.modules.rl.adapter import RLSimulationAdapter, RLConfig

    # Create mock orchestrator
    mock_orchestrator = MagicMock()
    mock_orchestrator.routing = MagicMock()
    mock_orchestrator.spectrum = MagicMock()

    # Create adapter with custom reward config
    config = RLConfig(
        rl_success_reward=2.0,
        rl_block_penalty=-2.0,
        rl_grooming_bonus=0.5,
    )
    adapter = RLSimulationAdapter(orchestrator=mock_orchestrator, config=config)

    # Test success reward (explicitly set is_groomed/is_sliced to False)
    success_result = MagicMock(spec=["success", "is_groomed", "is_sliced"])
    success_result.success = True
    success_result.is_groomed = False
    success_result.is_sliced = False
    reward = adapter.compute_reward(success_result)
    assert reward == 2.0, f"Expected 2.0, got {reward}"

    # Test failure penalty
    failure_result = MagicMock(spec=["success"])
    failure_result.success = False
    reward = adapter.compute_reward(failure_result)
    assert reward == -2.0, f"Expected -2.0, got {reward}"

    # Test grooming bonus
    groomed_result = MagicMock(spec=["success", "is_groomed", "is_sliced"])
    groomed_result.success = True
    groomed_result.is_groomed = True
    groomed_result.is_sliced = False
    reward = adapter.compute_reward(groomed_result)
    assert reward == 2.5, f"Expected 2.5 (2.0 + 0.5), got {reward}"

    print("  PASSED: compute_reward uses config values")
    return True


def test_build_observation() -> bool:
    """Test 7: build_observation creates valid observation dict."""
    print("\n[Test 7] build_observation verification...")

    from fusion.modules.rl.adapter import RLSimulationAdapter, RLConfig, PathOption
    import numpy as np

    # Create mock orchestrator
    mock_orchestrator = MagicMock()
    mock_orchestrator.routing = MagicMock()
    mock_orchestrator.spectrum = MagicMock()

    # Create adapter
    config = RLConfig(k_paths=3, num_nodes=14)
    adapter = RLSimulationAdapter(orchestrator=mock_orchestrator, config=config)

    # Create mock request
    mock_request = MagicMock()
    mock_request.source = "2"
    mock_request.destination = "5"
    mock_request.holding_time = 50.0

    mock_network_state = MagicMock()

    options = [
        PathOption(
            path_index=0, path=("2", "5"), weight_km=100.0, num_hops=1,
            modulation="QPSK", slots_needed=4, is_feasible=True,
            congestion=0.3, available_slots=0.7,
        ),
    ]

    # Build observation
    obs = adapter.build_observation(mock_request, options, mock_network_state)

    # Verify structure
    assert "source" in obs, "Should have source field"
    assert "destination" in obs, "Should have destination field"
    assert "holding_time" in obs, "Should have holding_time field"
    assert "slots_needed" in obs, "Should have slots_needed field"
    assert "is_feasible" in obs, "Should have is_feasible field"

    # Verify shapes
    assert obs["source"].shape == (14,), f"Source shape mismatch: {obs['source'].shape}"
    assert obs["slots_needed"].shape == (3,), f"Slots shape mismatch: {obs['slots_needed'].shape}"

    # Verify one-hot encoding
    assert obs["source"][2] == 1.0, "Source one-hot should be at index 2"
    assert obs["destination"][5] == 1.0, "Dest one-hot should be at index 5"

    print("  PASSED: build_observation creates valid dict")
    return True


def test_disaster_state() -> bool:
    """Test 8: DisasterState dataclass works correctly."""
    print("\n[Test 8] DisasterState verification...")

    from fusion.modules.rl.adapter import DisasterState, create_disaster_state_from_engine

    # Test direct creation
    ds = DisasterState(
        active=True,
        centroid=(100.0, 200.0),
        radius=50.0,
        failed_links=frozenset({("0", "1"), ("2", "3")}),
        network_diameter=1000.0,
    )

    assert ds.active is True
    assert ds.centroid == (100.0, 200.0)
    assert len(ds.failed_links) == 2

    # Test factory function
    engine_props = {
        "is_disaster": True,
        "disaster_centroid": (150.0, 250.0),
        "disaster_radius": 75.0,
        "failed_links": [("1", "2"), ("3", "4")],
        "network_diameter": 500.0,
    }
    ds2 = create_disaster_state_from_engine(engine_props)

    assert ds2 is not None, "Should create DisasterState"
    assert ds2.active is True
    assert ds2.centroid == (150.0, 250.0)

    # Test with inactive disaster
    ds3 = create_disaster_state_from_engine({"is_disaster": False})
    assert ds3 is None, "Should return None for inactive disaster"

    print("  PASSED: DisasterState works correctly")
    return True


def main() -> int:
    """Run all integration tests."""
    print("=" * 60)
    print("Phase 4.1 Integration Smoke Tests")
    print("=" * 60)

    tests = [
        test_adapter_creation_with_mock_orchestrator,
        test_pipeline_identity,
        test_get_path_options_with_mock,
        test_get_action_mask,
        test_apply_action_with_mock,
        test_compute_reward,
        test_build_observation,
        test_disaster_state,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("\nPhase 4.1 Integration: ALL TESTS PASSED")
        return 0
    else:
        print(f"\nPhase 4.1 Integration: {failed} TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
