# Component 10: Tests

**Files:** `tests/test_grooming.py`, modifications to existing test files
**Estimated Time:** 2 hours
**Dependencies:** All previous components

## Overview

Create comprehensive test suite for grooming functionality, including unit tests, integration tests, and end-to-end tests.

## New Test File

### Create `tests/test_grooming.py`

Port and adapt tests from v5.5 `tests/test_grooming.py`:

```python
"""
Comprehensive tests for traffic grooming functionality.
"""

import pytest
import networkx as nx

from fusion.core.grooming import Grooming
from fusion.core.properties import SDNProps, GroomingProps
from fusion.core.sdn_controller import SDNController


class TestGroomingInit:
    """Test grooming initialization."""

    def test_grooming_initialization(self, engine_props):
        """Test basic grooming object creation."""
        sdn_props = SDNProps()
        grooming = Grooming(engine_props, sdn_props)

        assert grooming.engine_props == engine_props
        assert grooming.sdn_props == sdn_props
        assert isinstance(grooming.grooming_props, GroomingProps)

    def test_grooming_props_initialization(self):
        """Test GroomingProps initialization."""
        props = GroomingProps()

        assert props.grooming_type is None
        assert props.lightpath_status_dict is None


class TestFindPathMaxBandwidth:
    """Test _find_path_max_bw method."""

    def test_find_path_no_lightpaths(self, engine_props):
        """Test with no existing lightpaths."""
        sdn_props = SDNProps()
        sdn_props.lightpath_status_dict = {('A', 'B'): {}}

        grooming = Grooming(engine_props, sdn_props)
        result = grooming._find_path_max_bw(('A', 'B'))

        assert result is None

    def test_find_path_single_lightpath(self, engine_props):
        """Test with single lightpath."""
        sdn_props = SDNProps()
        sdn_props.lightpath_status_dict = {
            ('A', 'B'): {
                1: {
                    'path': ['A', 'C', 'B'],
                    'remaining_bandwidth': 100,
                    'is_degraded': False
                }
            }
        }

        grooming = Grooming(engine_props, sdn_props)
        result = grooming._find_path_max_bw(('A', 'B'))

        assert result is not None
        assert result['total_remaining_bandwidth'] == 100
        assert 1 in result['lp_id_list']

    def test_find_path_skips_degraded(self, engine_props):
        """Test that degraded lightpaths are skipped."""
        sdn_props = SDNProps()
        sdn_props.lightpath_status_dict = {
            ('A', 'B'): {
                1: {
                    'path': ['A', 'C', 'B'],
                    'remaining_bandwidth': 100,
                    'is_degraded': True  # Should be skipped
                },
                2: {
                    'path': ['A', 'D', 'B'],
                    'remaining_bandwidth': 50,
                    'is_degraded': False
                }
            }
        }

        grooming = Grooming(engine_props, sdn_props)
        result = grooming._find_path_max_bw(('A', 'B'))

        assert result is not None
        assert result['total_remaining_bandwidth'] == 50
        assert 2 in result['lp_id_list']
        assert 1 not in result['lp_id_list']

    def test_find_path_groups_by_path(self, engine_props):
        """Test that lightpaths on same path are grouped."""
        sdn_props = SDNProps()
        sdn_props.lightpath_status_dict = {
            ('A', 'B'): {
                1: {
                    'path': ['A', 'C', 'B'],
                    'remaining_bandwidth': 50,
                    'is_degraded': False
                },
                2: {
                    'path': ['A', 'C', 'B'],  # Same path
                    'remaining_bandwidth': 30,
                    'is_degraded': False
                },
                3: {
                    'path': ['A', 'D', 'B'],  # Different path
                    'remaining_bandwidth': 60,
                    'is_degraded': False
                }
            }
        }

        grooming = Grooming(engine_props, sdn_props)
        result = grooming._find_path_max_bw(('A', 'B'))

        # Should select the A-C-B path group with total 80 bandwidth
        assert result is not None
        assert result['total_remaining_bandwidth'] == 80
        assert set(result['lp_id_list']) == {1, 2}


class TestEndToEndGrooming:
    """Test _end_to_end_grooming method."""

    def test_grooming_no_existing_lightpaths(self, engine_props):
        """Test grooming when no lightpaths exist."""
        sdn_props = SDNProps()
        sdn_props.source = 'A'
        sdn_props.destination = 'B'
        sdn_props.bandwidth = 100
        sdn_props.lightpath_status_dict = {}

        grooming = Grooming(engine_props, sdn_props)
        result = grooming._end_to_end_grooming()

        assert result is False

    def test_grooming_full_allocation(self, engine_props):
        """Test full grooming without creating new lightpath."""
        sdn_props = SDNProps()
        sdn_props.source = 'A'
        sdn_props.destination = 'B'
        sdn_props.bandwidth = 50
        sdn_props.request_id = 100
        sdn_props.arrive = 10.0
        sdn_props.lightpath_status_dict = {
            ('A', 'B'): {
                1: {
                    'path': ['A', 'C', 'B'],
                    'path_weight': 100.0,
                    'remaining_bandwidth': 100,
                    'lightpath_bandwidth': 200,
                    'is_degraded': False,
                    'core': 0,
                    'band': 'C',
                    'start_slot': 10,
                    'end_slot': 20,
                    'mod_format': 'QPSK',
                    'snr_cost': 0.5,
                    'requests_dict': {},
                    'time_bw_usage': {}
                }
            }
        }

        grooming = Grooming(engine_props, sdn_props)
        result = grooming._end_to_end_grooming()

        assert result is True
        assert sdn_props.was_groomed is True
        assert sdn_props.was_partially_groomed is False
        assert sdn_props.number_of_transponders == 0
        assert sdn_props.remaining_bw == "0"

        # Check lightpath was updated
        lp_info = sdn_props.lightpath_status_dict[('A', 'B')][1]
        assert lp_info['remaining_bandwidth'] == 50
        assert 100 in lp_info['requests_dict']
        assert lp_info['requests_dict'][100] == 50

    def test_grooming_partial_allocation(self, engine_props):
        """Test partial grooming when not enough bandwidth available."""
        sdn_props = SDNProps()
        sdn_props.source = 'A'
        sdn_props.destination = 'B'
        sdn_props.bandwidth = 150
        sdn_props.request_id = 100
        sdn_props.arrive = 10.0
        sdn_props.lightpath_status_dict = {
            ('A', 'B'): {
                1: {
                    'path': ['A', 'C', 'B'],
                    'path_weight': 100.0,
                    'remaining_bandwidth': 100,  # Not enough for full 150
                    'lightpath_bandwidth': 200,
                    'is_degraded': False,
                    'core': 0,
                    'band': 'C',
                    'start_slot': 10,
                    'end_slot': 20,
                    'mod_format': 'QPSK',
                    'snr_cost': 0.5,
                    'requests_dict': {},
                    'time_bw_usage': {}
                }
            }
        }

        grooming = Grooming(engine_props, sdn_props)
        result = grooming._end_to_end_grooming()

        assert result is False
        assert sdn_props.was_groomed is False
        assert sdn_props.was_partially_groomed is True
        assert sdn_props.remaining_bw == 50  # 150 - 100 = 50 remaining

        # Check lightpath was updated
        lp_info = sdn_props.lightpath_status_dict[('A', 'B')][1]
        assert lp_info['remaining_bandwidth'] == 0


class TestReleaseService:
    """Test _release_service method."""

    def test_release_single_lightpath(self, engine_props):
        """Test releasing a request from single lightpath."""
        sdn_props = SDNProps()
        sdn_props.source = 'A'
        sdn_props.destination = 'B'
        sdn_props.request_id = 100
        sdn_props.depart = 50.0
        sdn_props.remaining_bw = 50
        sdn_props.lightpath_id_list = [1]
        sdn_props.lightpath_bandwidth_list = [200]
        sdn_props.lightpath_status_dict = {
            ('A', 'B'): {
                1: {
                    'remaining_bandwidth': 100,
                    'lightpath_bandwidth': 200,
                    'requests_dict': {100: 50},  # This request allocated 50
                    'time_bw_usage': {0.0: 25.0}
                }
            }
        }

        grooming = Grooming(engine_props, sdn_props)
        release_list = grooming._release_service()

        # Bandwidth should be freed
        lp_info = sdn_props.lightpath_status_dict[('A', 'B')][1]
        assert lp_info['remaining_bandwidth'] == 150  # 100 + 50
        assert 100 not in lp_info['requests_dict']
        assert len(release_list) == 0  # Still has capacity, not released

    def test_release_lightpath_becomes_empty(self, engine_props):
        """Test releasing request that empties lightpath."""
        sdn_props = SDNProps()
        sdn_props.source = 'A'
        sdn_props.destination = 'B'
        sdn_props.request_id = 100
        sdn_props.depart = 50.0
        sdn_props.remaining_bw = 200
        sdn_props.lightpath_id_list = [1]
        sdn_props.lightpath_bandwidth_list = [200]
        sdn_props.lightpath_status_dict = {
            ('A', 'B'): {
                1: {
                    'remaining_bandwidth': 0,  # Fully utilized
                    'lightpath_bandwidth': 200,
                    'requests_dict': {100: 200},  # Only this request
                    'time_bw_usage': {0.0: 100.0}
                }
            }
        }

        grooming = Grooming(engine_props, sdn_props)
        release_list = grooming._release_service()

        # Lightpath should be marked for release
        assert 1 in release_list
        lp_info = sdn_props.lightpath_status_dict[('A', 'B')][1]
        assert lp_info['remaining_bandwidth'] == 200  # Fully freed


class TestHandleGrooming:
    """Test handle_grooming dispatcher method."""

    def test_handle_grooming_arrival(self, engine_props):
        """Test handling arrival request."""
        sdn_props = SDNProps()
        sdn_props.source = 'A'
        sdn_props.destination = 'B'
        sdn_props.bandwidth = 100
        sdn_props.request_id = 100
        sdn_props.arrive = 10.0
        sdn_props.lightpath_status_dict = {}

        grooming = Grooming(engine_props, sdn_props)
        result = grooming.handle_grooming("arrival")

        assert isinstance(result, bool)

    def test_handle_grooming_release(self, engine_props):
        """Test handling release request."""
        sdn_props = SDNProps()
        sdn_props.source = 'A'
        sdn_props.destination = 'B'
        sdn_props.request_id = 100
        sdn_props.lightpath_id_list = []
        sdn_props.lightpath_status_dict = {('A', 'B'): {}}

        grooming = Grooming(engine_props, sdn_props)
        result = grooming.handle_grooming("release")

        assert isinstance(result, list)


class TestGroomingIntegration:
    """Integration tests with SDN controller."""

    def test_sdn_controller_has_grooming(self, engine_props):
        """Test SDN controller initializes grooming object."""
        sdn = SDNController(engine_props)

        assert hasattr(sdn, 'grooming_obj')
        assert isinstance(sdn.grooming_obj, Grooming)

    def test_end_to_end_grooming_workflow(self, engine_props, setup_topology):
        """Test complete grooming workflow."""
        engine_props['is_grooming_enabled'] = True

        # Setup SDN controller with topology
        sdn = SDNController(engine_props)
        sdn.sdn_props.topology = setup_topology
        sdn.sdn_props.lightpath_status_dict = {}

        # First request - should create new lightpath
        sdn.sdn_props.source = 'A'
        sdn.sdn_props.destination = 'B'
        sdn.sdn_props.bandwidth = 100
        sdn.sdn_props.request_id = 1

        result1 = sdn.grooming_obj.handle_grooming("arrival")
        assert result1 is False  # No existing lightpath

        # TODO: Add more complete workflow test


# Fixtures

@pytest.fixture
def engine_props():
    """Basic engine properties."""
    return {
        'is_grooming_enabled': True,
        'cores_per_link': 7,
        'band_list': ['C'],
        'guard_slots': 1
    }


@pytest.fixture
def setup_topology():
    """Create simple test topology."""
    G = nx.Graph()
    G.add_edge('A', 'B', weight=100, length=100)
    G.add_edge('A', 'C', weight=150, length=150)
    G.add_edge('C', 'B', weight=120, length=120)
    return G
```

## Modifications to Existing Test Files

### Update `fusion/core/tests/test_sdn_controller.py`

```python
def test_release_with_lightpath_id(engine_props):
    """Test release method with lightpath ID."""
    sdn = SDNController(engine_props)
    # ... setup network ...

    sdn.release(lightpath_id=1, slicing_flag=False)
    # ... assertions ...


def test_allocate_with_grooming(engine_props):
    """Test allocation with grooming enabled."""
    engine_props['is_grooming_enabled'] = True
    sdn = SDNController(engine_props)
    # ... test allocation ...
```

### Update `fusion/core/tests/test_spectrum_assignment.py`

```python
def test_spectrum_generates_lightpath_id(engine_props):
    """Test lightpath ID generation."""
    # ... test spectrum assignment generates IDs ...


def test_partial_grooming_bandwidth(engine_props):
    """Test bandwidth calculation for partial grooming."""
    # ... test _calculate_slots_needed with partial grooming ...
```

### Update `fusion/core/tests/test_properties.py`

```python
def test_grooming_props():
    """Test GroomingProps class."""
    props = GroomingProps()
    assert props.grooming_type is None
    assert props.lightpath_status_dict is None


def test_sdn_props_grooming_attributes():
    """Test SDN props has grooming attributes."""
    props = SDNProps()
    assert hasattr(props, 'lightpath_status_dict')
    assert hasattr(props, 'was_groomed')
    assert hasattr(props, 'get_lightpath_id')
```

## Running Tests

```bash
# Run all grooming tests
python -m pytest tests/test_grooming.py -v

# Run specific test class
python -m pytest tests/test_grooming.py::TestEndToEndGrooming -v

# Run with coverage
python -m pytest tests/test_grooming.py --cov=fusion.core.grooming --cov-report=html

# Run all tests affected by grooming
python -m pytest -k grooming -v

# Run integration tests
python -m pytest tests/test_grooming.py::TestGroomingIntegration -v
```

## Test Coverage Goals

- **Unit Test Coverage:** >90% for grooming.py
- **Integration Test Coverage:** Key workflows tested
- **Edge Cases:** Empty dicts, degraded lightpaths, partial grooming
- **Error Handling:** Invalid inputs, missing data

## Validation Checklist

- [ ] `test_grooming.py` created with all test classes
- [ ] Unit tests for all grooming methods
- [ ] Integration tests with SDN controller
- [ ] Edge cases covered
- [ ] Existing test files updated
- [ ] All tests pass
- [ ] Code coverage >90% for grooming module
- [ ] Tests pass pylint and mypy
- [ ] CI/CD pipeline passes

## Final Integration Test

Create end-to-end test that exercises the complete grooming workflow:

```bash
# Run full simulation with grooming enabled
python -m fusion.cli.main --config grooming_test.ini --iterations 10
```

Verify:
- Requests are groomed to existing lightpaths
- Lightpath status dict is maintained
- Statistics are collected correctly
- No memory leaks or resource exhaustion

## Completion

After all tests pass, the grooming feature port is complete!

Return to: [Overview](00-overview.md) for next steps.
