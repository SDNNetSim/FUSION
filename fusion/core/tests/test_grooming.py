"""
Comprehensive tests for traffic grooming functionality.
"""

import networkx as nx
import pytest

from fusion.core.grooming import Grooming
from fusion.core.properties import GroomingProps, SDNProps
from fusion.core.sdn_controller import SDNController


class TestGroomingInitialization:
    """Tests for Grooming class initialization."""

    @pytest.fixture
    def engine_props(self) -> dict:
        """Provide basic engine properties for testing."""
        return {
            "is_grooming_enabled": True,
            "can_partially_serve": True,
            "cores_per_link": 7,
            "band_list": ["C"],
            "guard_slots": 1,
        }

    def test_grooming_init(self, engine_props: dict) -> None:
        """Test grooming initialization."""
        sdn_props = SDNProps()
        grooming = Grooming(engine_props, sdn_props)

        assert grooming.grooming_props is not None
        assert grooming.engine_props == engine_props
        assert grooming.sdn_props == sdn_props

    def test_grooming_has_required_methods(self, engine_props: dict) -> None:
        """Test that Grooming has all required methods."""
        sdn_props = SDNProps()
        grooming = Grooming(engine_props, sdn_props)

        assert hasattr(grooming, "handle_grooming")
        assert hasattr(grooming, "_end_to_end_grooming")
        assert hasattr(grooming, "_release_service")
        assert hasattr(grooming, "_find_path_max_bw")
        assert callable(grooming.handle_grooming)

    def test_grooming_props_initialization(self) -> None:
        """Test GroomingProps initialization."""
        props = GroomingProps()

        assert props.grooming_type is None
        assert props.lightpath_status_dict is None


class TestFindPathMaxBandwidth:
    """Test _find_path_max_bw method."""

    @pytest.fixture
    def engine_props(self) -> dict:
        """Provide engine properties."""
        return {
            "is_grooming_enabled": True,
            "cores_per_link": 7,
            "band_list": ["C"],
        }

    def test_find_path_no_lightpaths(self, engine_props: dict) -> None:
        """Test with no existing lightpaths."""
        sdn_props = SDNProps()
        sdn_props.lightpath_status_dict = {("A", "B"): {}}

        grooming = Grooming(engine_props, sdn_props)
        result = grooming._find_path_max_bw(("A", "B"))

        assert result is None

    def test_find_path_single_lightpath(self, engine_props: dict) -> None:
        """Test with single lightpath."""
        sdn_props = SDNProps()
        sdn_props.lightpath_status_dict = {
            ("A", "B"): {
                1: {
                    "path": ["A", "C", "B"],
                    "remaining_bandwidth": 100,
                    "is_degraded": False,
                }
            }
        }

        grooming = Grooming(engine_props, sdn_props)
        result = grooming._find_path_max_bw(("A", "B"))

        assert result is not None
        assert result["total_remaining_bandwidth"] == 100
        assert 1 in result["lp_id_list"]

    def test_find_path_skips_degraded(self, engine_props: dict) -> None:
        """Test that degraded lightpaths are skipped."""
        sdn_props = SDNProps()
        sdn_props.lightpath_status_dict = {
            ("A", "B"): {
                1: {
                    "path": ["A", "C", "B"],
                    "remaining_bandwidth": 100,
                    "is_degraded": True,  # Should be skipped
                },
                2: {
                    "path": ["A", "D", "B"],
                    "remaining_bandwidth": 50,
                    "is_degraded": False,
                },
            }
        }

        grooming = Grooming(engine_props, sdn_props)
        result = grooming._find_path_max_bw(("A", "B"))

        assert result is not None
        assert result["total_remaining_bandwidth"] == 50
        assert 2 in result["lp_id_list"]
        assert 1 not in result["lp_id_list"]

    def test_find_path_skips_zero_bandwidth(self, engine_props: dict) -> None:
        """Test that lightpaths with zero bandwidth are skipped."""
        sdn_props = SDNProps()
        sdn_props.lightpath_status_dict = {
            ("A", "B"): {
                1: {
                    "path": ["A", "C", "B"],
                    "remaining_bandwidth": 0,  # No bandwidth
                    "is_degraded": False,
                },
                2: {
                    "path": ["A", "D", "B"],
                    "remaining_bandwidth": 50,
                    "is_degraded": False,
                },
            }
        }

        grooming = Grooming(engine_props, sdn_props)
        result = grooming._find_path_max_bw(("A", "B"))

        assert result is not None
        assert result["total_remaining_bandwidth"] == 50
        assert 2 in result["lp_id_list"]
        assert 1 not in result["lp_id_list"]

    def test_find_path_groups_by_path(self, engine_props: dict) -> None:
        """Test that lightpaths on same path are grouped."""
        sdn_props = SDNProps()
        sdn_props.lightpath_status_dict = {
            ("A", "B"): {
                1: {
                    "path": ["A", "C", "B"],
                    "remaining_bandwidth": 50,
                    "is_degraded": False,
                },
                2: {
                    "path": ["A", "C", "B"],  # Same path
                    "remaining_bandwidth": 30,
                    "is_degraded": False,
                },
                3: {
                    "path": ["A", "D", "B"],  # Different path
                    "remaining_bandwidth": 60,
                    "is_degraded": False,
                },
            }
        }

        grooming = Grooming(engine_props, sdn_props)
        result = grooming._find_path_max_bw(("A", "B"))

        # Should select the A-C-B path group with total 80 bandwidth
        assert result is not None
        assert result["total_remaining_bandwidth"] == 80
        assert set(result["lp_id_list"]) == {1, 2}

    def test_find_path_reverse_paths_grouped(self, engine_props: dict) -> None:
        """Test that forward and reverse paths are treated as same group."""
        sdn_props = SDNProps()
        sdn_props.lightpath_status_dict = {
            ("A", "B"): {
                1: {
                    "path": ["A", "C", "B"],
                    "remaining_bandwidth": 50,
                    "is_degraded": False,
                },
                2: {
                    "path": ["B", "C", "A"],  # Reverse of path 1
                    "remaining_bandwidth": 30,
                    "is_degraded": False,
                },
            }
        }

        grooming = Grooming(engine_props, sdn_props)
        result = grooming._find_path_max_bw(("A", "B"))

        # Should group both paths together
        assert result is not None
        assert result["total_remaining_bandwidth"] == 80


class TestEndToEndGrooming:
    """Test _end_to_end_grooming method."""

    @pytest.fixture
    def engine_props(self) -> dict:
        """Provide engine properties."""
        return {
            "is_grooming_enabled": True,
            "cores_per_link": 7,
            "band_list": ["C"],
        }

    def test_grooming_no_existing_lightpaths(self, engine_props: dict) -> None:
        """Test grooming when no lightpaths exist."""
        sdn_props = SDNProps()
        sdn_props.source = "A"
        sdn_props.destination = "B"
        sdn_props.bandwidth = 100
        sdn_props.lightpath_status_dict = {}

        grooming = Grooming(engine_props, sdn_props)
        result = grooming._end_to_end_grooming()

        assert result is False

    def test_grooming_full_allocation(self, engine_props: dict) -> None:
        """Test full grooming without creating new lightpath."""
        sdn_props = SDNProps()
        sdn_props.source = "A"
        sdn_props.destination = "B"
        sdn_props.bandwidth = 50
        sdn_props.request_id = 100
        sdn_props.arrive = 10.0
        sdn_props.bandwidth_list = []
        sdn_props.core_list = []
        sdn_props.band_list = []
        sdn_props.start_slot_list = []
        sdn_props.end_slot_list = []
        sdn_props.modulation_list = []
        sdn_props.crosstalk_list = []
        sdn_props.snr_list = []
        sdn_props.xt_list = []
        sdn_props.lightpath_bandwidth_list = []
        sdn_props.lightpath_id_list = []
        sdn_props.lightpath_status_dict = {
            ("A", "B"): {
                1: {
                    "path": ["A", "C", "B"],
                    "path_weight": 100.0,
                    "remaining_bandwidth": 100,
                    "lightpath_bandwidth": 200,
                    "is_degraded": False,
                    "core": 0,
                    "band": "C",
                    "start_slot": 10,
                    "end_slot": 20,
                    "mod_format": "QPSK",
                    "snr_cost": 0.5,
                    "xt_cost": 0.0,
                    "requests_dict": {},
                    "time_bw_usage": {},
                }
            }
        }

        grooming = Grooming(engine_props, sdn_props)
        result = grooming._end_to_end_grooming()

        assert result is True
        assert sdn_props.was_routed is True
        assert sdn_props.was_groomed is True
        assert sdn_props.was_partially_groomed is False
        assert sdn_props.number_of_transponders == 0
        assert sdn_props.remaining_bw == "0"

        # Check lightpath was updated
        lp_info = sdn_props.lightpath_status_dict[("A", "B")][1]
        assert lp_info["remaining_bandwidth"] == 50
        assert 100 in lp_info["requests_dict"]
        assert lp_info["requests_dict"][100] == 50

    def test_grooming_partial_allocation(self, engine_props: dict) -> None:
        """Test partial grooming when not enough bandwidth available."""
        sdn_props = SDNProps()
        sdn_props.source = "A"
        sdn_props.destination = "B"
        sdn_props.bandwidth = 150
        sdn_props.request_id = 100
        sdn_props.arrive = 10.0
        sdn_props.bandwidth_list = []
        sdn_props.core_list = []
        sdn_props.band_list = []
        sdn_props.start_slot_list = []
        sdn_props.end_slot_list = []
        sdn_props.modulation_list = []
        sdn_props.crosstalk_list = []
        sdn_props.snr_list = []
        sdn_props.xt_list = []
        sdn_props.lightpath_bandwidth_list = []
        sdn_props.lightpath_id_list = []
        sdn_props.lightpath_status_dict = {
            ("A", "B"): {
                1: {
                    "path": ["A", "C", "B"],
                    "path_weight": 100.0,
                    "remaining_bandwidth": 100,  # Not enough for full 150
                    "lightpath_bandwidth": 200,
                    "is_degraded": False,
                    "core": 0,
                    "band": "C",
                    "start_slot": 10,
                    "end_slot": 20,
                    "mod_format": "QPSK",
                    "snr_cost": 0.5,
                    "xt_cost": 0.0,
                    "requests_dict": {},
                    "time_bw_usage": {},
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
        lp_info = sdn_props.lightpath_status_dict[("A", "B")][1]
        assert lp_info["remaining_bandwidth"] == 0

    def test_grooming_multiple_lightpaths(self, engine_props: dict) -> None:
        """Test grooming across multiple lightpaths on same path."""
        sdn_props = SDNProps()
        sdn_props.source = "A"
        sdn_props.destination = "B"
        sdn_props.bandwidth = 120
        sdn_props.request_id = 100
        sdn_props.arrive = 10.0
        sdn_props.bandwidth_list = []
        sdn_props.core_list = []
        sdn_props.band_list = []
        sdn_props.start_slot_list = []
        sdn_props.end_slot_list = []
        sdn_props.modulation_list = []
        sdn_props.crosstalk_list = []
        sdn_props.snr_list = []
        sdn_props.xt_list = []
        sdn_props.lightpath_bandwidth_list = []
        sdn_props.lightpath_id_list = []
        sdn_props.lightpath_status_dict = {
            ("A", "B"): {
                1: {
                    "path": ["A", "C", "B"],
                    "path_weight": 100.0,
                    "remaining_bandwidth": 80,
                    "lightpath_bandwidth": 200,
                    "is_degraded": False,
                    "core": 0,
                    "band": "C",
                    "start_slot": 10,
                    "end_slot": 20,
                    "mod_format": "QPSK",
                    "snr_cost": 0.5,
                    "xt_cost": 0.0,
                    "requests_dict": {},
                    "time_bw_usage": {},
                },
                2: {
                    "path": ["A", "C", "B"],  # Same path
                    "path_weight": 100.0,
                    "remaining_bandwidth": 50,
                    "lightpath_bandwidth": 200,
                    "is_degraded": False,
                    "core": 1,
                    "band": "C",
                    "start_slot": 30,
                    "end_slot": 40,
                    "mod_format": "16QAM",
                    "snr_cost": 0.3,
                    "xt_cost": 0.0,
                    "requests_dict": {},
                    "time_bw_usage": {},
                },
            }
        }

        grooming = Grooming(engine_props, sdn_props)
        result = grooming._end_to_end_grooming()

        assert result is True
        assert sdn_props.was_groomed is True
        assert sdn_props.was_partially_groomed is False
        assert len(sdn_props.lightpath_id_list) == 2

        # Check both lightpaths were used
        lp1 = sdn_props.lightpath_status_dict[("A", "B")][1]
        lp2 = sdn_props.lightpath_status_dict[("A", "B")][2]
        assert lp1["remaining_bandwidth"] == 0  # Used all 80
        assert lp2["remaining_bandwidth"] == 10  # Used 40 out of 50


class TestReleaseService:
    """Test _release_service method."""

    @pytest.fixture
    def engine_props(self) -> dict:
        """Provide engine properties."""
        return {
            "is_grooming_enabled": True,
            "cores_per_link": 7,
            "band_list": ["C"],
        }

    def test_release_single_lightpath(self, engine_props: dict) -> None:
        """Test releasing a request from single lightpath."""
        sdn_props = SDNProps()
        sdn_props.source = "A"
        sdn_props.destination = "B"
        sdn_props.request_id = 100
        sdn_props.depart = 50.0
        sdn_props.remaining_bw = 50
        sdn_props.lightpath_id_list = [1]
        sdn_props.lightpath_bandwidth_list = [200]
        sdn_props.lightpath_status_dict = {
            ("A", "B"): {
                1: {
                    "remaining_bandwidth": 100,
                    "lightpath_bandwidth": 200,
                    "requests_dict": {100: 50},  # This request allocated 50
                    "time_bw_usage": {0.0: 25.0},
                }
            }
        }

        grooming = Grooming(engine_props, sdn_props)
        release_list = grooming._release_service()

        # Bandwidth should be freed
        lp_info = sdn_props.lightpath_status_dict[("A", "B")][1]
        assert lp_info["remaining_bandwidth"] == 150  # 100 + 50
        assert 100 not in lp_info["requests_dict"]
        assert len(release_list) == 0  # Still has capacity, not released

    def test_release_lightpath_becomes_empty(self, engine_props: dict) -> None:
        """Test releasing request that empties lightpath."""
        sdn_props = SDNProps()
        sdn_props.source = "A"
        sdn_props.destination = "B"
        sdn_props.request_id = 100
        sdn_props.depart = 50.0
        sdn_props.remaining_bw = 200
        sdn_props.lightpath_id_list = [1]
        sdn_props.lightpath_bandwidth_list = [200]
        sdn_props.lightpath_status_dict = {
            ("A", "B"): {
                1: {
                    "remaining_bandwidth": 0,  # Fully utilized
                    "lightpath_bandwidth": 200,
                    "requests_dict": {100: 200},  # Only this request
                    "time_bw_usage": {0.0: 100.0},
                }
            }
        }

        grooming = Grooming(engine_props, sdn_props)
        release_list = grooming._release_service()

        # Lightpath should be marked for release
        assert 1 in release_list
        lp_info = sdn_props.lightpath_status_dict[("A", "B")][1]
        assert lp_info["remaining_bandwidth"] == 200  # Fully freed

    def test_release_multiple_lightpaths(self, engine_props: dict) -> None:
        """Test releasing request from multiple lightpaths."""
        sdn_props = SDNProps()
        sdn_props.source = "A"
        sdn_props.destination = "B"
        sdn_props.request_id = 100
        sdn_props.depart = 50.0
        sdn_props.remaining_bw = 150
        sdn_props.lightpath_id_list = [1, 2]
        sdn_props.lightpath_bandwidth_list = [200, 200]
        sdn_props.lightpath_status_dict = {
            ("A", "B"): {
                1: {
                    "remaining_bandwidth": 0,
                    "lightpath_bandwidth": 200,
                    "requests_dict": {100: 100},
                    "time_bw_usage": {0.0: 50.0},
                },
                2: {
                    "remaining_bandwidth": 150,
                    "lightpath_bandwidth": 200,
                    "requests_dict": {100: 50},
                    "time_bw_usage": {0.0: 25.0},
                },
            }
        }

        grooming = Grooming(engine_props, sdn_props)
        release_list = grooming._release_service()

        # Check both lightpaths released
        lp1 = sdn_props.lightpath_status_dict[("A", "B")][1]
        lp2 = sdn_props.lightpath_status_dict[("A", "B")][2]
        assert lp1["remaining_bandwidth"] == 100
        assert lp2["remaining_bandwidth"] == 200  # Fully freed
        assert 2 in release_list  # LP2 should be marked for release
        assert 1 not in release_list  # LP1 still has capacity


class TestHandleGrooming:
    """Test handle_grooming dispatcher method."""

    @pytest.fixture
    def engine_props(self) -> dict:
        """Provide engine properties."""
        return {
            "is_grooming_enabled": True,
            "cores_per_link": 7,
            "band_list": ["C"],
        }

    def test_handle_grooming_arrival(self, engine_props: dict) -> None:
        """Test handling arrival request."""
        sdn_props = SDNProps()
        sdn_props.source = "A"
        sdn_props.destination = "B"
        sdn_props.bandwidth = 100
        sdn_props.request_id = 100
        sdn_props.arrive = 10.0
        sdn_props.lightpath_status_dict = {}

        grooming = Grooming(engine_props, sdn_props)
        result = grooming.handle_grooming("arrival")

        assert isinstance(result, bool)
        assert result is False  # No lightpaths available

    def test_handle_grooming_release(self, engine_props: dict) -> None:
        """Test handling release request."""
        sdn_props = SDNProps()
        sdn_props.source = "A"
        sdn_props.destination = "B"
        sdn_props.request_id = 100
        sdn_props.bandwidth = 100  # Required for remaining_bw calculation
        sdn_props.lightpath_id_list = []
        sdn_props.lightpath_bandwidth_list = []
        sdn_props.lightpath_status_dict = {("A", "B"): {}}

        grooming = Grooming(engine_props, sdn_props)
        result = grooming.handle_grooming("release")

        assert isinstance(result, list)


class TestGroomingIntegration:
    """Integration tests with SDN controller."""

    @pytest.fixture
    def engine_props(self) -> dict:
        """Provide engine properties."""
        return {
            "is_grooming_enabled": True,
            "cores_per_link": 7,
            "band_list": ["C"],
            "guard_slots": 1,
            "snr_type": None,
        }

    @pytest.fixture
    def setup_topology(self) -> nx.Graph:
        """Create simple test topology."""
        topology = nx.Graph()
        topology.add_edge("A", "B", weight=100, length=100)
        topology.add_edge("A", "C", weight=150, length=150)
        topology.add_edge("C", "B", weight=120, length=120)
        return topology

    def test_sdn_controller_has_grooming(self, engine_props: dict) -> None:
        """Test SDN controller initializes grooming object."""
        sdn = SDNController(engine_props)

        assert hasattr(sdn, "grooming_obj")
        assert isinstance(sdn.grooming_obj, Grooming)

    def test_grooming_obj_shares_sdn_props(self, engine_props: dict) -> None:
        """Test that grooming object shares SDN props."""
        sdn = SDNController(engine_props)

        # They should share the same SDNProps instance
        assert sdn.grooming_obj.sdn_props is sdn.sdn_props


# Fixtures for shared use
@pytest.fixture
def basic_engine_props() -> dict:
    """Basic engine properties."""
    return {
        "is_grooming_enabled": True,
        "cores_per_link": 7,
        "band_list": ["C"],
        "guard_slots": 1,
    }


@pytest.fixture
def test_topology() -> nx.Graph:
    """Create simple test topology."""
    topology = nx.Graph()
    topology.add_edge("A", "B", weight=100, length=100)
    topology.add_edge("A", "C", weight=150, length=150)
    topology.add_edge("C", "B", weight=120, length=120)
    topology.add_edge("B", "D", weight=130, length=130)
    return topology
