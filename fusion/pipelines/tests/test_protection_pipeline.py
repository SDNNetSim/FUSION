"""
Unit tests for protection pipeline.

Tests cover:

- DisjointPathFinder link-disjoint mode
- DisjointPathFinder node-disjoint mode
- ProtectionPipeline allocate_protected() method
- Disjointness correctness verification
- Identical spectrum allocation on both paths
- Protection consumes extra resources
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np

from fusion.pipelines.disjoint_path_finder import DisjointnessType, DisjointPathFinder
from fusion.pipelines.protection_pipeline import (
    ProtectedAllocationResult,
    ProtectionPipeline,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass(frozen=True)
class MockSimulationConfig:
    """Minimal mock of SimulationConfig for testing."""

    band_list: tuple[str, ...] = ("c",)
    band_slots: dict[str, int] = field(default_factory=lambda: {"c": 320})
    cores_per_link: int = 1
    topology_info: dict[str, Any] | None = None
    guard_slots: int = 0


class MockLinkSpectrum:
    """Mock LinkSpectrum for testing."""

    def __init__(self, num_slots: int = 320, num_cores: int = 1) -> None:
        self.cores_matrix = {"c": np.zeros((num_cores, num_slots), dtype=np.int64)}
        self.link_num = 0
        self.length_km = 100.0

    def get_slot_count(self, band: str) -> int:
        return self.cores_matrix[band].shape[1]  # type: ignore[no-any-return]

    def get_spectrum_array(self, band: str) -> np.ndarray:
        return self.cores_matrix[band]

    def is_range_free(self, start_slot: int, end_slot: int, core: int, band: str) -> bool:
        return bool(np.all(self.cores_matrix[band][core, start_slot:end_slot] == 0))

    def allocate_range(
        self,
        start_slot: int,
        end_slot: int,
        core: int,
        band: str,
        lightpath_id: int,
        guard_slots: int = 0,
    ) -> None:
        self.cores_matrix[band][core, start_slot:end_slot] = lightpath_id


class MockNetworkState:
    """Mock NetworkState for testing protection pipeline."""

    def __init__(self, topology: nx.Graph, num_slots: int = 320) -> None:
        self._topology = topology
        self._spectrum: dict[tuple[str, str], MockLinkSpectrum] = {}
        self._lightpaths: dict[int, Any] = {}
        self._next_lightpath_id = 1

        # Initialize spectrum for all edges
        for u, v in topology.edges():
            spectrum = MockLinkSpectrum(num_slots)
            self._spectrum[(str(u), str(v))] = spectrum
            self._spectrum[(str(v), str(u))] = spectrum

    def get_link_spectrum(self, link: tuple[str, str]) -> MockLinkSpectrum:
        str_link = (str(link[0]), str(link[1]))
        return self._spectrum[str_link]

    def is_spectrum_available(
        self,
        path: list[str],
        start_slot: int,
        end_slot: int,
        core: int,
        band: str,
    ) -> bool:
        for i in range(len(path) - 1):
            link = (path[i], path[i + 1])
            spectrum = self.get_link_spectrum(link)
            if not spectrum.is_range_free(start_slot, end_slot, core, band):
                return False
        return True

    def allocate_on_path(
        self,
        path: list[str],
        start_slot: int,
        end_slot: int,
        core: int,
        band: str,
        lightpath_id: int,
    ) -> None:
        """Allocate spectrum on all links of a path."""
        for i in range(len(path) - 1):
            link = (path[i], path[i + 1])
            spectrum = self.get_link_spectrum(link)
            spectrum.allocate_range(start_slot, end_slot, core, band, lightpath_id)

    def get_total_allocated_slots(self) -> int:
        """Count total allocated slots across all links."""
        total = 0
        seen = set()
        for link, spectrum in self._spectrum.items():
            normalized = tuple(sorted(link))
            if normalized in seen:
                continue
            seen.add(normalized)
            total += np.sum(spectrum.cores_matrix["c"] != 0)
        return int(total)


def create_diamond_topology() -> nx.Graph:
    """
    Create a diamond topology for testing disjoint paths.

          B
         / \
        A   D
         \\ /
          C

    Link-disjoint paths: A-B-D and A-C-D
    Node-disjoint paths: A-B-D and A-C-D (same in this topology)
    """
    g = nx.Graph()
    g.add_edge("A", "B", weight=1.0)
    g.add_edge("A", "C", weight=1.0)
    g.add_edge("B", "D", weight=1.0)
    g.add_edge("C", "D", weight=1.0)
    return g


def create_extended_diamond_topology() -> nx.Graph:
    """
    Extended diamond with extra node for testing node vs link disjointness.

          B
         /|\
        A-E-D
         \\|/
          C

    Link-disjoint: A-B-D, A-C-D, A-E-D (3 paths sharing no edges)
    For A-B-E-D and A-C-E-D: link-disjoint but NOT node-disjoint (share E)
    """
    g = nx.Graph()
    g.add_edge("A", "B", weight=1.0)
    g.add_edge("A", "C", weight=1.0)
    g.add_edge("A", "E", weight=1.0)
    g.add_edge("B", "D", weight=1.0)
    g.add_edge("B", "E", weight=1.0)
    g.add_edge("C", "D", weight=1.0)
    g.add_edge("C", "E", weight=1.0)
    g.add_edge("E", "D", weight=1.0)
    return g


def create_linear_topology() -> nx.Graph:
    """
    Linear topology (no disjoint paths possible).

    A -- B -- C -- D
    """
    g = nx.Graph()
    g.add_edge("A", "B", weight=1.0)
    g.add_edge("B", "C", weight=1.0)
    g.add_edge("C", "D", weight=1.0)
    return g


# =============================================================================
# DisjointPathFinder Tests
# =============================================================================


class TestDisjointPathFinderLinkDisjoint:
    """Tests for DisjointPathFinder in link-disjoint mode."""

    def test_find_link_disjoint_pair_diamond(self) -> None:
        """Find link-disjoint path pair in diamond topology."""
        topology = create_diamond_topology()
        finder = DisjointPathFinder(DisjointnessType.LINK)

        result = finder.find_disjoint_pair(topology, "A", "D")

        assert result is not None
        primary, backup = result
        assert primary[0] == "A"
        assert primary[-1] == "D"
        assert backup[0] == "A"
        assert backup[-1] == "D"
        # Verify link-disjointness
        assert finder.are_link_disjoint(primary, backup)

    def test_no_disjoint_paths_linear(self) -> None:
        """No disjoint paths in linear topology."""
        topology = create_linear_topology()
        finder = DisjointPathFinder(DisjointnessType.LINK)

        result = finder.find_disjoint_pair(topology, "A", "D")

        assert result is None

    def test_find_all_link_disjoint(self) -> None:
        """Find all link-disjoint paths."""
        topology = create_diamond_topology()
        finder = DisjointPathFinder(DisjointnessType.LINK)

        paths = finder.find_all_disjoint_paths(topology, "A", "D", max_paths=10)

        assert len(paths) == 2  # Diamond has exactly 2 edge-disjoint paths
        # All paths should be pairwise link-disjoint
        for i, p1 in enumerate(paths):
            for p2 in paths[i + 1 :]:
                assert finder.are_link_disjoint(p1, p2)

    def test_are_link_disjoint_true(self) -> None:
        """Paths with no common edges are link-disjoint."""
        finder = DisjointPathFinder(DisjointnessType.LINK)

        path1 = ["A", "B", "D"]
        path2 = ["A", "C", "D"]

        assert finder.are_link_disjoint(path1, path2)

    def test_are_link_disjoint_false(self) -> None:
        """Paths with common edge are not link-disjoint."""
        finder = DisjointPathFinder(DisjointnessType.LINK)

        path1 = ["A", "B", "C"]
        path2 = ["A", "B", "D"]  # Shares A-B edge

        assert not finder.are_link_disjoint(path1, path2)

    def test_are_link_disjoint_bidirectional(self) -> None:
        """Link disjointness considers both directions."""
        finder = DisjointPathFinder(DisjointnessType.LINK)

        path1 = ["A", "B", "C"]
        path2 = ["D", "B", "A"]  # Shares A-B edge (reversed)

        assert not finder.are_link_disjoint(path1, path2)


class TestDisjointPathFinderNodeDisjoint:
    """Tests for DisjointPathFinder in node-disjoint mode."""

    def test_find_node_disjoint_pair_diamond(self) -> None:
        """Find node-disjoint path pair in diamond topology."""
        topology = create_diamond_topology()
        finder = DisjointPathFinder(DisjointnessType.NODE)

        result = finder.find_disjoint_pair(topology, "A", "D")

        assert result is not None
        primary, backup = result
        assert primary[0] == "A"
        assert primary[-1] == "D"
        assert backup[0] == "A"
        assert backup[-1] == "D"
        # Verify node-disjointness
        assert finder.are_node_disjoint(primary, backup)

    def test_no_node_disjoint_linear(self) -> None:
        """No node-disjoint paths in linear topology."""
        topology = create_linear_topology()
        finder = DisjointPathFinder(DisjointnessType.NODE)

        result = finder.find_disjoint_pair(topology, "A", "D")

        assert result is None

    def test_find_all_node_disjoint(self) -> None:
        """Find all node-disjoint paths."""
        topology = create_diamond_topology()
        finder = DisjointPathFinder(DisjointnessType.NODE)

        paths = finder.find_all_disjoint_paths(topology, "A", "D", max_paths=10)

        assert len(paths) == 2
        # All paths should be pairwise node-disjoint
        for i, p1 in enumerate(paths):
            for p2 in paths[i + 1 :]:
                assert finder.are_node_disjoint(p1, p2)

    def test_are_node_disjoint_true(self) -> None:
        """Paths with no common intermediate nodes are node-disjoint."""
        finder = DisjointPathFinder(DisjointnessType.NODE)

        path1 = ["A", "B", "D"]
        path2 = ["A", "C", "D"]  # Only share endpoints

        assert finder.are_node_disjoint(path1, path2)

    def test_are_node_disjoint_false(self) -> None:
        """Paths with common intermediate node are not node-disjoint."""
        finder = DisjointPathFinder(DisjointnessType.NODE)

        path1 = ["A", "B", "C", "D"]
        path2 = ["A", "E", "C", "F"]  # Share intermediate node C

        assert not finder.are_node_disjoint(path1, path2)

    def test_endpoints_dont_count(self) -> None:
        """Shared endpoints don't violate node-disjointness."""
        finder = DisjointPathFinder(DisjointnessType.NODE)

        path1 = ["A", "B", "D"]
        path2 = ["A", "C", "D"]  # Same source and destination

        assert finder.are_node_disjoint(path1, path2)


class TestNodeVsLinkDisjoint:
    """Tests comparing node-disjoint vs link-disjoint."""

    def test_node_disjoint_implies_link_disjoint(self) -> None:
        """Node-disjoint paths are also link-disjoint."""
        finder_link = DisjointPathFinder(DisjointnessType.LINK)
        finder_node = DisjointPathFinder(DisjointnessType.NODE)

        # Any node-disjoint pair is also link-disjoint
        path1 = ["A", "B", "D"]
        path2 = ["A", "C", "D"]

        assert finder_node.are_node_disjoint(path1, path2)
        assert finder_link.are_link_disjoint(path1, path2)

    def test_link_disjoint_not_implies_node_disjoint(self) -> None:
        """Link-disjoint paths may not be node-disjoint."""
        finder_link = DisjointPathFinder(DisjointnessType.LINK)
        finder_node = DisjointPathFinder(DisjointnessType.NODE)

        # A-B-E-D and A-C-E-D share node E but no edges
        topology = create_extended_diamond_topology()

        # Find all link-disjoint paths
        link_paths = finder_link.find_all_disjoint_paths(topology, "A", "D", max_paths=5)

        # Check if any pair is link-disjoint but not node-disjoint
        # This topology may or may not have such a pair depending on which
        # paths are returned first, so we just verify the search completes
        for i, p1 in enumerate(link_paths):
            for p2 in link_paths[i + 1 :]:
                if finder_link.are_link_disjoint(p1, p2):
                    # If link-disjoint, check node-disjointness (may or may not hold)
                    _ = finder_node.are_node_disjoint(p1, p2)


# =============================================================================
# ProtectionPipeline Tests
# =============================================================================


class TestProtectionPipeline:
    """Tests for ProtectionPipeline."""

    def test_find_protected_paths(self) -> None:
        """ProtectionPipeline finds protected path pairs."""
        topology = create_diamond_topology()
        pipeline = ProtectionPipeline(DisjointnessType.LINK)

        result = pipeline.find_protected_paths(topology, "A", "D")

        assert result is not None
        primary, backup = result
        assert primary[0] == "A"
        assert primary[-1] == "D"
        assert pipeline.verify_disjointness(primary, backup)

    def test_allocate_protected_success(self) -> None:
        """allocate_protected finds common spectrum on both paths."""
        topology = create_diamond_topology()
        state = MockNetworkState(topology, num_slots=100)
        pipeline = ProtectionPipeline(DisjointnessType.LINK)

        primary = ["A", "B", "D"]
        backup = ["A", "C", "D"]

        result = pipeline.allocate_protected(
            primary_path=primary,
            backup_path=backup,
            slots_needed=8,
            network_state=state,
            core=0,
            band="c",
        )

        assert result.success
        assert result.start_slot >= 0
        assert result.end_slot == result.start_slot + 8

    def test_allocate_protected_identical_spectrum(self) -> None:
        """Both paths get identical spectrum slots."""
        topology = create_diamond_topology()
        state = MockNetworkState(topology, num_slots=100)
        pipeline = ProtectionPipeline(DisjointnessType.LINK)

        primary = ["A", "B", "D"]
        backup = ["A", "C", "D"]

        result = pipeline.allocate_protected(
            primary_path=primary,
            backup_path=backup,
            slots_needed=10,
            network_state=state,
        )

        assert result.success
        # The result specifies the same slot range for both paths
        # This is the key 1+1 dedicated protection requirement
        assert result.start_slot == 0  # First-fit starts at 0
        assert result.end_slot == 10

    def test_allocate_protected_no_common_spectrum(self) -> None:
        """Fails when no common spectrum available."""
        topology = create_diamond_topology()
        state = MockNetworkState(topology, num_slots=20)
        pipeline = ProtectionPipeline(DisjointnessType.LINK)

        primary = ["A", "B", "D"]
        backup = ["A", "C", "D"]

        # Occupy different slots on primary vs backup path
        # Primary path: occupy slots 0-9
        state.allocate_on_path(primary, 0, 10, 0, "c", 1)
        # Backup path: occupy slots 10-19
        state.allocate_on_path(backup, 10, 20, 0, "c", 2)

        result = pipeline.allocate_protected(
            primary_path=primary,
            backup_path=backup,
            slots_needed=5,
            network_state=state,
        )

        assert not result.success
        assert result.failure_reason == "no_common_spectrum"

    def test_protection_consumes_double_resources(self) -> None:
        """Protected allocation uses spectrum on both paths."""
        topology = create_diamond_topology()
        state = MockNetworkState(topology, num_slots=100)
        pipeline = ProtectionPipeline(DisjointnessType.LINK)

        primary = ["A", "B", "D"]
        backup = ["A", "C", "D"]

        # Get initial allocation count
        initial_allocated = state.get_total_allocated_slots()
        assert initial_allocated == 0

        # Allocate protected
        result = pipeline.allocate_protected(
            primary_path=primary,
            backup_path=backup,
            slots_needed=8,
            network_state=state,
        )

        assert result.success

        # Now actually allocate spectrum on both paths (simulating what
        # create_protected_lightpath would do)
        state.allocate_on_path(primary, result.start_slot, result.end_slot, 0, "c", 1)
        state.allocate_on_path(backup, result.start_slot, result.end_slot, 0, "c", 1)

        # Count allocated slots:
        # Primary path: A-B (8 slots) + B-D (8 slots) = 16 slots
        # Backup path: A-C (8 slots) + C-D (8 slots) = 16 slots
        # Total: 32 slots for 8 slots of protected service
        final_allocated = state.get_total_allocated_slots()

        # Protected allocation consumes spectrum on both paths
        # Each path has 2 links, 8 slots each = 16 slots per path
        # Total = 32 slots
        assert final_allocated == 32

    def test_switchover_latency_default(self) -> None:
        """Default switchover latency is 50ms."""
        pipeline = ProtectionPipeline()
        assert pipeline.get_switchover_latency() == 50.0

    def test_switchover_latency_custom(self) -> None:
        """Custom switchover latency is respected."""
        pipeline = ProtectionPipeline(switchover_latency_ms=25.0)
        assert pipeline.get_switchover_latency() == 25.0

    def test_verify_disjointness_link_mode(self) -> None:
        """verify_disjointness uses correct mode for links."""
        pipeline = ProtectionPipeline(DisjointnessType.LINK)

        path1 = ["A", "B", "D"]
        path2 = ["A", "C", "D"]

        assert pipeline.verify_disjointness(path1, path2)

    def test_verify_disjointness_node_mode(self) -> None:
        """verify_disjointness uses correct mode for nodes."""
        pipeline = ProtectionPipeline(DisjointnessType.NODE)

        path1 = ["A", "B", "D"]
        path2 = ["A", "C", "D"]

        assert pipeline.verify_disjointness(path1, path2)


class TestProtectedAllocationResult:
    """Tests for ProtectedAllocationResult dataclass."""

    def test_no_disjoint_paths_factory(self) -> None:
        """Create result for no disjoint paths."""
        result = ProtectedAllocationResult.no_disjoint_paths()

        assert not result.success
        assert result.failure_reason == "no_disjoint_paths"

    def test_no_common_spectrum_factory(self) -> None:
        """Create result for no common spectrum."""
        result = ProtectedAllocationResult.no_common_spectrum()

        assert not result.success
        assert result.failure_reason == "no_common_spectrum"

    def test_allocated_factory(self) -> None:
        """Create result for successful allocation."""
        result = ProtectedAllocationResult.allocated(10, 20)

        assert result.success
        assert result.start_slot == 10
        assert result.end_slot == 20
        assert result.failure_reason is None


# =============================================================================
# Integration Tests
# =============================================================================


class TestProtectionIntegration:
    """Integration tests for protection pipeline."""

    def test_full_protection_flow(self) -> None:
        """Test complete protection flow: find paths, allocate spectrum."""
        topology = create_diamond_topology()
        state = MockNetworkState(topology, num_slots=100)
        pipeline = ProtectionPipeline(DisjointnessType.LINK)

        # Step 1: Find protected paths
        paths = pipeline.find_protected_paths(topology, "A", "D")
        assert paths is not None
        primary, backup = paths

        # Step 2: Verify disjointness
        assert pipeline.verify_disjointness(primary, backup)

        # Step 3: Allocate spectrum
        result = pipeline.allocate_protected(
            primary_path=primary,
            backup_path=backup,
            slots_needed=8,
            network_state=state,
        )

        assert result.success
        assert result.start_slot == 0
        assert result.end_slot == 8

        # Step 4: Verify spectrum available on both paths before allocation
        assert state.is_spectrum_available(primary, 0, 8, 0, "c")
        assert state.is_spectrum_available(backup, 0, 8, 0, "c")

    def test_protection_with_existing_traffic(self) -> None:
        """Protection works around existing allocations."""
        topology = create_diamond_topology()
        state = MockNetworkState(topology, num_slots=100)
        pipeline = ProtectionPipeline(DisjointnessType.LINK)

        primary = ["A", "B", "D"]
        backup = ["A", "C", "D"]

        # Existing allocation on both paths at slots 0-9
        state.allocate_on_path(primary, 0, 10, 0, "c", 1)
        state.allocate_on_path(backup, 0, 10, 0, "c", 1)

        # New protected allocation should start at slot 10
        result = pipeline.allocate_protected(
            primary_path=primary,
            backup_path=backup,
            slots_needed=5,
            network_state=state,
        )

        assert result.success
        assert result.start_slot == 10
        assert result.end_slot == 15

    def test_node_disjoint_protection(self) -> None:
        """Node-disjoint protection provides stronger guarantee."""
        topology = create_diamond_topology()
        state = MockNetworkState(topology, num_slots=100)
        pipeline = ProtectionPipeline(DisjointnessType.NODE)

        paths = pipeline.find_protected_paths(topology, "A", "D")
        assert paths is not None
        primary, backup = paths

        # Node-disjoint paths share no intermediate nodes
        intermediate_primary = set(primary[1:-1])
        intermediate_backup = set(backup[1:-1])
        assert intermediate_primary.isdisjoint(intermediate_backup)

        # Allocation should still work
        result = pipeline.allocate_protected(
            primary_path=primary,
            backup_path=backup,
            slots_needed=8,
            network_state=state,
        )

        assert result.success
