"""
Unit tests for routing strategies.

Tests cover:
- RouteConstraints creation and manipulation
- KShortestPathStrategy route selection
- LoadBalancedStrategy utilization scoring
- ProtectionAwareStrategy disjoint path finding

Phase: P3.1.e - Routing Strategy Pattern (Gap Analysis)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from fusion.pipelines.routing_strategies import (
    KShortestPathStrategy,
    LoadBalancedStrategy,
    ProtectionAwareStrategy,
    RouteConstraints,
)


# =============================================================================
# RouteConstraints Tests
# =============================================================================


class TestRouteConstraints:
    """Tests for RouteConstraints dataclass."""

    def test_default_values(self) -> None:
        """RouteConstraints has sensible defaults."""
        constraints = RouteConstraints()

        assert constraints.max_hops is None
        assert constraints.max_length_km is None
        assert constraints.min_bandwidth_gbps is None
        assert constraints.exclude_links == frozenset()
        assert constraints.exclude_nodes == frozenset()
        assert constraints.required_modulation is None
        assert constraints.protection_mode is False

    def test_create_with_values(self) -> None:
        """RouteConstraints can be created with specific values."""
        constraints = RouteConstraints(
            max_hops=5,
            max_length_km=1000.0,
            exclude_links=frozenset({("A", "B")}),
            protection_mode=True,
        )

        assert constraints.max_hops == 5
        assert constraints.max_length_km == 1000.0
        assert ("A", "B") in constraints.exclude_links
        assert constraints.protection_mode is True

    def test_with_exclusions_adds_links(self) -> None:
        """with_exclusions adds new links to exclusion set."""
        original = RouteConstraints(
            exclude_links=frozenset({("A", "B")}),
        )

        updated = original.with_exclusions(links={("C", "D")})

        assert ("A", "B") in updated.exclude_links
        assert ("C", "D") in updated.exclude_links
        # Original unchanged
        assert ("C", "D") not in original.exclude_links

    def test_with_exclusions_adds_nodes(self) -> None:
        """with_exclusions adds new nodes to exclusion set."""
        original = RouteConstraints(
            exclude_nodes=frozenset({"X"}),
        )

        updated = original.with_exclusions(nodes={"Y", "Z"})

        assert "X" in updated.exclude_nodes
        assert "Y" in updated.exclude_nodes
        assert "Z" in updated.exclude_nodes

    def test_frozen(self) -> None:
        """RouteConstraints is immutable."""
        constraints = RouteConstraints()

        with pytest.raises(AttributeError):
            constraints.max_hops = 10  # type: ignore[misc]


# =============================================================================
# KShortestPathStrategy Tests
# =============================================================================


class TestKShortestPathStrategy:
    """Tests for KShortestPathStrategy."""

    def test_init_default_k(self) -> None:
        """KShortestPathStrategy defaults to k=3."""
        strategy = KShortestPathStrategy()
        assert strategy.k == 3
        assert strategy.name == "k_shortest_3"

    def test_init_custom_k(self) -> None:
        """KShortestPathStrategy accepts custom k."""
        strategy = KShortestPathStrategy(k=5)
        assert strategy.k == 5
        assert strategy.name == "k_shortest_5"

    def test_select_routes_returns_empty_on_no_path(self) -> None:
        """select_routes returns empty result when no path exists."""
        strategy = KShortestPathStrategy(k=3)

        mock_network_state = MagicMock()
        mock_network_state.topology = MagicMock()
        mock_network_state.config = MagicMock()

        with patch("networkx.shortest_simple_paths") as mock_paths:
            import networkx as nx
            mock_paths.side_effect = nx.NetworkXNoPath()

            result = strategy.select_routes(
                "A", "Z", 100, mock_network_state
            )

        assert result.is_empty
        assert result.strategy_name == "k_shortest_3"

    def test_select_routes_returns_paths(self) -> None:
        """select_routes returns paths when they exist."""
        strategy = KShortestPathStrategy(k=2)

        # Create mock topology
        mock_topology = MagicMock()
        mock_topology.get_edge_data.return_value = {"weight": 100.0}
        mock_topology.copy.return_value = mock_topology
        mock_topology.has_edge.return_value = False
        mock_topology.has_node.return_value = False

        mock_network_state = MagicMock()
        mock_network_state.topology = mock_topology
        mock_network_state.config = MagicMock()

        with patch("networkx.shortest_simple_paths") as mock_paths:
            mock_paths.return_value = iter([
                ["A", "B", "Z"],
                ["A", "C", "Z"],
            ])

            result = strategy.select_routes(
                "A", "Z", 100, mock_network_state
            )

        assert not result.is_empty
        assert len(result.paths) == 2
        assert result.paths[0] == ("A", "B", "Z")
        assert result.paths[1] == ("A", "C", "Z")

    def test_select_routes_applies_hop_constraint(self) -> None:
        """select_routes filters paths exceeding max_hops."""
        strategy = KShortestPathStrategy(k=3)

        mock_topology = MagicMock()
        mock_topology.get_edge_data.return_value = {"weight": 100.0}
        mock_topology.copy.return_value = mock_topology
        mock_topology.has_edge.return_value = False
        mock_topology.has_node.return_value = False

        mock_network_state = MagicMock()
        mock_network_state.topology = mock_topology
        mock_network_state.config = MagicMock()

        constraints = RouteConstraints(max_hops=2)

        with patch("networkx.shortest_simple_paths") as mock_paths:
            # First path has 2 hops, second has 3 hops
            mock_paths.return_value = iter([
                ["A", "B", "Z"],  # 2 hops - OK
                ["A", "B", "C", "Z"],  # 3 hops - filtered
            ])

            result = strategy.select_routes(
                "A", "Z", 100, mock_network_state, constraints
            )

        assert len(result.paths) == 1
        assert result.paths[0] == ("A", "B", "Z")


# =============================================================================
# LoadBalancedStrategy Tests
# =============================================================================


class TestLoadBalancedStrategy:
    """Tests for LoadBalancedStrategy."""

    def test_init_defaults(self) -> None:
        """LoadBalancedStrategy has correct defaults."""
        strategy = LoadBalancedStrategy()
        assert strategy.k == 3
        assert strategy.utilization_weight == 0.5
        assert strategy.name == "load_balanced"

    def test_init_custom_values(self) -> None:
        """LoadBalancedStrategy accepts custom values."""
        strategy = LoadBalancedStrategy(k=5, utilization_weight=0.7)
        assert strategy.k == 5
        assert strategy.utilization_weight == 0.7

    def test_utilization_weight_clamped(self) -> None:
        """utilization_weight is clamped to [0, 1]."""
        strategy_low = LoadBalancedStrategy(utilization_weight=-0.5)
        assert strategy_low.utilization_weight == 0.0

        strategy_high = LoadBalancedStrategy(utilization_weight=1.5)
        assert strategy_high.utilization_weight == 1.0

    def test_select_routes_returns_empty_when_base_empty(self) -> None:
        """select_routes returns empty when no base paths found."""
        strategy = LoadBalancedStrategy()

        mock_network_state = MagicMock()
        mock_network_state.topology = MagicMock()
        mock_network_state.config = MagicMock()
        mock_network_state.get_link_utilization.return_value = 0.0

        with patch.object(
            KShortestPathStrategy, "select_routes"
        ) as mock_ksp:
            from fusion.domain.results import RouteResult
            mock_ksp.return_value = RouteResult.empty()

            result = strategy.select_routes(
                "A", "Z", 100, mock_network_state
            )

        assert result.is_empty
        assert result.strategy_name == "load_balanced"


# =============================================================================
# ProtectionAwareStrategy Tests
# =============================================================================


class TestProtectionAwareStrategy:
    """Tests for ProtectionAwareStrategy."""

    def test_init_defaults(self) -> None:
        """ProtectionAwareStrategy has correct defaults."""
        strategy = ProtectionAwareStrategy()
        assert strategy.node_disjoint is False
        assert strategy.name == "protection_aware"

    def test_init_node_disjoint(self) -> None:
        """ProtectionAwareStrategy can be node-disjoint."""
        strategy = ProtectionAwareStrategy(node_disjoint=True)
        assert strategy.node_disjoint is True

    def test_find_disjoint_pair_returns_empty_protection_on_no_working(
        self,
    ) -> None:
        """find_disjoint_pair returns empty protection when no working path."""
        strategy = ProtectionAwareStrategy()

        mock_network_state = MagicMock()
        mock_network_state.topology = MagicMock()
        mock_network_state.config = MagicMock()

        with patch.object(strategy, "select_routes") as mock_select:
            from fusion.domain.results import RouteResult
            mock_select.return_value = RouteResult.empty()

            working, protection = strategy.find_disjoint_pair(
                "A", "Z", 100, mock_network_state
            )

        assert working.is_empty
        assert protection.is_empty

    def test_find_disjoint_pair_excludes_working_links(self) -> None:
        """find_disjoint_pair excludes working path links for protection."""
        strategy = ProtectionAwareStrategy()

        mock_network_state = MagicMock()
        mock_network_state.topology = MagicMock()
        mock_network_state.config = MagicMock()

        from fusion.domain.results import RouteResult

        working_result = RouteResult(
            paths=(("A", "B", "Z"),),
            weights_km=(200.0,),
            modulations=(("QPSK",),),
            strategy_name="protection_aware",
        )

        protection_result = RouteResult(
            paths=(("A", "C", "Z"),),
            weights_km=(250.0,),
            modulations=(("QPSK",),),
            strategy_name="protection_aware",
        )

        call_count = 0

        def mock_select(*args: object, **kwargs: object) -> RouteResult:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return working_result
            return protection_result

        with patch.object(strategy, "select_routes", side_effect=mock_select):
            working, protection = strategy.find_disjoint_pair(
                "A", "Z", 100, mock_network_state
            )

        assert not working.is_empty
        assert working.best_path == ("A", "B", "Z")
        assert not protection.is_empty
        assert protection.best_path == ("A", "C", "Z")
