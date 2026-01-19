"""Tests for MLControlPolicy."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from fusion.interfaces.control_policy import ControlPolicy
from fusion.modules.rl.adapter import PathOption
from fusion.policies.heuristic_policy import (
    FirstFeasiblePolicy,
    LeastCongestedPolicy,
    ShortestFeasiblePolicy,
)
from fusion.policies.ml_policy import (
    CallableModelWrapper,
    FeatureBuilder,
    MLControlPolicy,
    SklearnModelWrapper,
)

if TYPE_CHECKING:
    from fusion.domain.network_state import NetworkState
    from fusion.domain.request import Request


@dataclass
class MockRequest:
    """Mock Request for testing."""

    request_id: int = 1
    source: str = "0"
    destination: str = "5"
    bandwidth_gbps: float = 100.0
    arrive_time: float = 0.0
    depart_time: float = 1.0


@dataclass
class MockNetworkState:
    """Mock NetworkState for testing."""

    pass


def mock_request(**kwargs: Any) -> "Request":
    """Create a mock request cast to Request type for testing."""
    return cast("Request", MockRequest(**kwargs))


def mock_network_state() -> "NetworkState":
    """Create a mock network state cast to NetworkState type for testing."""
    return cast("NetworkState", MockNetworkState())


def create_path_option(
    path_index: int = 0,
    is_feasible: bool = True,
    weight_km: float = 100.0,
    congestion: float = 0.3,
    slots_needed: int = 4,
) -> PathOption:
    """Helper to create a PathOption for testing."""
    return PathOption(
        path_index=path_index,
        path=("0", "1", "2"),
        weight_km=weight_km,
        num_hops=2,
        modulation="QPSK",
        slots_needed=slots_needed,
        is_feasible=is_feasible,
        congestion=congestion,
        available_slots=0.7,
    )


class MockSklearnModel:
    """Mock sklearn model with predict_proba."""

    def __init__(self, proba_output: np.ndarray) -> None:
        """
        Initialize mock sklearn model.

        Args:
            proba_output: Fixed output to return from predict_proba
        """
        self._proba_output = proba_output

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return fixed probability distribution."""
        # Return 2D array as sklearn does
        if features.ndim == 1:
            return self._proba_output.reshape(1, -1)
        return np.tile(self._proba_output, (len(features), 1))


class MockTorchLikeCallable:
    """Mock torch-like callable model returning logits."""

    def __init__(self, logits_output: np.ndarray) -> None:
        """
        Initialize mock torch-like model.

        Args:
            logits_output: Fixed logits to return
        """
        self._logits_output = logits_output

    def __call__(self, features: np.ndarray) -> np.ndarray:
        """Return fixed logits."""
        return self._logits_output


class MockFailingModel:
    """Mock model that raises on predict."""

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Always raise an error."""
        raise RuntimeError("Model inference failed")


class TestFeatureBuilder:
    """Tests for FeatureBuilder."""

    def test_feature_size_k5(self) -> None:
        """Feature size should be 1 + 4*k_paths for k=5."""
        builder = FeatureBuilder(k_paths=5)
        assert builder.feature_size == 21

    def test_feature_size_k3(self) -> None:
        """Feature size should be 1 + 4*k_paths for k=3."""
        builder = FeatureBuilder(k_paths=3)
        assert builder.feature_size == 13

    def test_build_with_all_paths(self) -> None:
        """Should build correct features with full path list."""
        builder = FeatureBuilder(k_paths=3)
        request = mock_request(bandwidth_gbps=100.0)
        options = [
            create_path_option(path_index=0, is_feasible=True, weight_km=500.0, congestion=0.3, slots_needed=4),
            create_path_option(path_index=1, is_feasible=False, weight_km=800.0, congestion=0.7, slots_needed=8),
            create_path_option(path_index=2, is_feasible=True, weight_km=600.0, congestion=0.5, slots_needed=6),
        ]

        features = builder.build(request, options, mock_network_state())

        assert features.shape == (13,)
        # Request feature: 100/1000 = 0.1
        assert features[0] == pytest.approx(0.1)
        # Path 0: weight=500/10000=0.05, congestion=0.3, feasible=1, slots=4/100=0.04
        assert features[1] == pytest.approx(0.05)
        assert features[2] == pytest.approx(0.3)
        assert features[3] == pytest.approx(1.0)
        assert features[4] == pytest.approx(0.04)

    def test_build_with_padding(self) -> None:
        """Should pad missing paths correctly."""
        builder = FeatureBuilder(k_paths=3)
        request = mock_request(bandwidth_gbps=100.0)
        options = [
            create_path_option(path_index=0, is_feasible=True, weight_km=500.0, congestion=0.3),
        ]

        features = builder.build(request, options, mock_network_state())

        assert features.shape == (13,)
        # Path 1 should be padded: weight=0, congestion=1, feasible=0, slots=0
        assert features[5] == pytest.approx(0.0)  # path_1_weight
        assert features[6] == pytest.approx(1.0)  # path_1_congestion (padding)
        assert features[7] == pytest.approx(0.0)  # path_1_feasible
        assert features[8] == pytest.approx(0.0)  # path_1_slots

    def test_build_empty_options(self) -> None:
        """Should handle empty options with all padding."""
        builder = FeatureBuilder(k_paths=2)
        request = mock_request(bandwidth_gbps=50.0)

        features = builder.build(request, [], mock_network_state())

        assert features.shape == (9,)  # 1 + 2*4
        assert features[0] == pytest.approx(0.05)  # bandwidth
        # All path features should be padding
        assert features[3] == pytest.approx(0.0)  # path_0_feasible

    def test_dtype_is_float32(self) -> None:
        """Features should be float32."""
        builder = FeatureBuilder(k_paths=3)
        features = builder.build(mock_request(), [], mock_network_state())
        assert features.dtype == np.float32

    def test_none_request_handled(self) -> None:
        """Should handle None request gracefully."""
        builder = FeatureBuilder(k_paths=2)
        # Intentionally pass None to test error handling
        features = builder.build(None, [], None)  # type: ignore[arg-type]
        assert features.shape == (9,)
        assert features[0] == 0.0  # bandwidth defaults to 0


class TestMLControlPolicyProtocolCompliance:
    """Tests for MLControlPolicy implementing ControlPolicy protocol."""

    def test_implements_control_policy(self) -> None:
        """MLControlPolicy should satisfy ControlPolicy protocol."""
        # Create with callable model
        model = CallableModelWrapper(lambda x: np.array([0.1, 0.9, 0.0]))
        policy = MLControlPolicy(model=model, k_paths=3)

        assert isinstance(policy, ControlPolicy)

    def test_has_required_methods(self) -> None:
        """MLControlPolicy should have all required methods."""
        model = CallableModelWrapper(lambda x: np.array([0.5, 0.5]))
        policy = MLControlPolicy(model=model, k_paths=2)

        assert hasattr(policy, "select_action")
        assert hasattr(policy, "update")
        assert hasattr(policy, "get_name")
        assert callable(policy.select_action)
        assert callable(policy.update)
        assert callable(policy.get_name)


class TestMLControlPolicyInitialization:
    """Tests for MLControlPolicy initialization."""

    def test_init_with_model(self) -> None:
        """Should initialize with provided model wrapper."""
        model = CallableModelWrapper(lambda x: np.array([0.5, 0.5]))
        policy = MLControlPolicy(model=model, k_paths=2)

        assert policy._model is model

    def test_init_requires_model_or_path(self) -> None:
        """Should raise if neither model nor model_path provided."""
        with pytest.raises(ValueError, match="Either model_path or model"):
            MLControlPolicy()

    def test_init_rejects_both_model_and_path(self) -> None:
        """Should raise if both model and model_path provided."""
        model = CallableModelWrapper(lambda x: np.array([0.5]))
        with pytest.raises(ValueError, match="Cannot provide both"):
            MLControlPolicy(model=model, model_path="some_path.pt")

    def test_fallback_type_first_feasible(self) -> None:
        """Should create FirstFeasiblePolicy for 'first_feasible'."""
        model = CallableModelWrapper(lambda x: np.array([0.5]))
        policy = MLControlPolicy(model=model, fallback_type="first_feasible")

        assert isinstance(policy.fallback, FirstFeasiblePolicy)

    def test_fallback_type_shortest_feasible(self) -> None:
        """Should create ShortestFeasiblePolicy for 'shortest_feasible'."""
        model = CallableModelWrapper(lambda x: np.array([0.5]))
        policy = MLControlPolicy(model=model, fallback_type="shortest_feasible")

        assert isinstance(policy.fallback, ShortestFeasiblePolicy)

    def test_fallback_type_least_congested(self) -> None:
        """Should create LeastCongestedPolicy for 'least_congested'."""
        model = CallableModelWrapper(lambda x: np.array([0.5]))
        policy = MLControlPolicy(model=model, fallback_type="least_congested")

        assert isinstance(policy.fallback, LeastCongestedPolicy)

    def test_custom_fallback_policy(self) -> None:
        """Should use custom fallback policy if provided."""
        model = CallableModelWrapper(lambda x: np.array([0.5]))
        custom_fallback = ShortestFeasiblePolicy()
        policy = MLControlPolicy(model=model, fallback_policy=custom_fallback)

        assert policy.fallback is custom_fallback

    def test_invalid_fallback_type(self) -> None:
        """Should raise for unknown fallback type."""
        model = CallableModelWrapper(lambda x: np.array([0.5]))
        with pytest.raises(ValueError, match="Unknown fallback type"):
            MLControlPolicy(model=model, fallback_type="invalid")


class TestMLControlPolicySelectAction:
    """Tests for MLControlPolicy.select_action method."""

    def test_selects_highest_score_action(self) -> None:
        """Should select action with highest model score."""
        # Model outputs scores favoring index 2
        model = CallableModelWrapper(lambda x: np.array([0.1, 0.3, 0.9]))
        policy = MLControlPolicy(model=model, k_paths=3)

        options = [
            create_path_option(path_index=0, is_feasible=True),
            create_path_option(path_index=1, is_feasible=True),
            create_path_option(path_index=2, is_feasible=True),
        ]

        action = policy.select_action(mock_request(), options, mock_network_state())

        assert action == 2

    def test_masks_infeasible_actions(self) -> None:
        """Should not select infeasible actions even with high score."""
        # Model favors index 0, but it's infeasible
        model = CallableModelWrapper(lambda x: np.array([0.9, 0.3, 0.1]))
        policy = MLControlPolicy(model=model, k_paths=3)

        options = [
            create_path_option(path_index=0, is_feasible=False),  # High score but infeasible
            create_path_option(path_index=1, is_feasible=True),
            create_path_option(path_index=2, is_feasible=True),
        ]

        action = policy.select_action(mock_request(), options, mock_network_state())

        # Should select index 1 (next highest feasible)
        assert action == 1

    def test_empty_options_returns_negative_one(self) -> None:
        """Should return -1 for empty options."""
        model = CallableModelWrapper(lambda x: np.array([0.5]))
        policy = MLControlPolicy(model=model, k_paths=1)

        action = policy.select_action(mock_request(), [], mock_network_state())

        assert action == -1

    def test_all_infeasible_returns_negative_one(self) -> None:
        """Should return -1 when all options infeasible."""
        model = CallableModelWrapper(lambda x: np.array([0.5, 0.5]))
        policy = MLControlPolicy(model=model, k_paths=2)

        options = [
            create_path_option(path_index=0, is_feasible=False),
            create_path_option(path_index=1, is_feasible=False),
        ]

        action = policy.select_action(mock_request(), options, mock_network_state())

        assert action == -1

    def test_deterministic_behavior(self) -> None:
        """Same features should produce same action."""
        model = CallableModelWrapper(lambda x: np.array([0.2, 0.8, 0.3]))
        policy = MLControlPolicy(model=model, k_paths=3)

        options = [
            create_path_option(path_index=0, is_feasible=True, weight_km=100.0, congestion=0.3),
            create_path_option(path_index=1, is_feasible=True, weight_km=200.0, congestion=0.5),
            create_path_option(path_index=2, is_feasible=True, weight_km=150.0, congestion=0.4),
        ]
        request = mock_request(bandwidth_gbps=100.0)
        network_state = mock_network_state()

        # Run multiple times - should always get same result
        actions = [policy.select_action(request, options, network_state) for _ in range(10)]

        assert all(a == actions[0] for a in actions)


class TestMLControlPolicySklearnModel:
    """Tests for MLControlPolicy with sklearn-like models."""

    def test_sklearn_predict_proba(self) -> None:
        """Should work with sklearn predict_proba interface."""
        # sklearn model that favors index 1
        proba = np.array([0.1, 0.7, 0.2])
        mock_sklearn = MockSklearnModel(proba)
        model = SklearnModelWrapper(mock_sklearn)
        policy = MLControlPolicy(model=model, k_paths=3)

        options = [
            create_path_option(path_index=0, is_feasible=True),
            create_path_option(path_index=1, is_feasible=True),
            create_path_option(path_index=2, is_feasible=True),
        ]

        action = policy.select_action(mock_request(), options, mock_network_state())

        assert action == 1

    def test_sklearn_with_masking(self) -> None:
        """sklearn model should respect feasibility masking."""
        # sklearn model favors index 0, but it's infeasible
        proba = np.array([0.8, 0.15, 0.05])
        mock_sklearn = MockSklearnModel(proba)
        model = SklearnModelWrapper(mock_sklearn)
        policy = MLControlPolicy(model=model, k_paths=3)

        options = [
            create_path_option(path_index=0, is_feasible=False),
            create_path_option(path_index=1, is_feasible=True),
            create_path_option(path_index=2, is_feasible=True),
        ]

        action = policy.select_action(mock_request(), options, mock_network_state())

        # Should select index 1 (highest feasible)
        assert action == 1


class TestMLControlPolicyTorchLikeModel:
    """Tests for MLControlPolicy with torch-like callable models."""

    def test_torch_like_logits(self) -> None:
        """Should work with torch-like logits output."""
        # Logits favoring index 2
        logits = np.array([1.0, 2.0, 5.0])
        mock_torch = MockTorchLikeCallable(logits)
        model = CallableModelWrapper(mock_torch)
        policy = MLControlPolicy(model=model, k_paths=3)

        options = [
            create_path_option(path_index=0, is_feasible=True),
            create_path_option(path_index=1, is_feasible=True),
            create_path_option(path_index=2, is_feasible=True),
        ]

        action = policy.select_action(mock_request(), options, mock_network_state())

        assert action == 2


class TestMLControlPolicyFallback:
    """Tests for MLControlPolicy fallback mechanism."""

    def test_fallback_on_model_error(self) -> None:
        """Should use fallback when model raises exception."""

        class FailingWrapper:
            def predict(self, features: np.ndarray) -> np.ndarray:
                raise RuntimeError("Inference failed")

        policy = MLControlPolicy(model=FailingWrapper(), k_paths=3)

        options = [
            create_path_option(path_index=0, is_feasible=False),
            create_path_option(path_index=1, is_feasible=True),
            create_path_option(path_index=2, is_feasible=True),
        ]

        action = policy.select_action(mock_request(), options, mock_network_state())

        # Should get valid action from fallback
        assert action >= 0
        assert options[action].is_feasible

    def test_fallback_on_infeasible_model_selection(self) -> None:
        """Should use fallback when model selects infeasible action."""
        # Model always outputs highest score for index 0
        model = CallableModelWrapper(lambda x: np.array([0.9, 0.05, 0.05]))
        policy = MLControlPolicy(model=model, k_paths=3)

        options = [
            create_path_option(path_index=0, is_feasible=False),  # Model picks this
            create_path_option(path_index=1, is_feasible=True),  # Fallback picks this
            create_path_option(path_index=2, is_feasible=True),
        ]

        action = policy.select_action(mock_request(), options, mock_network_state())

        # Model selects 0, but it's infeasible, so masking selects 1
        assert action == 1

    def test_fallback_statistics_tracking(self) -> None:
        """Should track fallback usage statistics."""

        class FailingWrapper:
            def predict(self, features: np.ndarray) -> np.ndarray:
                raise RuntimeError("Inference failed")

        policy = MLControlPolicy(model=FailingWrapper(), k_paths=2)

        options = [
            create_path_option(path_index=0, is_feasible=True),
            create_path_option(path_index=1, is_feasible=True),
        ]

        # Make several calls that trigger fallback
        for _ in range(5):
            policy.select_action(mock_request(), options, mock_network_state())

        stats = policy.get_stats()
        assert stats["total_calls"] == 5
        assert stats["fallback_calls"] == 5
        assert stats["fallback_rate"] == 1.0
        assert "runtime_error" in stats["error_types"]

    def test_reset_stats(self) -> None:
        """Should be able to reset statistics."""

        class FailingWrapper:
            def predict(self, features: np.ndarray) -> np.ndarray:
                raise RuntimeError("Inference failed")

        policy = MLControlPolicy(model=FailingWrapper(), k_paths=2)

        options = [create_path_option(path_index=0, is_feasible=True)]
        policy.select_action(mock_request(), options, mock_network_state())

        assert policy.get_stats()["total_calls"] == 1

        policy.reset_stats()

        stats = policy.get_stats()
        assert stats["total_calls"] == 0
        assert stats["fallback_calls"] == 0
        assert stats["error_types"] == {}


class TestMLControlPolicyUpdate:
    """Tests for MLControlPolicy.update method."""

    def test_update_is_noop(self) -> None:
        """update() should be a no-op (deployment only)."""
        model = CallableModelWrapper(lambda x: np.array([0.5]))
        policy = MLControlPolicy(model=model, k_paths=1)

        # Should not raise
        policy.update(mock_request(), 0, 1.0)
        policy.update(mock_request(), 1, -1.0)


class TestMLControlPolicyGetName:
    """Tests for MLControlPolicy.get_name method."""

    def test_get_name_includes_model_info(self) -> None:
        """get_name should include model path or marker."""
        model = CallableModelWrapper(lambda x: np.array([0.5]))
        policy = MLControlPolicy(model=model, k_paths=1)

        name = policy.get_name()

        assert "MLControlPolicy" in name


class TestMLControlPolicyDifferentFromHeuristic:
    """Tests proving ML policy uses the model, not just fallback."""

    def test_ml_chooses_differently_than_first_feasible(self) -> None:
        """ML policy should choose differently than FirstFeasible for same options."""
        # Model strongly prefers index 2
        model = CallableModelWrapper(lambda x: np.array([0.1, 0.2, 0.9]))
        ml_policy = MLControlPolicy(model=model, k_paths=3)
        heuristic_policy = FirstFeasiblePolicy()

        options = [
            create_path_option(path_index=0, is_feasible=True, weight_km=100.0),
            create_path_option(path_index=1, is_feasible=True, weight_km=200.0),
            create_path_option(path_index=2, is_feasible=True, weight_km=300.0),
        ]

        ml_action = ml_policy.select_action(mock_request(), options, mock_network_state())
        heuristic_action = heuristic_policy.select_action(mock_request(), options, mock_network_state())

        # FirstFeasible selects 0, ML selects 2
        assert heuristic_action == 0
        assert ml_action == 2
        assert ml_action != heuristic_action

    def test_ml_chooses_differently_than_shortest_feasible(self) -> None:
        """ML policy can choose non-shortest path."""
        # Model prefers index 2 (longest path)
        model = CallableModelWrapper(lambda x: np.array([0.1, 0.2, 0.9]))
        ml_policy = MLControlPolicy(model=model, k_paths=3)
        heuristic_policy = ShortestFeasiblePolicy()

        options = [
            create_path_option(path_index=0, is_feasible=True, weight_km=100.0),  # Shortest
            create_path_option(path_index=1, is_feasible=True, weight_km=200.0),
            create_path_option(path_index=2, is_feasible=True, weight_km=300.0),  # Longest, ML pick
        ]

        ml_action = ml_policy.select_action(mock_request(), options, mock_network_state())
        heuristic_action = heuristic_policy.select_action(mock_request(), options, mock_network_state())

        # ShortestFeasible selects 0, ML selects 2
        assert heuristic_action == 0
        assert ml_action == 2


class TestPolicyImports:
    """Tests for policy imports from package."""

    def test_import_from_policies_package(self) -> None:
        """Should be able to import MLControlPolicy from fusion.policies."""
        from fusion.policies import FeatureBuilder, MLControlPolicy

        assert MLControlPolicy is not None
        assert FeatureBuilder is not None
