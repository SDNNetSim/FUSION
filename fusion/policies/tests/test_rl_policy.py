"""Tests for RLPolicy wrapper."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from fusion.interfaces.control_policy import ControlPolicy
from fusion.modules.rl.adapter import PathOption
from fusion.policies.rl_policy import RLPolicy

if TYPE_CHECKING:
    from fusion.domain.network_state import NetworkState
    from fusion.domain.request import Request


@dataclass
class MockRequest:
    """Mock Request for testing."""

    request_id: int = 1
    source: str = "0"
    destination: str = "5"
    bandwidth_gbps: int = 100
    arrive_time: float = 0.0
    depart_time: float = 1.0


@dataclass
class MockNetworkState:
    """Mock NetworkState for testing."""

    pass


def mock_request() -> "Request":
    """Create a mock request cast to Request type for testing."""
    return cast("Request", MockRequest())


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


class TestRLPolicyProtocolCompliance:
    """Tests for RLPolicy implementing ControlPolicy protocol."""

    def test_rl_policy_implements_control_policy(self) -> None:
        """RLPolicy should satisfy ControlPolicy protocol."""
        mock_model = MagicMock()
        mock_model.predict.return_value = (1, None)

        policy = RLPolicy(mock_model)

        # Should pass isinstance check via @runtime_checkable
        assert isinstance(policy, ControlPolicy)

    def test_rl_policy_has_required_methods(self) -> None:
        """RLPolicy should have all required methods."""
        mock_model = MagicMock()
        mock_model.predict.return_value = (0, None)

        policy = RLPolicy(mock_model)

        assert hasattr(policy, "select_action")
        assert hasattr(policy, "update")
        assert hasattr(policy, "get_name")
        assert callable(policy.select_action)
        assert callable(policy.update)
        assert callable(policy.get_name)


class TestRLPolicyInitialization:
    """Tests for RLPolicy initialization."""

    def test_init_with_model(self) -> None:
        """RLPolicy can be initialized with a model."""
        mock_model = MagicMock()
        mock_model.predict.return_value = (0, None)

        policy = RLPolicy(mock_model)

        assert policy.model is mock_model
        assert policy.k_paths == 5  # Default

    def test_init_with_k_paths(self) -> None:
        """RLPolicy can be initialized with custom k_paths."""
        mock_model = MagicMock()
        mock_model.predict.return_value = (0, None)

        policy = RLPolicy(mock_model, k_paths=10)

        assert policy.k_paths == 10

    def test_init_rejects_model_without_predict(self) -> None:
        """RLPolicy should reject models without predict method."""
        mock_model = MagicMock(spec=[])  # No methods

        with pytest.raises(ValueError, match="does not have predict"):
            RLPolicy(mock_model)


class TestRLPolicySelectAction:
    """Tests for RLPolicy.select_action method."""

    def test_select_action_calls_model_predict(self) -> None:
        """select_action should call model.predict."""
        mock_model = MagicMock()
        mock_model.predict.return_value = (1, None)

        policy = RLPolicy(mock_model)
        options = [
            create_path_option(path_index=0, is_feasible=False),
            create_path_option(path_index=1, is_feasible=True),
            create_path_option(path_index=2, is_feasible=True),
        ]

        action = policy.select_action(mock_request(), options, mock_network_state())

        assert mock_model.predict.called
        assert action == 1

    def test_select_action_returns_model_prediction(self) -> None:
        """select_action should return the model's prediction."""
        mock_model = MagicMock()
        mock_model.predict.return_value = (2, None)

        policy = RLPolicy(mock_model)
        options = [
            create_path_option(path_index=0, is_feasible=True),
            create_path_option(path_index=1, is_feasible=True),
            create_path_option(path_index=2, is_feasible=True),
        ]

        action = policy.select_action(mock_request(), options, mock_network_state())

        assert action == 2

    def test_infeasible_prediction_falls_back(self) -> None:
        """Should select first feasible when model returns infeasible."""
        mock_model = MagicMock()
        mock_model.predict.return_value = (0, None)  # Index 0 is infeasible

        policy = RLPolicy(mock_model)
        options = [
            create_path_option(path_index=0, is_feasible=False),
            create_path_option(path_index=1, is_feasible=True),
            create_path_option(path_index=2, is_feasible=True),
        ]

        action = policy.select_action(mock_request(), options, mock_network_state())

        assert action == 1  # First feasible

    def test_all_infeasible_returns_negative_one(self) -> None:
        """Should return -1 when no feasible options."""
        mock_model = MagicMock()
        mock_model.predict.return_value = (0, None)

        policy = RLPolicy(mock_model)
        infeasible_options = [
            create_path_option(path_index=i, is_feasible=False) for i in range(3)
        ]

        action = policy.select_action(mock_request(), infeasible_options, mock_network_state())

        assert action == -1

    def test_empty_options_returns_negative_one(self) -> None:
        """Should return -1 when options list is empty."""
        mock_model = MagicMock()
        mock_model.predict.return_value = (0, None)

        policy = RLPolicy(mock_model)

        action = policy.select_action(mock_request(), [], mock_network_state())

        assert action == -1

    def test_model_exception_returns_negative_one(self) -> None:
        """Should return -1 when model raises exception."""
        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("Model error")

        policy = RLPolicy(mock_model)
        options = [create_path_option(path_index=0, is_feasible=True)]

        action = policy.select_action(mock_request(), options, mock_network_state())

        assert action == -1


class TestRLPolicyActionMasking:
    """Tests for RLPolicy action masking behavior."""

    def test_action_mask_built_from_feasibility(self) -> None:
        """Action mask should reflect feasibility flags."""
        mock_model = MagicMock()
        mock_model.predict.return_value = (1, None)

        policy = RLPolicy(mock_model, k_paths=3)
        options = [
            create_path_option(path_index=0, is_feasible=True),
            create_path_option(path_index=1, is_feasible=False),
            create_path_option(path_index=2, is_feasible=True),
        ]

        mask = policy._build_action_mask(options)

        assert mask == [True, False, True]

    def test_action_mask_padded_to_k_paths(self) -> None:
        """Action mask should be padded to k_paths."""
        mock_model = MagicMock()
        mock_model.predict.return_value = (0, None)

        policy = RLPolicy(mock_model, k_paths=5)
        options = [
            create_path_option(path_index=0, is_feasible=True),
            create_path_option(path_index=1, is_feasible=True),
        ]

        mask = policy._build_action_mask(options)

        assert mask == [True, True, False, False, False]

    def test_maskable_ppo_uses_action_masks(self) -> None:
        """MaskablePPO should use action_masks parameter."""
        mock_model = MagicMock()
        mock_model.__class__.__name__ = "MaskablePPO"
        mock_model.predict.return_value = (1, None)

        policy = RLPolicy(mock_model, k_paths=3)
        options = [
            create_path_option(path_index=0, is_feasible=True),
            create_path_option(path_index=1, is_feasible=True),
            create_path_option(path_index=2, is_feasible=False),
        ]

        policy.select_action(mock_request(), options, mock_network_state())

        # Check that action_masks was passed
        call_kwargs = mock_model.predict.call_args[1]
        assert "action_masks" in call_kwargs
        np.testing.assert_array_equal(
            call_kwargs["action_masks"], np.array([True, True, False])
        )


class TestRLPolicyUpdate:
    """Tests for RLPolicy.update method."""

    def test_update_is_noop(self) -> None:
        """update() should be a no-op (pre-trained model)."""
        mock_model = MagicMock()
        mock_model.predict.return_value = (0, None)

        policy = RLPolicy(mock_model)

        # Should not raise
        policy.update(mock_request(), 1, 1.0)

        # Model should not be modified
        assert not mock_model.learn.called


class TestRLPolicyGetName:
    """Tests for RLPolicy.get_name method."""

    def test_get_name_includes_model_type(self) -> None:
        """get_name should include model class name."""
        mock_model = MagicMock()
        mock_model.__class__.__name__ = "PPO"
        mock_model.predict.return_value = (0, None)

        policy = RLPolicy(mock_model)

        name = policy.get_name()

        assert "RLPolicy" in name
        assert "PPO" in name

    def test_get_name_format(self) -> None:
        """get_name should have expected format."""
        mock_model = MagicMock()
        mock_model.__class__.__name__ = "DQN"
        mock_model.predict.return_value = (0, None)

        policy = RLPolicy(mock_model)

        name = policy.get_name()

        assert name == "RLPolicy(DQN)"


class TestRLPolicyFromFile:
    """Tests for RLPolicy.from_file classmethod."""

    def test_from_file_loads_ppo(self) -> None:
        """from_file should load PPO model."""
        mock_model = MagicMock()
        mock_model.predict.return_value = (0, None)

        with patch("stable_baselines3.PPO") as mock_ppo_class:
            mock_ppo_class.load.return_value = mock_model

            policy = RLPolicy.from_file("model.zip", algorithm="PPO")

            mock_ppo_class.load.assert_called_once_with("model.zip")
            assert policy.model is mock_model

    def test_from_file_loads_maskable_ppo(self) -> None:
        """from_file should load MaskablePPO from sb3_contrib."""
        mock_model = MagicMock()
        mock_model.predict.return_value = (0, None)

        # Mock sb3 not having MaskablePPO
        mock_sb3 = MagicMock()
        mock_sb3.MaskablePPO = None
        del mock_sb3.MaskablePPO  # Make getattr return None

        mock_sb3_contrib = MagicMock()
        mock_sb3_contrib.MaskablePPO.load.return_value = mock_model

        with patch.dict(
            "sys.modules",
            {
                "stable_baselines3": mock_sb3,
                "sb3_contrib": mock_sb3_contrib,
            },
        ):
            # Need to re-import to use patched modules

            with patch("importlib.import_module") as mock_import:

                def import_side_effect(name: str) -> Any:
                    if name == "stable_baselines3":
                        return mock_sb3
                    elif name == "sb3_contrib":
                        return mock_sb3_contrib
                    raise ImportError(f"No module {name}")

                mock_import.side_effect = import_side_effect

                policy = RLPolicy.from_file("model.zip", algorithm="MaskablePPO")

                mock_sb3_contrib.MaskablePPO.load.assert_called_once_with("model.zip")
                assert policy.model is mock_model

    def test_from_file_unknown_algorithm_raises(self) -> None:
        """from_file should raise for unknown algorithm."""
        with patch("importlib.import_module") as mock_import:
            mock_sb3 = MagicMock(spec=[])  # No algorithms
            mock_import.side_effect = [mock_sb3, ImportError("No sb3_contrib")]

            with pytest.raises(ValueError, match="Unknown algorithm"):
                RLPolicy.from_file("model.zip", algorithm="UnknownAlgo")

    def test_from_file_passes_kwargs(self) -> None:
        """from_file should pass kwargs to __init__."""
        mock_model = MagicMock()
        mock_model.predict.return_value = (0, None)

        with patch("stable_baselines3.PPO") as mock_ppo_class:
            mock_ppo_class.load.return_value = mock_model

            policy = RLPolicy.from_file("model.zip", algorithm="PPO", k_paths=10)

            assert policy.k_paths == 10


class TestRLPolicyObservationBuilding:
    """Tests for RLPolicy observation building."""

    def test_builds_observation_array(self) -> None:
        """Should build numpy observation array."""
        mock_model = MagicMock()
        mock_model.predict.return_value = (0, None)

        policy = RLPolicy(mock_model, k_paths=3)
        options = [
            create_path_option(path_index=0, is_feasible=True),
            create_path_option(path_index=1, is_feasible=False),
        ]

        obs = policy._build_observation(mock_request(), options, mock_network_state())

        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32

    def test_uses_adapter_if_provided(self) -> None:
        """Should use adapter for observation building if provided."""
        mock_model = MagicMock()
        mock_model.predict.return_value = (0, None)

        mock_adapter = MagicMock()
        expected_obs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        mock_adapter.build_observation.return_value = expected_obs

        policy = RLPolicy(mock_model, adapter=mock_adapter)
        options = [create_path_option(path_index=0, is_feasible=True)]

        obs = policy._build_observation(mock_request(), options, mock_network_state())

        mock_adapter.build_observation.assert_called_once()
        np.testing.assert_array_equal(obs, expected_obs)

    def test_set_adapter(self) -> None:
        """Should be able to set adapter after initialization."""
        mock_model = MagicMock()
        mock_model.predict.return_value = (0, None)

        policy = RLPolicy(mock_model)
        assert policy._adapter is None

        mock_adapter = MagicMock()
        policy.set_adapter(mock_adapter)

        assert policy._adapter is mock_adapter
