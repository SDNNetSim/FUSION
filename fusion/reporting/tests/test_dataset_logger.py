"""
Tests for DatasetLogger.
"""

import json
from typing import Any

from fusion.modules.rl.policies import KSPFFPolicy
from fusion.reporting.dataset_logger import (
    DatasetLogger,
    filter_by_window,
    load_dataset,
    select_path_with_epsilon_mix,
)


class TestDatasetLogger:
    """Test cases for DatasetLogger."""

    def test_dataset_logger_creates_file(self, tmp_path: Any) -> None:
        """Test that JSONL file created at specified path."""
        output_path = tmp_path / "test_data.jsonl"
        engine_props = {"seed": 42}

        logger = DatasetLogger(str(output_path), engine_props)
        logger.close()

        assert output_path.exists()

    def test_transition_logged_correctly(self, tmp_path: Any) -> None:
        """Test that logged transition matches expected schema."""
        output_path = tmp_path / "test_data.jsonl"
        engine_props = {"seed": 42}

        logger = DatasetLogger(str(output_path), engine_props)

        state = {
            "src": 0,
            "dst": 5,
            "slots_needed": 2,
            "est_remaining_time": 1.5,
            "is_disaster": 0,
            "paths": [
                {
                    "path_hops": 3,
                    "min_residual_slots": 10,
                    "frag_indicator": 0.2,
                    "failure_mask": 0,
                    "dist_to_disaster_centroid": 0,
                }
            ],
        }

        logger.log_transition(
            state=state,
            action=0,
            reward=1.0,
            next_state=None,
            action_mask=[True],
            meta={"decision_time_ms": 0.5, "bp_window_tag": "pre"},
        )

        logger.close()

        # Read and verify
        with open(output_path) as f:
            transition = json.loads(f.readline())

        assert transition["t"] == 0
        assert transition["seed"] == 42
        assert transition["state"]["src"] == 0
        assert transition["action"] == 0
        assert transition["reward"] == 1.0
        assert transition["action_mask"] == [True]
        assert transition["meta"]["bp_window_tag"] == "pre"

    def test_multiple_transitions_logged(self, tmp_path: Any) -> None:
        """Test that multiple transitions are logged correctly."""
        output_path = tmp_path / "test_data.jsonl"
        engine_props = {"seed": 42}

        logger = DatasetLogger(str(output_path), engine_props)

        # Log 3 transitions
        for i in range(3):
            logger.log_transition(
                state={"src": i, "dst": i + 1, "paths": []},
                action=i,
                reward=1.0 if i % 2 == 0 else -1.0,
                next_state=None,
                action_mask=[True],
                meta={"decision_time_ms": 0.5, "bp_window_tag": "pre"},
            )

        logger.close()

        # Read all transitions
        with open(output_path) as f:
            lines = f.readlines()

        assert len(lines) == 3

        # Verify sequence numbers
        for i, line in enumerate(lines):
            transition = json.loads(line)
            assert transition["t"] == i
            assert transition["action"] == i

    def test_action_mask_logged(self, tmp_path: Any) -> None:
        """Test that action_mask array logged correctly."""
        output_path = tmp_path / "test_data.jsonl"
        logger = DatasetLogger(str(output_path), {"seed": 42})

        action_mask = [True, False, True, False]

        logger.log_transition(
            state={"src": 0, "dst": 1, "paths": []},
            action=0,
            reward=1.0,
            next_state=None,
            action_mask=action_mask,
            meta={"decision_time_ms": 0.5, "bp_window_tag": "pre"},
        )

        logger.close()

        # Verify
        with open(output_path) as f:
            transition = json.loads(f.readline())

        assert transition["action_mask"] == action_mask

    def test_meta_fields_populated(self, tmp_path: Any) -> None:
        """Test that all meta fields present in logged data."""
        output_path = tmp_path / "test_data.jsonl"
        logger = DatasetLogger(str(output_path), {"seed": 42})

        meta = {
            "decision_time_ms": 0.35,
            "restoration_latency_ms": 100,
            "bp_window_tag": "fail",
            "backup_available": True,
            "was_groomed": False,
        }

        logger.log_transition(
            state={"src": 0, "dst": 1, "paths": []},
            action=0,
            reward=1.0,
            next_state=None,
            action_mask=[True],
            meta=meta,
        )

        logger.close()

        # Verify
        with open(output_path) as f:
            transition = json.loads(f.readline())

        assert all(key in transition["meta"] for key in meta.keys())
        assert transition["meta"] == meta

    def test_context_manager(self, tmp_path: Any) -> None:
        """Test that DatasetLogger works as context manager."""
        output_path = tmp_path / "test_data.jsonl"

        with DatasetLogger(str(output_path), {"seed": 42}) as logger:
            logger.log_transition(
                state={"src": 0, "dst": 1, "paths": []},
                action=0,
                reward=1.0,
                next_state=None,
                action_mask=[True],
                meta={"decision_time_ms": 0.5, "bp_window_tag": "pre"},
            )

        # File should be closed and exist
        assert output_path.exists()

        with open(output_path) as f:
            lines = f.readlines()
            assert len(lines) == 1

    def test_directory_creation(self, tmp_path: Any) -> None:
        """Test that parent directories are created if needed."""
        output_path = tmp_path / "subdir" / "nested" / "test_data.jsonl"

        logger = DatasetLogger(str(output_path), {"seed": 42})
        logger.close()

        assert output_path.exists()
        assert output_path.parent.exists()


class TestEpsilonMix:
    """Test cases for epsilon-mix path selection."""

    def test_epsilon_mix_selects_second_best_with_epsilon_one(self) -> None:
        """Test that second-best path selected with epsilon=1.0."""
        policy = KSPFFPolicy()
        state = {"src": 0, "dst": 1, "paths": [{}, {}, {}]}
        action_mask = [True, True, True, False]

        # With epsilon=1.0, should always select second
        selected = select_path_with_epsilon_mix(policy, state, action_mask, epsilon=1.0)
        assert selected == 1

    def test_epsilon_mix_selects_first_with_epsilon_zero(self) -> None:
        """Test that first path selected with epsilon=0.0."""
        policy = KSPFFPolicy()
        state = {"src": 0, "dst": 1, "paths": [{}, {}, {}]}
        action_mask = [True, True, True, False]

        # With epsilon=0.0, should always select first (KSP-FF behavior)
        selected = select_path_with_epsilon_mix(policy, state, action_mask, epsilon=0.0)
        assert selected == 0

    def test_epsilon_mix_returns_negative_when_all_blocked(self) -> None:
        """Test that -1 returned when all paths blocked."""
        policy = KSPFFPolicy()
        state = {"src": 0, "dst": 1, "paths": []}
        action_mask: list[bool] = []

        selected = select_path_with_epsilon_mix(policy, state, action_mask, epsilon=0.1)
        assert selected == -1

    def test_epsilon_mix_handles_single_feasible_path(self) -> None:
        """Test epsilon-mix with only one feasible path."""
        policy = KSPFFPolicy()
        state = {"src": 0, "dst": 1, "paths": [{}, {}]}
        action_mask = [False, True]

        # Only one feasible path, should select it regardless of epsilon
        selected = select_path_with_epsilon_mix(policy, state, action_mask, epsilon=1.0)
        assert selected == 1


class TestLoadDataset:
    """Test cases for dataset loading utilities."""

    def test_load_dataset(self, tmp_path: Any) -> None:
        """Test that dataset loads correctly from JSONL."""
        output_path = tmp_path / "test_data.jsonl"

        # Write test data
        with DatasetLogger(str(output_path), {"seed": 42}) as logger:
            for i in range(3):
                logger.log_transition(
                    state={"src": i, "dst": i + 1, "paths": []},
                    action=i,
                    reward=1.0,
                    next_state=None,
                    action_mask=[True],
                    meta={"decision_time_ms": 0.5, "bp_window_tag": "pre"},
                )

        # Load and verify
        transitions = list(load_dataset(str(output_path)))

        assert len(transitions) == 3
        for i, trans in enumerate(transitions):
            assert trans["t"] == i
            assert trans["action"] == i

    def test_filter_by_window(self, tmp_path: Any) -> None:
        """Test filtering dataset by BP window tag."""
        output_path = tmp_path / "test_data.jsonl"

        # Write test data with different window tags
        with DatasetLogger(str(output_path), {"seed": 42}) as logger:
            for tag in ["pre", "fail", "post", "pre", "fail"]:
                logger.log_transition(
                    state={"src": 0, "dst": 1, "paths": []},
                    action=0,
                    reward=1.0,
                    next_state=None,
                    action_mask=[True],
                    meta={"decision_time_ms": 0.5, "bp_window_tag": tag},
                )

        # Filter by window
        pre_transitions = filter_by_window(str(output_path), "pre")
        fail_transitions = filter_by_window(str(output_path), "fail")
        post_transitions = filter_by_window(str(output_path), "post")

        assert len(pre_transitions) == 2
        assert len(fail_transitions) == 2
        assert len(post_transitions) == 1

        # Verify all have correct tag
        for trans in pre_transitions:
            assert trans["meta"]["bp_window_tag"] == "pre"
