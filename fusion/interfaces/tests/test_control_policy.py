"""Tests for ControlPolicy protocol."""

from typing import Any

from fusion.interfaces.control_policy import ControlPolicy, PolicyAction


class TestControlPolicyProtocol:
    """Tests for ControlPolicy protocol using @runtime_checkable."""

    def test_protocol_isinstance_check_valid(self) -> None:
        """Protocol should support isinstance() checks for valid implementations."""

        class ValidPolicy:
            """A valid policy implementation."""

            def select_action(
                self, request: Any, options: Any, network_state: Any
            ) -> int:
                return 0

            def update(self, request: Any, action: int, reward: float) -> None:
                pass

            def get_name(self) -> str:
                return "ValidPolicy"

        policy = ValidPolicy()
        assert isinstance(policy, ControlPolicy)

    def test_missing_select_action_fails_isinstance(self) -> None:
        """Class missing select_action should fail isinstance()."""

        class InvalidPolicy:
            """Missing select_action method."""

            def update(self, request: Any, action: int, reward: float) -> None:
                pass

            def get_name(self) -> str:
                return "InvalidPolicy"

        policy = InvalidPolicy()
        assert not isinstance(policy, ControlPolicy)

    def test_missing_update_fails_isinstance(self) -> None:
        """Class missing update should fail isinstance()."""

        class InvalidPolicy:
            """Missing update method."""

            def select_action(
                self, request: Any, options: Any, network_state: Any
            ) -> int:
                return 0

            def get_name(self) -> str:
                return "InvalidPolicy"

        policy = InvalidPolicy()
        assert not isinstance(policy, ControlPolicy)

    def test_missing_get_name_fails_isinstance(self) -> None:
        """Class missing get_name should fail isinstance()."""

        class InvalidPolicy:
            """Missing get_name method."""

            def select_action(
                self, request: Any, options: Any, network_state: Any
            ) -> int:
                return 0

            def update(self, request: Any, action: int, reward: float) -> None:
                pass

        policy = InvalidPolicy()
        assert not isinstance(policy, ControlPolicy)

    def test_minimal_policy_implementation(self) -> None:
        """Minimal compliant implementation should work."""

        class MinimalPolicy:
            """Minimal implementation that satisfies ControlPolicy protocol."""

            def select_action(
                self, request: Any, options: Any, network_state: Any
            ) -> int:
                # Select first feasible option
                for opt in options:
                    if getattr(opt, "is_feasible", False):
                        return getattr(opt, "path_index", 0)
                return -1

            def update(self, request: Any, action: int, reward: float) -> None:
                pass

            def get_name(self) -> str:
                return "MinimalPolicy"

        policy = MinimalPolicy()
        assert isinstance(policy, ControlPolicy)
        assert policy.get_name() == "MinimalPolicy"

    def test_policy_with_state(self) -> None:
        """Policy with internal state should also pass isinstance()."""

        class StatefulPolicy:
            """Policy with internal state."""

            def __init__(self, alpha: float = 0.5) -> None:
                self.alpha = alpha
                self.call_count = 0

            def select_action(
                self, request: Any, options: Any, network_state: Any
            ) -> int:
                self.call_count += 1
                return 0 if options else -1

            def update(self, request: Any, action: int, reward: float) -> None:
                # Could update internal state here
                pass

            def get_name(self) -> str:
                return f"StatefulPolicy(alpha={self.alpha})"

        policy = StatefulPolicy(alpha=0.7)
        assert isinstance(policy, ControlPolicy)
        assert policy.get_name() == "StatefulPolicy(alpha=0.7)"


class TestPolicyActionTypeAlias:
    """Tests for PolicyAction type alias."""

    def test_policy_action_is_int(self) -> None:
        """PolicyAction should be an int alias."""
        # PolicyAction should accept int values
        action: PolicyAction = 0
        assert isinstance(action, int)

        action = -1
        assert isinstance(action, int)

        action = 4
        assert isinstance(action, int)
