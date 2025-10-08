"""Tests for traffic grooming module."""

import pytest

from fusion.core.grooming import Grooming
from fusion.core.properties import SDNProps


class TestGroomingInitialization:
    """Tests for Grooming class initialization."""

    @pytest.fixture
    def engine_props(self) -> dict:
        """Provide basic engine properties for testing."""
        return {
            "is_grooming_enabled": True,
            "can_partially_serve": True,
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
