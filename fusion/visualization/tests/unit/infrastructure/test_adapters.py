"""Unit tests for data adapters."""

import pytest
from typing import Dict, Any

from fusion.visualization.infrastructure.adapters import (
    V1DataAdapter,
    V2DataAdapter,
    DataAdapterRegistry,
    CanonicalData,
)
from fusion.visualization.domain.value_objects import DataVersion
from fusion.visualization.domain.exceptions import UnsupportedDataFormatError


class TestV1DataAdapter:
    """Tests for V1 data adapter."""

    def test_version(self, v1_adapter: V1DataAdapter) -> None:
        """Should return correct version."""
        assert v1_adapter.version == DataVersion.from_string("v1")

    def test_can_handle_v1_format(self, v1_adapter: V1DataAdapter, sample_v1_data: Dict[str, Any]) -> None:
        """Should recognize V1 format."""
        assert v1_adapter.can_handle(sample_v1_data)

    def test_can_handle_v2_format_returns_false(self, v1_adapter: V1DataAdapter, sample_v2_data: Dict[str, Any]) -> None:
        """Should not recognize V2 format."""
        assert not v1_adapter.can_handle(sample_v2_data)

    def test_can_handle_missing_fields_returns_false(self, v1_adapter: V1DataAdapter) -> None:
        """Should return False for data without V1 fields."""
        data = {"unknown_field": 123}
        assert not v1_adapter.can_handle(data)

    def test_to_canonical_converts_v1_data(self, v1_adapter: V1DataAdapter, sample_v1_data: Dict[str, Any]) -> None:
        """Should convert V1 data to canonical format."""
        canonical = v1_adapter.to_canonical(sample_v1_data)

        assert isinstance(canonical, CanonicalData)
        assert canonical.version == "v1"
        assert canonical.blocking_probability == 0.045
        assert len(canonical.iterations) == 2

    def test_to_canonical_preserves_iteration_data(self, v1_adapter: V1DataAdapter, sample_v1_data: Dict[str, Any]) -> None:
        """Should preserve iteration data correctly."""
        canonical = v1_adapter.to_canonical(sample_v1_data)

        iter0 = canonical.iterations[0]
        assert iter0.iteration == 0
        assert iter0.hops_mean == 3.2
        assert iter0.sim_block_list == [0.05, 0.04, 0.045, 0.043]

    def test_to_canonical_extracts_metadata(self, v1_adapter: V1DataAdapter, sample_v1_data: Dict[str, Any]) -> None:
        """Should extract metadata from V1 data."""
        canonical = v1_adapter.to_canonical(sample_v1_data)

        assert "sim_end_time" in canonical.metadata
        assert canonical.sim_start_time == "0429_21_14_39_491949"

    def test_to_canonical_wrong_format_raises_error(self, v1_adapter: V1DataAdapter, sample_v2_data: Dict[str, Any]) -> None:
        """Should raise UnsupportedDataFormatError for wrong format."""
        with pytest.raises(UnsupportedDataFormatError):
            v1_adapter.to_canonical(sample_v2_data)

    def test_validate_data_valid(self, v1_adapter: V1DataAdapter, sample_v1_data: Dict[str, Any]) -> None:
        """Should validate valid V1 data."""
        assert v1_adapter.validate_data(sample_v1_data)

    def test_validate_data_missing_fields(self, v1_adapter: V1DataAdapter) -> None:
        """Should return False for missing required fields."""
        data = {"other_field": 123}
        assert not v1_adapter.validate_data(data)


class TestV2DataAdapter:
    """Tests for V2 data adapter."""

    def test_version(self, v2_adapter: V2DataAdapter) -> None:
        """Should return correct version."""
        assert v2_adapter.version == DataVersion.from_string("v2")

    def test_can_handle_v2_format(self, v2_adapter: V2DataAdapter, sample_v2_data: Dict[str, Any]) -> None:
        """Should recognize V2 format."""
        assert v2_adapter.can_handle(sample_v2_data)

    def test_can_handle_v1_format_returns_false(self, v2_adapter: V2DataAdapter, sample_v1_data: Dict[str, Any]) -> None:
        """Should not recognize V1 format."""
        assert not v2_adapter.can_handle(sample_v1_data)

    def test_can_handle_explicit_version(self, v2_adapter: V2DataAdapter) -> None:
        """Should recognize explicit version field."""
        data = {"version": "v2", "metrics": {}}
        assert v2_adapter.can_handle(data)

    def test_to_canonical_converts_v2_data(self, v2_adapter: V2DataAdapter, sample_v2_data: Dict[str, Any]) -> None:
        """Should convert V2 data to canonical format."""
        canonical = v2_adapter.to_canonical(sample_v2_data)

        assert isinstance(canonical, CanonicalData)
        assert canonical.version == "v2"
        assert canonical.blocking_probability == 0.045
        assert len(canonical.iterations) == 2

    def test_to_canonical_preserves_iteration_data(self, v2_adapter: V2DataAdapter, sample_v2_data: Dict[str, Any]) -> None:
        """Should preserve iteration data correctly."""
        canonical = v2_adapter.to_canonical(sample_v2_data)

        iter0 = canonical.iterations[0]
        assert iter0.iteration == 0
        assert iter0.hops_mean == 3.2
        assert iter0.sim_block_list == [0.05, 0.04, 0.045, 0.043]

    def test_to_canonical_extracts_timing_info(self, v2_adapter: V2DataAdapter, sample_v2_data: Dict[str, Any]) -> None:
        """Should extract timing information from V2 data."""
        canonical = v2_adapter.to_canonical(sample_v2_data)

        assert canonical.sim_start_time == "2024-04-29T21:14:39.491949"
        assert canonical.duration_seconds == 122.631507

    def test_to_canonical_extracts_metadata(self, v2_adapter: V2DataAdapter, sample_v2_data: Dict[str, Any]) -> None:
        """Should extract metadata from V2 data."""
        canonical = v2_adapter.to_canonical(sample_v2_data)

        assert canonical.metadata["network"] == "NSFNet"
        assert canonical.metadata["algorithm"] == "ppo_obs_7"

    def test_validate_data_valid(self, v2_adapter: V2DataAdapter, sample_v2_data: Dict[str, Any]) -> None:
        """Should validate valid V2 data."""
        assert v2_adapter.validate_data(sample_v2_data)

    def test_validate_data_missing_fields(self, v2_adapter: V2DataAdapter) -> None:
        """Should return False for missing required fields."""
        data = {"other_field": 123}
        assert not v2_adapter.validate_data(data)


class TestDataAdapterRegistry:
    """Tests for DataAdapterRegistry."""

    def test_default_adapters_registered(self, adapter_registry: DataAdapterRegistry) -> None:
        """Should have V1 and V2 adapters registered by default."""
        versions = adapter_registry.get_supported_versions()
        assert DataVersion.from_string("v1") in versions
        assert DataVersion.from_string("v2") in versions

    def test_get_adapter_for_v1_data(self, adapter_registry: DataAdapterRegistry, sample_v1_data: Dict[str, Any]) -> None:
        """Should return V1 adapter for V1 data."""
        adapter = adapter_registry.get_adapter(sample_v1_data)
        assert isinstance(adapter, V1DataAdapter)

    def test_get_adapter_for_v2_data(self, adapter_registry: DataAdapterRegistry, sample_v2_data: Dict[str, Any]) -> None:
        """Should return V2 adapter for V2 data."""
        adapter = adapter_registry.get_adapter(sample_v2_data)
        assert isinstance(adapter, V2DataAdapter)

    def test_get_adapter_unsupported_format_raises_error(self, adapter_registry: DataAdapterRegistry) -> None:
        """Should raise UnsupportedDataFormatError for unknown format."""
        data = {"unknown_field": 123}
        with pytest.raises(UnsupportedDataFormatError):
            adapter_registry.get_adapter(data)

    def test_get_adapter_by_version(self, adapter_registry: DataAdapterRegistry) -> None:
        """Should get adapter by specific version."""
        v1 = adapter_registry.get_adapter_by_version(DataVersion.from_string("v1"))
        assert isinstance(v1, V1DataAdapter)

        v2 = adapter_registry.get_adapter_by_version(DataVersion.from_string("v2"))
        assert isinstance(v2, V2DataAdapter)

    def test_get_adapter_by_nonexistent_version(self, adapter_registry: DataAdapterRegistry) -> None:
        """Should return None for nonexistent version."""
        v99 = adapter_registry.get_adapter_by_version(DataVersion.from_string("v99"))
        assert v99 is None

    def test_register_custom_adapter(self, adapter_registry: DataAdapterRegistry) -> None:
        """Should allow registering custom adapters."""
        # Create mock adapter
        class V3DataAdapter(V1DataAdapter):
            @property
            def version(self) -> DataVersion:
                return DataVersion.from_string("v3")

        v3_adapter = V3DataAdapter()
        adapter_registry.register_adapter(v3_adapter)

        # Should be in supported versions
        versions = adapter_registry.get_supported_versions()
        assert DataVersion.from_string("v3") in versions

    def test_clear_registry(self, adapter_registry: DataAdapterRegistry) -> None:
        """Should clear all adapters."""
        adapter_registry.clear()
        versions = adapter_registry.get_supported_versions()
        assert len(versions) == 0


class TestCanonicalData:
    """Tests for CanonicalData."""

    def test_get_final_blocking_probability_from_field(self, v1_adapter: V1DataAdapter, sample_v1_data: Dict[str, Any]) -> None:
        """Should get blocking probability from direct field."""
        canonical = v1_adapter.to_canonical(sample_v1_data)
        bp = canonical.get_final_blocking_probability()
        assert bp == 0.045

    def test_get_final_blocking_probability_from_iterations(self, v1_adapter: V1DataAdapter) -> None:
        """Should calculate from iterations if no direct field."""
        data = {
            "iter_stats": {
                "0": {"sim_block_list": [0.05, 0.04]},
            }
        }
        canonical = v1_adapter.to_canonical(data)
        canonical.blocking_probability = None  # Remove direct field

        bp = canonical.get_final_blocking_probability()
        assert bp is not None

    def test_get_last_k_iterations(self, v1_adapter: V1DataAdapter, sample_v1_data: Dict[str, Any]) -> None:
        """Should return last K iterations."""
        canonical = v1_adapter.to_canonical(sample_v1_data)

        last_1 = canonical.get_last_k_iterations(1)
        assert len(last_1) == 1

        last_5 = canonical.get_last_k_iterations(5)
        assert len(last_5) == 2  # Only 2 iterations exist

    def test_get_metric_simple_path(self, v1_adapter: V1DataAdapter, sample_v1_data: Dict[str, Any]) -> None:
        """Should get metric with simple path."""
        canonical = v1_adapter.to_canonical(sample_v1_data)

        blocking = canonical.get_metric("blocking")
        assert blocking == 0.045

    def test_get_metric_nested_path(self, v1_adapter: V1DataAdapter, sample_v1_data: Dict[str, Any]) -> None:
        """Should get metric with nested path."""
        canonical = v1_adapter.to_canonical(sample_v1_data)

        iter0 = canonical.get_metric("iterations.0")
        assert iter0.iteration == 0

        hops = canonical.get_metric("iterations.0.hops_mean")
        assert hops == 3.2
