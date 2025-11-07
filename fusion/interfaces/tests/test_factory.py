"""
Unit tests for fusion.interfaces.factory module.

Tests the AlgorithmFactory and SimulationPipeline classes.
"""

from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from fusion.interfaces.factory import (
    AlgorithmFactory,
    SimulationPipeline,
    create_simulation_pipeline,
)
from fusion.interfaces.router import AbstractRoutingAlgorithm
from fusion.interfaces.snr import AbstractSNRMeasurer
from fusion.interfaces.spectrum import AbstractSpectrumAssigner

# ============================================================================
# Test AlgorithmFactory - Routing
# ============================================================================


class TestAlgorithmFactoryRouting:
    """Tests for AlgorithmFactory routing algorithm creation."""

    @patch("fusion.modules.routing.registry.create_algorithm")
    def test_create_routing_algorithm_with_valid_name(self, mock_create: Mock) -> None:
        """Test creating routing algorithm with valid name."""
        # Arrange
        mock_algo = Mock(spec=AbstractRoutingAlgorithm)
        mock_create.return_value = mock_algo
        engine_props = {"topology": None}
        sdn_props = Mock()

        # Act
        result = AlgorithmFactory.create_routing_algorithm(
            "k_shortest_path", engine_props, sdn_props
        )

        # Assert
        assert result == mock_algo
        mock_create.assert_called_once_with("k_shortest_path", engine_props, sdn_props)

    @patch("fusion.modules.routing.registry.create_algorithm")
    def test_create_routing_algorithm_with_invalid_name_raises_error(
        self, mock_create: Mock
    ) -> None:
        """Test creating routing algorithm with invalid name raises ValueError."""
        # Arrange
        mock_create.side_effect = KeyError("Unknown algorithm")
        engine_props = {"topology": None}
        sdn_props = Mock()

        # Act & Assert
        with pytest.raises(ValueError, match="Unknown routing algorithm"):
            AlgorithmFactory.create_routing_algorithm(
                "invalid_algo", engine_props, sdn_props
            )


# ============================================================================
# Test AlgorithmFactory - Spectrum
# ============================================================================


class TestAlgorithmFactorySpectrum:
    """Tests for AlgorithmFactory spectrum algorithm creation."""

    def test_create_spectrum_algorithm_with_valid_name(self) -> None:
        """Test creating spectrum algorithm with valid name."""
        # Arrange
        mock_algo = Mock(spec=AbstractSpectrumAssigner)
        engine_props = {"topology": None}
        sdn_props = Mock()
        route_props = Mock()

        with patch(
            "fusion.interfaces.factory.create_spectrum_algorithm",
            return_value=mock_algo,
        ) as mock_create:
            # Act
            result = AlgorithmFactory.create_spectrum_algorithm(
                "first_fit", engine_props, sdn_props, route_props
            )

            # Assert
            assert result == mock_algo
            mock_create.assert_called_once_with(
                "first_fit", engine_props, sdn_props, route_props
            )

    def test_create_spectrum_algorithm_with_invalid_name_raises_error(self) -> None:
        """Test creating spectrum algorithm with invalid name raises ValueError."""
        # Arrange
        engine_props = {"topology": None}
        sdn_props = Mock()
        route_props = Mock()

        with patch(
            "fusion.interfaces.factory.create_spectrum_algorithm",
            side_effect=KeyError("Unknown algorithm"),
        ):
            # Act & Assert
            with pytest.raises(ValueError, match="Unknown spectrum algorithm"):
                AlgorithmFactory.create_spectrum_algorithm(
                    "invalid_algo", engine_props, sdn_props, route_props
                )


# ============================================================================
# Test AlgorithmFactory - SNR
# ============================================================================


class TestAlgorithmFactorySNR:
    """Tests for AlgorithmFactory SNR algorithm creation."""

    def test_create_snr_algorithm_with_valid_name(self) -> None:
        """Test creating SNR algorithm with valid name."""
        # Arrange
        mock_algo = Mock(spec=AbstractSNRMeasurer)
        engine_props = {"topology": None}
        sdn_props = Mock()
        spectrum_props = Mock()
        route_props = Mock()

        with patch(
            "fusion.interfaces.factory.create_snr_algorithm", return_value=mock_algo
        ) as mock_create:
            # Act
            result = AlgorithmFactory.create_snr_algorithm(
                "standard_snr",
                engine_props,
                sdn_props,
                spectrum_props,
                route_props,
            )

            # Assert
            assert result == mock_algo
            mock_create.assert_called_once_with(
                "standard_snr",
                engine_props,
                sdn_props,
                spectrum_props,
                route_props,
            )

    def test_create_snr_algorithm_with_invalid_name_raises_error(self) -> None:
        """Test creating SNR algorithm with invalid name raises ValueError."""
        # Arrange
        engine_props = {"topology": None}
        sdn_props = Mock()
        spectrum_props = Mock()
        route_props = Mock()

        with patch(
            "fusion.interfaces.factory.create_snr_algorithm",
            side_effect=KeyError("Unknown algorithm"),
        ):
            # Act & Assert
            with pytest.raises(ValueError, match="Unknown SNR algorithm"):
                AlgorithmFactory.create_snr_algorithm(
                    "invalid_algo",
                    engine_props,
                    sdn_props,
                    spectrum_props,
                    route_props,
                )


# ============================================================================
# Test SimulationPipeline - Initialization
# ============================================================================


class TestSimulationPipelineInitialization:
    """Tests for SimulationPipeline initialization."""

    @patch("fusion.modules.routing.registry.create_algorithm")
    @patch("fusion.modules.spectrum.registry.create_spectrum_algorithm")
    @patch("fusion.modules.snr.registry.create_snr_algorithm")
    def test_pipeline_initialization_with_valid_config(
        self, mock_snr: Mock, mock_spectrum: Mock, mock_routing: Mock
    ) -> None:
        """Test pipeline initialization with valid configuration."""
        # Arrange
        mock_routing.return_value = Mock(spec=AbstractRoutingAlgorithm)
        mock_spectrum.return_value = Mock(spec=AbstractSpectrumAssigner)
        mock_snr.return_value = Mock(spec=AbstractSNRMeasurer)

        config = {
            "engine_props": {},
            "sdn_props": Mock(),
            "route_props": Mock(),
            "spectrum_props": Mock(),
            "routing_algorithm": "k_shortest_path",
            "spectrum_algorithm": "first_fit",
            "snr_algorithm": "standard_snr",
        }

        # Act
        pipeline = SimulationPipeline(config)

        # Assert
        assert pipeline.routing_algorithm is not None
        assert pipeline.spectrum_algorithm is not None
        assert pipeline.snr_algorithm is not None

    @patch("fusion.modules.routing.registry.create_algorithm")
    def test_pipeline_initialization_with_missing_algorithm_raises_error(
        self, mock_routing: Mock
    ) -> None:
        """Test pipeline initialization with missing algorithm raises error."""
        # Arrange
        mock_routing.side_effect = ValueError("Algorithm not found")

        config = {
            "engine_props": {},
            "sdn_props": Mock(),
            "route_props": Mock(),
            "spectrum_props": Mock(),
        }

        # Act & Assert
        with pytest.raises(RuntimeError, match="Failed to initialize"):
            SimulationPipeline(config)

    @patch("fusion.modules.routing.registry.create_algorithm")
    @patch("fusion.interfaces.factory.create_spectrum_algorithm")
    @patch("fusion.interfaces.factory.create_snr_algorithm")
    def test_pipeline_uses_default_algorithm_names(
        self, mock_snr: Mock, mock_spectrum: Mock, mock_routing: Mock
    ) -> None:
        """Test pipeline uses default algorithm names when not specified."""
        # Arrange
        mock_routing.return_value = Mock(spec=AbstractRoutingAlgorithm)
        mock_spectrum.return_value = Mock(spec=AbstractSpectrumAssigner)
        mock_snr.return_value = Mock(spec=AbstractSNRMeasurer)

        config = {
            "engine_props": {},
            "sdn_props": Mock(),
            "route_props": Mock(),
            "spectrum_props": Mock(),
        }

        # Act
        SimulationPipeline(config)

        # Assert
        mock_routing.assert_called_once()
        assert "k_shortest_path" in str(mock_routing.call_args)
        mock_spectrum.assert_called_once()
        assert "first_fit" in str(mock_spectrum.call_args)
        mock_snr.assert_called_once()
        assert "standard_snr" in str(mock_snr.call_args)


# ============================================================================
# Test SimulationPipeline - Process Request Success
# ============================================================================


class TestSimulationPipelineProcessRequestSuccess:
    """Tests for successful request processing."""

    def _create_mock_pipeline(self) -> SimulationPipeline:
        """Create a mock pipeline with mocked algorithms."""
        with (
            patch("fusion.modules.routing.registry.create_algorithm"),
            patch("fusion.modules.spectrum.registry.create_spectrum_algorithm"),
            patch("fusion.modules.snr.registry.create_snr_algorithm"),
        ):
            config: dict[str, Any] = {
                "engine_props": {"topology": MagicMock()},
                "sdn_props": Mock(),
                "route_props": Mock(),
                "spectrum_props": Mock(),
            }
            pipeline = SimulationPipeline(config)

            # Replace with mocks
            pipeline.routing_algorithm = Mock(spec=AbstractRoutingAlgorithm)
            pipeline.spectrum_algorithm = Mock(spec=AbstractSpectrumAssigner)
            pipeline.snr_algorithm = Mock(spec=AbstractSNRMeasurer)

            # Setup route_props for routing algorithm
            pipeline.routing_algorithm.route_props = Mock()
            pipeline.routing_algorithm.route_props.paths_matrix = []

            # Setup get_metrics
            pipeline.routing_algorithm.get_metrics.return_value = {}
            pipeline.spectrum_algorithm.get_metrics.return_value = {}
            pipeline.snr_algorithm.get_metrics.return_value = {}

            return pipeline

    def test_process_request_with_successful_allocation(self) -> None:
        """Test process_request with successful spectrum allocation."""
        # Arrange
        pipeline = self._create_mock_pipeline()
        # Setup paths_matrix to contain the path found by routing
        pipeline.routing_algorithm.route_props.paths_matrix = [[1, 2, 3]]
        pipeline.spectrum_algorithm.assign.return_value = {  # type: ignore[attr-defined]
            "start_slot": 0,
            "end_slot": 5,
            "core_num": 0,
            "band": "C",
        }
        pipeline.snr_algorithm.calculate_snr.return_value = 20.0  # type: ignore[attr-defined]
        pipeline.snr_algorithm.get_required_snr_threshold.return_value = 15.0  # type: ignore[attr-defined]
        pipeline.snr_algorithm.is_snr_acceptable.return_value = True  # type: ignore[attr-defined]
        pipeline.spectrum_algorithm.allocate_spectrum.return_value = True  # type: ignore[attr-defined]

        # Mock topology for path length calculation
        mock_topology = MagicMock()
        mock_topology.__getitem__.return_value.__getitem__.return_value = {
            "length": 100
        }
        pipeline.engine_props["topology"] = mock_topology

        request = Mock()
        request.modulation = "QPSK"

        # Act
        result = pipeline.process_request(1, 3, request)

        # Assert
        assert result["success"] is True
        assert result["path"] == [1, 2, 3]
        assert result["spectrum_assignment"] is not None
        assert result["snr"] == 20.0


# ============================================================================
# Test SimulationPipeline - Process Request Failures
# ============================================================================


class TestSimulationPipelineProcessRequestFailures:
    """Tests for request processing failures."""

    def _create_mock_pipeline(self) -> SimulationPipeline:
        """Create a mock pipeline with mocked algorithms."""
        with (
            patch("fusion.modules.routing.registry.create_algorithm"),
            patch("fusion.modules.spectrum.registry.create_spectrum_algorithm"),
            patch("fusion.modules.snr.registry.create_snr_algorithm"),
        ):
            config: dict[str, Any] = {
                "engine_props": {"topology": MagicMock()},
                "sdn_props": Mock(),
                "route_props": Mock(),
                "spectrum_props": Mock(),
            }
            pipeline = SimulationPipeline(config)
            pipeline.routing_algorithm = Mock(spec=AbstractRoutingAlgorithm)
            pipeline.spectrum_algorithm = Mock(spec=AbstractSpectrumAssigner)
            pipeline.snr_algorithm = Mock(spec=AbstractSNRMeasurer)
            # Setup route_props for routing algorithm
            pipeline.routing_algorithm.route_props = Mock()
            pipeline.routing_algorithm.route_props.paths_matrix = []
            pipeline.routing_algorithm.get_metrics.return_value = {}
            pipeline.spectrum_algorithm.get_metrics.return_value = {}
            pipeline.snr_algorithm.get_metrics.return_value = {}
            return pipeline

    def test_process_request_when_no_path_found(self) -> None:
        """Test process_request when no path is found."""
        # Arrange
        pipeline = self._create_mock_pipeline()
        # Empty paths_matrix indicates no path was found
        pipeline.routing_algorithm.route_props.paths_matrix = []
        request = Mock()

        # Act
        result = pipeline.process_request(1, 3, request)

        # Assert
        assert result["success"] is False
        assert result["failure_reason"] == "No path found"
        assert result["path"] is None

    def test_process_request_when_no_spectrum_available(self) -> None:
        """Test process_request when no spectrum is available."""
        # Arrange
        pipeline = self._create_mock_pipeline()
        pipeline.routing_algorithm.route_props.paths_matrix = [[1, 2, 3]]
        pipeline.spectrum_algorithm.assign.return_value = None  # type: ignore[attr-defined]
        request = Mock()

        # Act
        result = pipeline.process_request(1, 3, request)

        # Assert
        assert result["success"] is False
        assert result["failure_reason"] == "No spectrum available"

    def test_process_request_when_snr_too_low(self) -> None:
        """Test process_request when SNR is below threshold."""
        # Arrange
        pipeline = self._create_mock_pipeline()
        pipeline.routing_algorithm.route_props.paths_matrix = [[1, 2, 3]]
        pipeline.spectrum_algorithm.assign.return_value = {  # type: ignore[attr-defined]
            "start_slot": 0,
            "end_slot": 5,
            "core_num": 0,
            "band": "C",
        }
        pipeline.snr_algorithm.calculate_snr.return_value = 10.0  # type: ignore[attr-defined]
        pipeline.snr_algorithm.get_required_snr_threshold.return_value = 15.0  # type: ignore[attr-defined]
        pipeline.snr_algorithm.is_snr_acceptable.return_value = False  # type: ignore[attr-defined]

        # Mock topology
        mock_topology = MagicMock()
        mock_topology.__getitem__.return_value.__getitem__.return_value = {
            "length": 100
        }
        pipeline.engine_props["topology"] = mock_topology

        request = Mock()
        request.modulation = "QPSK"

        # Act
        result = pipeline.process_request(1, 3, request)

        # Assert
        assert result["success"] is False
        assert "SNR too low" in result["failure_reason"]
        assert result["snr"] == 10.0

    def test_process_request_when_allocation_fails(self) -> None:
        """Test process_request when spectrum allocation fails."""
        # Arrange
        pipeline = self._create_mock_pipeline()
        pipeline.routing_algorithm.route_props.paths_matrix = [[1, 2, 3]]
        pipeline.spectrum_algorithm.assign.return_value = {  # type: ignore[attr-defined]
            "start_slot": 0,
            "end_slot": 5,
            "core_num": 0,
            "band": "C",
        }
        pipeline.snr_algorithm.calculate_snr.return_value = 20.0  # type: ignore[attr-defined]
        pipeline.snr_algorithm.get_required_snr_threshold.return_value = 15.0  # type: ignore[attr-defined]
        pipeline.snr_algorithm.is_snr_acceptable.return_value = True  # type: ignore[attr-defined]
        pipeline.spectrum_algorithm.allocate_spectrum.return_value = False  # type: ignore[attr-defined]

        # Mock topology
        mock_topology = MagicMock()
        mock_topology.__getitem__.return_value.__getitem__.return_value = {
            "length": 100
        }
        pipeline.engine_props["topology"] = mock_topology

        request = Mock()
        request.modulation = "QPSK"

        # Act
        result = pipeline.process_request(1, 3, request)

        # Assert
        assert result["success"] is False
        assert result["failure_reason"] == "Spectrum allocation failed"


# ============================================================================
# Test SimulationPipeline - Metrics
# ============================================================================


class TestSimulationPipelineMetrics:
    """Tests for SimulationPipeline metrics collection."""

    def _create_mock_pipeline(self) -> SimulationPipeline:
        """Create a mock pipeline with mocked algorithms."""
        with (
            patch("fusion.modules.routing.registry.create_algorithm"),
            patch("fusion.modules.spectrum.registry.create_spectrum_algorithm"),
            patch("fusion.modules.snr.registry.create_snr_algorithm"),
        ):
            config: dict[str, Any] = {
                "engine_props": {},
                "sdn_props": Mock(),
                "route_props": Mock(),
                "spectrum_props": Mock(),
            }
            pipeline = SimulationPipeline(config)
            pipeline.routing_algorithm = Mock(spec=AbstractRoutingAlgorithm)
            pipeline.spectrum_algorithm = Mock(spec=AbstractSpectrumAssigner)
            pipeline.snr_algorithm = Mock(spec=AbstractSNRMeasurer)
            return pipeline

    def test_process_request_includes_metrics(self) -> None:
        """Test that process_request includes metrics from all algorithms."""
        # Arrange
        pipeline = self._create_mock_pipeline()
        pipeline.routing_algorithm.route.return_value = None  # type: ignore[attr-defined]
        pipeline.routing_algorithm.get_metrics.return_value = {"routing_metric": 1}  # type: ignore[attr-defined]
        pipeline.spectrum_algorithm.get_metrics.return_value = {"spectrum_metric": 2}  # type: ignore[attr-defined]
        pipeline.snr_algorithm.get_metrics.return_value = {"snr_metric": 3}  # type: ignore[attr-defined]

        request = Mock()

        # Act
        result = pipeline.process_request(1, 3, request)

        # Assert
        assert "metrics" in result
        assert result["metrics"]["routing"] == {"routing_metric": 1}
        assert result["metrics"]["spectrum"] == {"spectrum_metric": 2}
        assert result["metrics"]["snr"] == {"snr_metric": 3}


# ============================================================================
# Test SimulationPipeline - Utility Methods
# ============================================================================


class TestSimulationPipelineUtilityMethods:
    """Tests for SimulationPipeline utility methods."""

    def _create_mock_pipeline(self) -> SimulationPipeline:
        """Create a mock pipeline with mocked algorithms."""
        with (
            patch("fusion.modules.routing.registry.create_algorithm"),
            patch("fusion.modules.spectrum.registry.create_spectrum_algorithm"),
            patch("fusion.modules.snr.registry.create_snr_algorithm"),
        ):
            config: dict[str, Any] = {
                "engine_props": {},
                "sdn_props": Mock(),
                "route_props": Mock(),
                "spectrum_props": Mock(),
            }
            pipeline = SimulationPipeline(config)
            pipeline.routing_algorithm = Mock(spec=AbstractRoutingAlgorithm)
            pipeline.spectrum_algorithm = Mock(spec=AbstractSpectrumAssigner)
            pipeline.snr_algorithm = Mock(spec=AbstractSNRMeasurer)
            pipeline.routing_algorithm.algorithm_name = "test_routing"
            pipeline.routing_algorithm.supported_topologies = ["NSFNet"]
            pipeline.spectrum_algorithm.algorithm_name = "test_spectrum"
            pipeline.spectrum_algorithm.supports_multiband = True
            pipeline.snr_algorithm.algorithm_name = "test_snr"
            pipeline.snr_algorithm.supports_multicore = False
            return pipeline

    def test_get_algorithm_info_returns_all_algorithms(self) -> None:
        """Test get_algorithm_info returns information for all algorithms."""
        # Arrange
        pipeline = self._create_mock_pipeline()

        # Act
        info = pipeline.get_algorithm_info()

        # Assert
        assert "routing" in info
        assert "spectrum" in info
        assert "snr" in info
        assert info["routing"]["name"] == "test_routing"
        assert info["spectrum"]["name"] == "test_spectrum"
        assert info["snr"]["name"] == "test_snr"

    def test_reset_all_algorithms_calls_reset_on_all(self) -> None:
        """Test reset_all_algorithms calls reset on all algorithms."""
        # Arrange
        pipeline = self._create_mock_pipeline()

        # Act
        pipeline.reset_all_algorithms()

        # Assert
        pipeline.routing_algorithm.reset.assert_called_once()  # type: ignore[attr-defined]
        pipeline.spectrum_algorithm.reset.assert_called_once()  # type: ignore[attr-defined]
        pipeline.snr_algorithm.reset.assert_called_once()  # type: ignore[attr-defined]


# ============================================================================
# Test create_simulation_pipeline Function
# ============================================================================


class TestCreateSimulationPipelineFunction:
    """Tests for create_simulation_pipeline factory function."""

    @patch("fusion.modules.routing.registry.create_algorithm")
    @patch("fusion.modules.spectrum.registry.create_spectrum_algorithm")
    @patch("fusion.modules.snr.registry.create_snr_algorithm")
    def test_create_simulation_pipeline_returns_pipeline_instance(
        self, mock_snr: Mock, mock_spectrum: Mock, mock_routing: Mock
    ) -> None:
        """Test create_simulation_pipeline returns SimulationPipeline instance."""
        # Arrange
        mock_routing.return_value = Mock(spec=AbstractRoutingAlgorithm)
        mock_spectrum.return_value = Mock(spec=AbstractSpectrumAssigner)
        mock_snr.return_value = Mock(spec=AbstractSNRMeasurer)

        config = {
            "engine_props": {},
            "sdn_props": Mock(),
            "route_props": Mock(),
            "spectrum_props": Mock(),
        }

        # Act
        pipeline = create_simulation_pipeline(config)

        # Assert
        assert isinstance(pipeline, SimulationPipeline)


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestSimulationPipelineEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_pipeline_handles_exception_during_processing(self) -> None:
        """Test pipeline handles exceptions during request processing."""
        # Arrange
        with (
            patch("fusion.modules.routing.registry.create_algorithm"),
            patch("fusion.modules.spectrum.registry.create_spectrum_algorithm"),
            patch("fusion.modules.snr.registry.create_snr_algorithm"),
        ):
            config: dict[str, Any] = {
                "engine_props": {},
                "sdn_props": Mock(),
                "route_props": Mock(),
                "spectrum_props": Mock(),
            }
            pipeline = SimulationPipeline(config)
            pipeline.routing_algorithm = Mock(spec=AbstractRoutingAlgorithm)
            pipeline.routing_algorithm.route.side_effect = ValueError("Test error")
            pipeline.routing_algorithm.get_metrics.return_value = {}
            pipeline.spectrum_algorithm = Mock(spec=AbstractSpectrumAssigner)
            pipeline.spectrum_algorithm.get_metrics.return_value = {}
            pipeline.snr_algorithm = Mock(spec=AbstractSNRMeasurer)
            pipeline.snr_algorithm.get_metrics.return_value = {}

            request = Mock()

            # Act
            result = pipeline.process_request(1, 3, request)

            # Assert
            assert result["success"] is False
            assert "failure_reason" in result
            assert "Test error" in result["failure_reason"]

    def test_pipeline_handles_missing_topology_in_engine_props(self) -> None:
        """Test pipeline handles missing topology in engine_props."""
        # Arrange
        with (
            patch("fusion.modules.routing.registry.create_algorithm"),
            patch("fusion.modules.spectrum.registry.create_spectrum_algorithm"),
            patch("fusion.modules.snr.registry.create_snr_algorithm"),
        ):
            config: dict[str, Any] = {
                "engine_props": {},
                "sdn_props": Mock(),
                "route_props": Mock(),
                "spectrum_props": Mock(),
            }
            pipeline = SimulationPipeline(config)
            pipeline.routing_algorithm = Mock(spec=AbstractRoutingAlgorithm)
            pipeline.routing_algorithm.route_props = Mock()
            pipeline.routing_algorithm.route_props.paths_matrix = [[1, 2]]
            pipeline.routing_algorithm.get_metrics.return_value = {}
            pipeline.spectrum_algorithm = Mock(spec=AbstractSpectrumAssigner)
            pipeline.spectrum_algorithm.assign.return_value = {
                "start_slot": 0,
                "end_slot": 5,
                "core_num": 0,
                "band": "C",
            }
            pipeline.spectrum_algorithm.get_metrics.return_value = {}
            pipeline.snr_algorithm = Mock(spec=AbstractSNRMeasurer)
            pipeline.snr_algorithm.calculate_snr.return_value = 20.0
            pipeline.snr_algorithm.get_metrics.return_value = {}
            pipeline.sdn_props = None

            request = Mock()
            request.modulation = "QPSK"

            # Act
            result = pipeline.process_request(1, 2, request)

            # Assert
            assert result["success"] is False
            assert "No topology available" in result["failure_reason"]
