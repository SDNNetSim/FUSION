"""
Factory classes for creating algorithm instances using interfaces.
"""

from typing import Dict, Any
from fusion.interfaces.router import AbstractRoutingAlgorithm
from fusion.interfaces.spectrum import AbstractSpectrumAssigner
from fusion.interfaces.snr import AbstractSNRMeasurer

# Import registries
from fusion.modules.routing.registry import create_algorithm as create_routing_algorithm
from fusion.modules.spectrum.registry import create_spectrum_algorithm
from fusion.modules.snr.registry import create_snr_algorithm


class AlgorithmFactory:
    """Factory for creating algorithm instances using the interface system."""

    @staticmethod
    def create_routing_algorithm(name: str, engine_props: dict,
                                 sdn_props: object) -> AbstractRoutingAlgorithm:
        """Create a routing algorithm instance.
        
        Args:
            name: Name of the routing algorithm
            engine_props: Engine configuration properties
            sdn_props: SDN controller properties
            
        Returns:
            Configured routing algorithm instance
            
        Raises:
            ValueError: If algorithm name is not found
        """
        try:
            return create_routing_algorithm(name, engine_props, sdn_props)
        except KeyError as e:
            available = ["k_shortest_path", "congestion_aware", "fragmentation_aware",
                         "nli_aware", "xt_aware"]
            raise ValueError(f"Unknown routing algorithm '{name}'. Available: {available}") from e

    @staticmethod
    def create_spectrum_algorithm(name: str, engine_props: dict, sdn_props: object,
                                  route_props: object) -> AbstractSpectrumAssigner:
        """Create a spectrum assignment algorithm instance.
        
        Args:
            name: Name of the spectrum algorithm
            engine_props: Engine configuration properties
            sdn_props: SDN controller properties
            route_props: Routing properties
            
        Returns:
            Configured spectrum assignment algorithm instance
            
        Raises:
            ValueError: If algorithm name is not found
        """
        try:
            return create_spectrum_algorithm(name, engine_props, sdn_props, route_props)
        except KeyError as e:
            available = ["first_fit", "best_fit", "last_fit"]
            raise ValueError(f"Unknown spectrum algorithm '{name}'. Available: {available}") from e

    @staticmethod
    def create_snr_algorithm(name: str, engine_props: dict, sdn_props: object,
                             spectrum_props: object, route_props: object) -> AbstractSNRMeasurer:
        """Create an SNR measurement algorithm instance.
        
        Args:
            name: Name of the SNR algorithm
            engine_props: Engine configuration properties
            sdn_props: SDN controller properties
            spectrum_props: Spectrum assignment properties
            route_props: Routing properties
            
        Returns:
            Configured SNR measurement algorithm instance
            
        Raises:
            ValueError: If algorithm name is not found
        """
        try:
            return create_snr_algorithm(name, engine_props, sdn_props, spectrum_props, route_props)
        except KeyError as e:
            available = ["standard_snr"]
            raise ValueError(f"Unknown SNR algorithm '{name}'. Available: {available}") from e


class SimulationPipeline:
    """Complete simulation pipeline using interface-based algorithms."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the simulation pipeline.
        
        Args:
            config: Configuration dictionary containing algorithm selections and parameters
        """
        self.config = config
        self.engine_props = config.get('engine_props', {})
        self.sdn_props = config.get('sdn_props')
        self.route_props = config.get('route_props')
        self.spectrum_props = config.get('spectrum_props')

        # Create algorithm instances with error handling
        try:
            self.routing_algorithm = self._create_routing_algorithm()
            self.spectrum_algorithm = self._create_spectrum_algorithm()
            self.snr_algorithm = self._create_snr_algorithm()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize simulation pipeline: {e}") from e

    def _create_routing_algorithm(self) -> AbstractRoutingAlgorithm:
        """Create routing algorithm from configuration."""
        routing_name = self.config.get('routing_algorithm', 'k_shortest_path')
        return AlgorithmFactory.create_routing_algorithm(
            routing_name, self.engine_props, self.sdn_props
        )

    def _create_spectrum_algorithm(self) -> AbstractSpectrumAssigner:
        """Create spectrum assignment algorithm from configuration."""
        spectrum_name = self.config.get('spectrum_algorithm', 'first_fit')
        return AlgorithmFactory.create_spectrum_algorithm(
            spectrum_name, self.engine_props, self.sdn_props, self.route_props
        )

    def _create_snr_algorithm(self) -> AbstractSNRMeasurer:
        """Create SNR measurement algorithm from configuration."""
        snr_name = self.config.get('snr_algorithm', 'standard_snr')
        return AlgorithmFactory.create_snr_algorithm(
            snr_name, self.engine_props, self.sdn_props,
            self.spectrum_props, self.route_props
        )

    def _create_base_result(self, source: Any, destination: Any) -> Dict[str, Any]:
        """Create base result dictionary."""
        return {
            'source': source,
            'destination': destination,
            'success': False,
            'path': None,
            'spectrum_assignment': None,
            'snr': None,
            'metrics': {}
        }

    def _validate_algorithms(self) -> str:
        """Validate that all required algorithms are initialized.
        
        Returns:
            Empty string if valid, error message if invalid
        """
        if not hasattr(self, 'routing_algorithm') or self.routing_algorithm is None:
            return 'Routing algorithm not initialized'
        if not hasattr(self, 'spectrum_algorithm') or self.spectrum_algorithm is None:
            return 'Spectrum algorithm not initialized'
        if not hasattr(self, 'snr_algorithm') or self.snr_algorithm is None:
            return 'SNR algorithm not initialized'
        return ''

    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect metrics from all algorithms."""
        try:
            metrics = {}
            if hasattr(self.routing_algorithm, 'get_metrics'):
                metrics['routing'] = self.routing_algorithm.get_metrics()
            else:
                metrics['routing'] = {'error': 'routing algorithm missing get_metrics method'}

            if hasattr(self.spectrum_algorithm, 'get_metrics'):
                metrics['spectrum'] = self.spectrum_algorithm.get_metrics()
            else:
                metrics['spectrum'] = {'error': 'spectrum algorithm missing get_metrics method'}

            if hasattr(self.snr_algorithm, 'get_metrics'):
                metrics['snr'] = self.snr_algorithm.get_metrics()
            else:
                metrics['snr'] = {'error': 'snr algorithm missing get_metrics method'}

            return metrics
        except (AttributeError, TypeError) as e:
            return {'error': f'Failed to collect metrics: {str(e)}'}

    def process_request(self, source: Any, destination: Any, request: Any) -> Dict[str, Any]:
        """Process a single network request through the complete pipeline.
        
        Args:
            source: Source node identifier
            destination: Destination node identifier
            request: Request object containing traffic demand details
            
        Returns:
            Dictionary containing processing results
        """
        result = self._create_base_result(source, destination)

        # Validate algorithms
        validation_error = self._validate_algorithms()
        if validation_error:
            result['failure_reason'] = validation_error
            return result

        try:
            # Process through the pipeline
            self._process_pipeline(result, source, destination, request)
        except (KeyError, AttributeError, TypeError, ValueError) as e:
            result['failure_reason'] = f'Processing error: {str(e)}'
        except (RuntimeError, OSError, ImportError) as e:
            result['failure_reason'] = f'System error: {str(e)}'

        # Collect metrics from all algorithms
        result['metrics'] = self._collect_metrics()

        # Final safety check - ensure result is never None
        if result is None:
            result = {
                'source': source,
                'destination': destination,
                'success': False,
                'path': None,
                'spectrum_assignment': None,
                'snr': None,
                'metrics': {'error': 'Unexpected None result'},
                'failure_reason': 'Internal error: result was None'
            }

        return result

    def _process_pipeline(self, result: Dict[str, Any], source: Any, destination: Any, request: Any) -> None:
        """Process request through the complete pipeline."""
        # Step 1: Routing
        if not hasattr(self.routing_algorithm, 'route'):
            result['failure_reason'] = 'Routing algorithm missing route method'
            return
        path = self.routing_algorithm.route(source, destination, request)
        if not path:
            result['failure_reason'] = 'No path found'
            return
        result['path'] = path

        # Step 2: Spectrum Assignment
        if not hasattr(self.spectrum_algorithm, 'assign'):
            result['failure_reason'] = 'Spectrum algorithm missing assign method'
            return
        spectrum_assignment = self.spectrum_algorithm.assign(path, request)
        if not spectrum_assignment:
            result['failure_reason'] = 'No spectrum available'
            return
        result['spectrum_assignment'] = spectrum_assignment

        # Step 3: SNR Check and Allocation
        self._process_snr_and_allocation(result, path, spectrum_assignment, request)

    def _process_snr_and_allocation(self, result: Dict[str, Any], path: Any,
                                    spectrum_assignment: Dict[str, Any], request: Any) -> None:
        """Process SNR check and spectrum allocation."""
        if not hasattr(self.snr_algorithm, 'calculate_snr'):
            result['failure_reason'] = 'SNR algorithm missing calculate_snr method'
            return
        snr_value = self.snr_algorithm.calculate_snr(path, spectrum_assignment)
        result['snr'] = snr_value

        # Get modulation format for SNR threshold
        modulation = request.modulation if hasattr(request, 'modulation') else 'QPSK'
        topology = self.engine_props.get('topology', self.sdn_props.topology)
        path_length = sum(topology[path[i]][path[i + 1]]['length']
                          for i in range(len(path) - 1))

        if not hasattr(self.snr_algorithm, 'get_required_snr_threshold'):
            result['failure_reason'] = 'SNR algorithm missing get_required_snr_threshold method'
            return
        if not hasattr(self.snr_algorithm, 'is_snr_acceptable'):
            result['failure_reason'] = 'SNR algorithm missing is_snr_acceptable method'
            return

        required_snr = self.snr_algorithm.get_required_snr_threshold(modulation, path_length)
        snr_margin = self.config.get('snr_margin', 1.0)

        if self.snr_algorithm.is_snr_acceptable(snr_value, required_snr, snr_margin):
            self._allocate_spectrum(result, path, spectrum_assignment, request, required_snr, snr_value)
        else:
            result['failure_reason'] = f'SNR too low ({snr_value:.1f} dB < {required_snr:.1f} dB)'

    def _allocate_spectrum(self, result: Dict[str, Any], path: Any,
                           spectrum_assignment: Dict[str, Any], request: Any,
                           required_snr: float, snr_value: float) -> None:
        """Allocate spectrum for the request."""
        if not hasattr(self.spectrum_algorithm, 'allocate_spectrum'):
            result['failure_reason'] = 'Spectrum algorithm missing allocate_spectrum method'
            return

        request_id = getattr(request, 'id', hash((result['source'], result['destination'])))
        success = self.spectrum_algorithm.allocate_spectrum(
            path,
            spectrum_assignment['start_slot'],
            spectrum_assignment['end_slot'],
            spectrum_assignment['core_num'],
            spectrum_assignment['band'],
            request_id
        )

        if success:
            result['success'] = True
            result['required_snr'] = required_snr
            result['snr_margin'] = snr_value - required_snr
        else:
            result['failure_reason'] = 'Spectrum allocation failed'

    def get_algorithm_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all algorithms in the pipeline."""
        return {
            'routing': {
                'name': self.routing_algorithm.algorithm_name,
                'supported_topologies': self.routing_algorithm.supported_topologies,
                'class': type(self.routing_algorithm).__name__
            },
            'spectrum': {
                'name': self.spectrum_algorithm.algorithm_name,
                'supports_multiband': self.spectrum_algorithm.supports_multiband,
                'class': type(self.spectrum_algorithm).__name__
            },
            'snr': {
                'name': self.snr_algorithm.algorithm_name,
                'supports_multicore': self.snr_algorithm.supports_multicore,
                'class': type(self.snr_algorithm).__name__
            }
        }

    def reset_all_algorithms(self):
        """Reset state for all algorithms."""
        self.routing_algorithm.reset()
        self.spectrum_algorithm.reset()
        self.snr_algorithm.reset()


def create_simulation_pipeline(config: Dict[str, Any]) -> SimulationPipeline:
    """Create a complete simulation pipeline from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured SimulationPipeline instance
    """
    return SimulationPipeline(config)
