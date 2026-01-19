"""
ML Control Policy for path selection using pre-trained models.

This module provides MLControlPolicy, which loads pre-trained ML models
(PyTorch, sklearn, ONNX) and uses them for path selection inference.

MLControlPolicy implements the ControlPolicy protocol with:
- Multi-framework support via file extension detection
- Robust fallback to heuristic policies on errors
- Action masking for feasibility constraints
- Feature engineering matching RL observation space

MLControlPolicy is deployment-only: no online training, update() is no-op.

Example:
    >>> from fusion.policies.ml_policy import MLControlPolicy
    >>> policy = MLControlPolicy("model.pt", fallback_type="first_feasible")
    >>> action = policy.select_action(request, options, network_state)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:
    from fusion.domain.network_state import NetworkState
    from fusion.domain.request import Request
    from fusion.interfaces.control_policy import ControlPolicy
    from fusion.modules.rl.adapter import PathOption

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """
    Build feature vectors for ML model inference.

    Creates fixed-size feature vectors from Request and PathOption
    inputs, with padding for variable numbers of paths.

    The feature layout matches the RL training observation space,
    ensuring model compatibility.

    :ivar k_paths: Expected number of paths (for padding).
    :vartype k_paths: int
    :ivar features_per_path: Number of features extracted per path.
    :vartype features_per_path: int

    Example::

        >>> builder = FeatureBuilder(k_paths=5)
        >>> features = builder.build(request, options, network_state)
        >>> features.shape
        (21,)  # 1 + 5*4
    """

    FEATURES_PER_PATH = 4

    # Normalization constants
    MAX_BANDWIDTH_GBPS = 1000.0
    MAX_WEIGHT_KM = 10000.0
    MAX_SLOTS = 100.0

    def __init__(self, k_paths: int = 5) -> None:
        """
        Initialize feature builder.

        :param k_paths: Expected number of path options.
        :type k_paths: int
        """
        self.k_paths = k_paths
        self._feature_size = 1 + k_paths * self.FEATURES_PER_PATH

    @property
    def feature_size(self) -> int:
        """
        Total size of feature vector.

        :return: Size of the feature vector.
        :rtype: int
        """
        return self._feature_size

    def build(
        self,
        request: Request,
        options: list[PathOption],
        network_state: NetworkState,
    ) -> np.ndarray:
        """
        Build feature vector from inputs.

        :param request: The request being processed.
        :type request: Request
        :param options: Available path options.
        :type options: list[PathOption]
        :param network_state: Current network state (for future extensions).
        :type network_state: NetworkState
        :return: Feature vector of shape (feature_size,).
        :rtype: np.ndarray
        """
        features: list[float] = []

        # Request-level features
        bandwidth = getattr(request, "bandwidth_gbps", 0.0) if request else 0.0
        features.append(self._normalize_bandwidth(bandwidth))

        # Per-path features
        for i in range(self.k_paths):
            if i < len(options):
                features.extend(self._extract_path_features(options[i]))
            else:
                features.extend(self._get_padding_features())

        return np.array(features, dtype=np.float32)

    def _normalize_bandwidth(self, bandwidth_gbps: float) -> float:
        """
        Normalize bandwidth to [0, 1] range.

        :param bandwidth_gbps: Bandwidth in Gbps.
        :type bandwidth_gbps: float
        :return: Normalized bandwidth value.
        :rtype: float
        """
        return bandwidth_gbps / self.MAX_BANDWIDTH_GBPS

    def _extract_path_features(self, opt: PathOption) -> list[float]:
        """
        Extract features from a single path option.

        :param opt: Path option to extract features from.
        :type opt: PathOption
        :return: List of feature values.
        :rtype: list[float]
        """
        return [
            opt.weight_km / self.MAX_WEIGHT_KM,
            opt.congestion,
            1.0 if opt.is_feasible else 0.0,
            (opt.slots_needed or 0) / self.MAX_SLOTS,
        ]

    def _get_padding_features(self) -> list[float]:
        """
        Get padding features for missing paths.

        Padding values chosen to represent "worst case" path:

        - weight: 0.0 (no path)
        - congestion: 1.0 (fully congested)
        - feasible: 0.0 (not available)
        - slots: 0.0 (not needed)

        :return: List of padding feature values.
        :rtype: list[float]
        """
        return [0.0, 1.0, 0.0, 0.0]


class ModelWrapper(Protocol):
    """Protocol for model wrappers providing predict interface."""

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict action scores from features.

        :param features: Feature array of shape (feature_size,) or (batch, feature_size).
        :type features: np.ndarray
        :return: Scores/logits for each action of shape (k_paths,) or (batch, k_paths).
        :rtype: np.ndarray
        """
        ...


class TorchModelWrapper:
    """Wrapper for PyTorch models."""

    def __init__(self, model: Any, device: str = "cpu") -> None:
        """
        Initialize torch model wrapper.

        :param model: PyTorch nn.Module.
        :type model: Any
        :param device: Device to run inference on.
        :type device: str
        """
        import torch

        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Run inference through PyTorch model.

        :param features: Input feature array.
        :type features: np.ndarray
        :return: Model output scores.
        :rtype: np.ndarray
        """
        import torch

        # Ensure 2D input
        if features.ndim == 1:
            features = features.reshape(1, -1)

        with torch.no_grad():
            tensor = torch.from_numpy(features).float().to(self.device)
            output = self.model(tensor)
            result: np.ndarray = output.cpu().numpy().squeeze()
            return result


class SklearnModelWrapper:
    """Wrapper for sklearn models."""

    def __init__(self, model: Any) -> None:
        """
        Initialize sklearn model wrapper.

        :param model: sklearn model with predict_proba or predict method.
        :type model: Any
        """
        self.model = model
        self._has_proba = hasattr(model, "predict_proba")

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Run inference through sklearn model.

        :param features: Input feature array.
        :type features: np.ndarray
        :return: Model output scores or predictions.
        :rtype: np.ndarray
        """
        # Ensure 2D input
        if features.ndim == 1:
            features = features.reshape(1, -1)

        if self._has_proba:
            probs = self.model.predict_proba(features)
            result: np.ndarray = np.asarray(probs).squeeze()
            return result
        else:
            # For regressors, predict returns shape (n_samples,) or (n_samples, n_outputs)
            output = self.model.predict(features)
            result = np.atleast_1d(np.asarray(output).squeeze())
            return result


class OnnxModelWrapper:
    """Wrapper for ONNX models."""

    def __init__(self, session: Any) -> None:
        """
        Initialize ONNX model wrapper.

        :param session: onnxruntime InferenceSession.
        :type session: Any
        """
        self.session = session
        self._input_name = session.get_inputs()[0].name

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Run inference through ONNX model.

        :param features: Input feature array.
        :type features: np.ndarray
        :return: Model output scores.
        :rtype: np.ndarray
        """
        # Ensure 2D input and float32
        if features.ndim == 1:
            features = features.reshape(1, -1)
        features = features.astype(np.float32)

        outputs = self.session.run(None, {self._input_name: features})
        result: np.ndarray = np.asarray(outputs[0]).squeeze()
        return result


class CallableModelWrapper:
    """Wrapper for callable models (functions or objects with __call__)."""

    def __init__(self, model: Callable[[np.ndarray], np.ndarray]) -> None:
        """
        Initialize callable model wrapper.

        :param model: Any callable that takes features and returns scores.
        :type model: Callable[[np.ndarray], np.ndarray]
        """
        self._callable = model

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Run inference through callable.

        :param features: Input feature array.
        :type features: np.ndarray
        :return: Model output scores.
        :rtype: np.ndarray
        """
        return self._callable(features)


def _load_torch_model(model_path: Path, device: str) -> TorchModelWrapper:
    """
    Load a PyTorch model from file.

    :param model_path: Path to the model file.
    :type model_path: Path
    :param device: Device to load model onto.
    :type device: str
    :return: Wrapped PyTorch model.
    :rtype: TorchModelWrapper
    :raises ImportError: If PyTorch is not installed.
    :raises ValueError: If model file contains only state_dict.
    """
    try:
        import torch
    except ImportError as e:
        raise ImportError("PyTorch not installed. Install with: pip install torch") from e

    model = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )  # nosec B614 - Loading trusted model files from local filesystem

    # Handle state_dict or full model
    if isinstance(model, dict):
        raise ValueError(
            "Model file contains state_dict only. Please save the full model with torch.save(model, path) or provide model architecture."
        )

    return TorchModelWrapper(model, device)


def _load_sklearn_model(model_path: Path) -> SklearnModelWrapper:
    """
    Load a sklearn model from joblib or pickle file.

    :param model_path: Path to the model file.
    :type model_path: Path
    :return: Wrapped sklearn model.
    :rtype: SklearnModelWrapper
    :raises ImportError: If joblib is not installed.
    """
    try:
        import joblib
    except ImportError as e:
        raise ImportError("joblib not installed. Install with: pip install joblib") from e

    model = joblib.load(model_path)
    return SklearnModelWrapper(model)


def _load_onnx_model(model_path: Path) -> OnnxModelWrapper:
    """
    Load an ONNX model.

    :param model_path: Path to the ONNX model file.
    :type model_path: Path
    :return: Wrapped ONNX model.
    :rtype: OnnxModelWrapper
    :raises ImportError: If onnxruntime is not installed.
    """
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise ImportError("onnxruntime not installed. Install with: pip install onnxruntime") from e

    session = ort.InferenceSession(str(model_path))
    return OnnxModelWrapper(session)


def load_model(model_path: str, device: str = "cpu") -> ModelWrapper:
    """
    Load a model based on file extension.

    Supported formats:

    - .pt, .pth: PyTorch models
    - .joblib, .pkl: sklearn models (via joblib)
    - .onnx: ONNX models

    :param model_path: Path to model file.
    :type model_path: str
    :param device: Device for PyTorch models ("cpu", "cuda", "mps").
    :type device: str
    :return: ModelWrapper with predict() method.
    :rtype: ModelWrapper
    :raises FileNotFoundError: If model file doesn't exist.
    :raises ValueError: If file extension not supported.
    :raises ImportError: If required framework not installed.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    suffix = path.suffix.lower()

    if suffix in (".pt", ".pth"):
        return _load_torch_model(path, device)
    elif suffix in (".joblib", ".pkl"):
        return _load_sklearn_model(path)
    elif suffix == ".onnx":
        return _load_onnx_model(path)
    else:
        raise ValueError(f"Unsupported model format: {suffix}. Supported: .pt, .pth, .joblib, .pkl, .onnx")


class MLControlPolicy:
    """
    ML-based control policy for path selection.

    Loads pre-trained ML models and uses them for deterministic inference.
    Implements robust fallback to heuristic policies when model fails.

    This is a deployment-only policy: update() is a no-op.

    :ivar model: Wrapped model with predict() interface.
    :vartype model: ModelWrapper
    :ivar fallback: Fallback heuristic policy.
    :vartype fallback: HeuristicPolicy
    :ivar feature_builder: Feature vector constructor.
    :vartype feature_builder: FeatureBuilder

    Example::

        >>> policy = MLControlPolicy("model.pt", fallback_type="first_feasible")
        >>> action = policy.select_action(request, options, network_state)
        >>> print(policy.get_stats())  # View fallback statistics
    """

    def __init__(
        self,
        model_path: str | None = None,
        model: ModelWrapper | None = None,
        device: str = "cpu",
        k_paths: int = 5,
        fallback_policy: Any | None = None,
        fallback_type: str = "first_feasible",
    ) -> None:
        """
        Initialize ML control policy.

        :param model_path: Path to model file. Mutually exclusive with model.
        :type model_path: str | None
        :param model: Pre-loaded model wrapper. Mutually exclusive with model_path.
        :type model: ModelWrapper | None
        :param device: Device for PyTorch models ("cpu", "cuda", "mps").
        :type device: str
        :param k_paths: Expected number of path options (for feature builder).
        :type k_paths: int
        :param fallback_policy: Explicit fallback policy instance.
        :type fallback_policy: Any | None
        :param fallback_type: Fallback type if policy not provided:
            "first_feasible" (default), "shortest_feasible",
            "least_congested", "random".
        :type fallback_type: str
        :raises ValueError: If neither or both model_path and model provided.
        :raises FileNotFoundError: If model file doesn't exist.
        :raises ImportError: If required framework not installed.
        """
        # Validate inputs
        if model_path is None and model is None:
            raise ValueError("Either model_path or model must be provided")
        if model_path is not None and model is not None:
            raise ValueError("Cannot provide both model_path and model")

        # Load model
        if model_path is not None:
            self._model: ModelWrapper = load_model(model_path, device)
            self._model_path = model_path
        else:
            # model is guaranteed non-None here due to validation above
            assert model is not None
            self._model = model
            self._model_path = "<provided>"

        # Setup feature builder
        self._feature_builder = FeatureBuilder(k_paths=k_paths)
        self._k_paths = k_paths

        # Setup fallback
        if fallback_policy is not None:
            self._fallback: ControlPolicy = fallback_policy
        else:
            self._fallback = self._create_fallback(fallback_type)

        # Statistics tracking
        self._total_calls = 0
        self._fallback_calls = 0
        self._error_types: dict[str, int] = {}

    def _create_fallback(self, fallback_type: str) -> ControlPolicy:
        """
        Create fallback policy from type string.

        :param fallback_type: Type of fallback policy.
        :type fallback_type: str
        :return: Instantiated fallback policy.
        :rtype: ControlPolicy
        :raises ValueError: If fallback_type is unknown.
        """
        from fusion.policies.heuristic_policy import (
            FirstFeasiblePolicy,
            LeastCongestedPolicy,
            RandomFeasiblePolicy,
            ShortestFeasiblePolicy,
        )

        fallback_map: dict[str, type[ControlPolicy]] = {
            "first_feasible": FirstFeasiblePolicy,
            "shortest_feasible": ShortestFeasiblePolicy,
            "least_congested": LeastCongestedPolicy,
            "random": RandomFeasiblePolicy,
        }

        if fallback_type not in fallback_map:
            raise ValueError(f"Unknown fallback type: {fallback_type}. Options: {list(fallback_map.keys())}")

        return fallback_map[fallback_type]()

    @property
    def fallback(self) -> ControlPolicy:
        """
        Current fallback policy.

        :return: The fallback policy instance.
        :rtype: ControlPolicy
        """
        return self._fallback

    @property
    def fallback_rate(self) -> float:
        """
        Percentage of calls that used fallback.

        :return: Fallback rate as a fraction (0.0 to 1.0).
        :rtype: float
        """
        if self._total_calls == 0:
            return 0.0
        return self._fallback_calls / self._total_calls

    def get_stats(self) -> dict[str, Any]:
        """
        Get fallback statistics.

        :return: Dictionary with total_calls, fallback_calls,
            fallback_rate, and error_types.
        :rtype: dict[str, Any]
        """
        return {
            "total_calls": self._total_calls,
            "fallback_calls": self._fallback_calls,
            "fallback_rate": self.fallback_rate,
            "error_types": self._error_types.copy(),
        }

    def reset_stats(self) -> None:
        """Reset fallback statistics."""
        self._total_calls = 0
        self._fallback_calls = 0
        self._error_types.clear()

    def select_action(
        self,
        request: Request,
        options: list[PathOption],
        network_state: NetworkState,
    ) -> int:
        """
        Select an action using ML model with fallback.

        Flow:

        1. Build features from inputs
        2. Run model inference
        3. Apply action masking (infeasible -> -inf)
        4. Select argmax action
        5. Validate action is feasible
        6. On any error/invalid action: use fallback

        :param request: The request to serve.
        :type request: Request
        :param options: Available path options.
        :type options: list[PathOption]
        :param network_state: Current network state.
        :type network_state: NetworkState
        :return: Path index (0 to len(options)-1), or -1 if no valid action.
        :rtype: int
        """
        self._total_calls += 1

        # Early return for empty options
        if not options:
            return -1

        # Check if any feasible options exist
        feasible_indices = [i for i, opt in enumerate(options) if opt.is_feasible]
        if not feasible_indices:
            return -1

        try:
            # Step 1: Build features
            features = self._feature_builder.build(request, options, network_state)

            # Step 2: Run inference
            raw_output = self._model.predict(features)

            # Step 3: Validate output
            if not self._validate_output(raw_output, len(options)):
                return self._use_fallback(request, options, network_state, "invalid_output")

            # Step 4: Apply mask and select
            action = self._apply_mask_and_select(raw_output, options)

            # Step 5: Validate action
            if self._is_valid_action(action, options):
                return action

            # Invalid action - fallback
            return self._use_fallback(request, options, network_state, "infeasible_action")

        except ImportError as e:
            logger.error("ML framework import error: %s", e)
            return self._use_fallback(request, options, network_state, "import_error")

        except RuntimeError as e:
            logger.warning("Model runtime error: %s", e)
            return self._use_fallback(request, options, network_state, "runtime_error")

        except Exception as e:
            logger.warning("Unexpected ML error: %s", e)
            return self._use_fallback(request, options, network_state, "unknown_error")

    def _validate_output(self, output: np.ndarray, expected_len: int) -> bool:
        """
        Check if model output is valid.

        :param output: Model output array.
        :type output: np.ndarray
        :param expected_len: Expected minimum length of output.
        :type expected_len: int
        :return: True if output is valid.
        :rtype: bool
        """
        if output is None:
            return False

        # Handle scalar output
        if output.ndim == 0:
            idx = int(output.item())
            return 0 <= idx < expected_len

        # Handle vector output
        if output.ndim == 1:
            # Must have at least one score
            return len(output) > 0

        # Multi-dimensional - unexpected for inference
        return False

    def _apply_mask_and_select(self, raw_output: np.ndarray, options: list[PathOption]) -> int:
        """
        Apply feasibility mask and select best action.

        :param raw_output: Raw model output scores.
        :type raw_output: np.ndarray
        :param options: Available path options.
        :type options: list[PathOption]
        :return: Selected action index.
        :rtype: int
        """
        # Handle scalar output (direct action index)
        if raw_output.ndim == 0:
            return int(raw_output.item())

        # Build mask
        scores = raw_output.copy()
        for i, opt in enumerate(options):
            if i < len(scores) and not opt.is_feasible:
                scores[i] = float("-inf")

        # Also mask any scores beyond options length
        if len(scores) > len(options):
            scores[len(options) :] = float("-inf")

        # Select argmax
        if np.all(np.isinf(scores)):
            return -1

        return int(np.argmax(scores))

    def _is_valid_action(self, action: int, options: list[PathOption]) -> bool:
        """
        Check if action is valid and feasible.

        :param action: Action index to validate.
        :type action: int
        :param options: Available path options.
        :type options: list[PathOption]
        :return: True if action is valid and feasible.
        :rtype: bool
        """
        if action < 0 or action >= len(options):
            return False
        return options[action].is_feasible

    def _use_fallback(
        self,
        request: Request,
        options: list[PathOption],
        network_state: NetworkState,
        reason: str = "unknown",
    ) -> int:
        """
        Use fallback and track statistics.

        :param request: The request to serve.
        :type request: Request
        :param options: Available path options.
        :type options: list[PathOption]
        :param network_state: Current network state.
        :type network_state: NetworkState
        :param reason: Reason for using fallback.
        :type reason: str
        :return: Selected action from fallback policy.
        :rtype: int
        """
        self._fallback_calls += 1
        self._error_types[reason] = self._error_types.get(reason, 0) + 1
        return self._fallback.select_action(request, options, network_state)

    def update(self, request: Request, action: int, reward: float) -> None:
        """
        Update policy based on experience.

        MLControlPolicy is deployment-only and does not learn online.
        This method is a no-op.

        :param request: The request that was served (ignored).
        :type request: Request
        :param action: The action taken (ignored).
        :type action: int
        :param reward: The reward received (ignored).
        :type reward: float
        """
        pass

    def get_name(self) -> str:
        """
        Return the policy name for logging.

        :return: Policy name with model path.
        :rtype: str
        """
        return f"MLControlPolicy({self._model_path})"
