"""
Model I/O utilities for machine learning module.

This module handles saving and loading of trained models, including
versioning and metadata management.
"""

import json
import os
import pickle
from datetime import datetime
from typing import Any

import joblib
import numpy as np

# Optional imports for model export
try:
    import skl2onnx
    from skl2onnx import convert_sklearn

    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

try:
    from sklearn2pmml import sklearn2pmml

    HAS_PMML = True
except ImportError:
    HAS_PMML = False

from fusion.utils.logging_config import get_logger
from fusion.utils.os import create_directory

logger = get_logger(__name__)


def save_model(
    simulation_dict: dict[str, Any],
    model: Any,
    algorithm: str,
    erlang: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    """
    Save a trained machine learning model with metadata.

    :param simulation_dict: Dictionary containing simulation parameters
    :type simulation_dict: Dict[str, Any]
    :param model: Trained model object
    :type model: Any
    :param algorithm: Name of the algorithm
    :type algorithm: str
    :param erlang: Traffic volume value as string
    :type erlang: str
    :param metadata: Optional metadata to save with model
    :type metadata: Dict[str, Any]
    :return: Path where model was saved
    :rtype: str

    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> model = RandomForestClassifier()
        >>> # ... train model ...
        >>> path = save_model(sim_dict, model, "random_forest", "1000")
        >>> print(f"Model saved to: {path}")
    """
    # Create directory structure
    base_filepath = os.path.join("logs", algorithm, simulation_dict["train_file_path"])
    create_directory(directory_path=base_filepath)

    # Generate filenames
    model_filename = f"{algorithm}_{erlang}.joblib"
    model_path = os.path.join(base_filepath, model_filename)

    # Save the model
    joblib.dump(model, model_path)
    logger.info("Saved %s model to: %s", algorithm, model_path)

    # Save metadata if provided
    if metadata:
        metadata_filename = f"{algorithm}_{erlang}_metadata.json"
        metadata_path = os.path.join(base_filepath, metadata_filename)

        # Add timestamp and version info
        metadata["saved_at"] = datetime.now().isoformat()
        metadata["model_file"] = model_filename
        metadata["algorithm"] = algorithm
        metadata["erlang"] = erlang

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.debug("Saved model metadata to: %s", metadata_path)

    return model_path


def load_model(engine_properties: dict[str, Any]) -> Any:
    """
    Load a trained machine learning model.

    :param engine_properties: Properties from engine including model path info
    :type engine_properties: Dict[str, Any]
    :return: Loaded model object
    :rtype: Any
    :raises FileNotFoundError: If model file doesn't exist

    Example:
        >>> engine_props = {
        ...     'ml_model': 'random_forest',
        ...     'train_file_path': 'experiment_001',
        ...     'erlang': 1000.0
        ... }
        >>> model = load_model(engine_props)
    """
    model_filepath = os.path.join(
        "logs",
        engine_properties["ml_model"],
        engine_properties["train_file_path"],
        f"{engine_properties['ml_model']}_{str(int(engine_properties['erlang']))}.joblib",
    )

    if not os.path.exists(model_filepath):
        raise FileNotFoundError(f"Model file not found: {model_filepath}")

    model = joblib.load(filename=model_filepath)
    logger.info("Loaded %s model from: %s", engine_properties["ml_model"], model_filepath)

    return model


def load_model_with_metadata(
    engine_properties: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """
    Load a model along with its metadata.

    :param engine_properties: Properties from engine
    :type engine_properties: Dict[str, Any]
    :return: Tuple of (model, metadata)
    :rtype: Tuple[Any, Dict[str, Any]]

    Example:
        >>> model, metadata = load_model_with_metadata(engine_props)
        >>> print(f"Model trained on: {metadata.get('saved_at')}")
    """
    # Load the model
    model = load_model(engine_properties)

    # Try to load metadata
    metadata_filepath = os.path.join(
        "logs",
        engine_properties["ml_model"],
        engine_properties["train_file_path"],
        f"{engine_properties['ml_model']}_{str(int(engine_properties['erlang']))}_metadata.json",
    )

    metadata = {}
    if os.path.exists(metadata_filepath):
        with open(metadata_filepath, encoding="utf-8") as f:
            metadata = json.load(f)
        logger.debug("Loaded model metadata")
    else:
        logger.warning("No metadata found for model at: %s", metadata_filepath)

    return model, metadata


def save_model_ensemble(models: list[Any], simulation_dict: dict[str, Any], ensemble_name: str, erlang: str) -> str:
    """
    Save an ensemble of models.

    :param models: List of model objects
    :type models: List[Any]
    :param simulation_dict: Simulation parameters
    :type simulation_dict: Dict[str, Any]
    :param ensemble_name: Name for the ensemble
    :type ensemble_name: str
    :param erlang: Traffic volume
    :type erlang: str
    :return: Path where ensemble was saved
    :rtype: str

    Example:
        >>> models = [model1, model2, model3]
        >>> path = save_model_ensemble(models, sim_dict, "voting_ensemble", "1000")
    """
    base_filepath = os.path.join("logs", ensemble_name, simulation_dict["train_file_path"])
    create_directory(directory_path=base_filepath)

    ensemble_data = {
        "models": models,
        "n_models": len(models),
        "ensemble_type": ensemble_name,
    }

    ensemble_path = os.path.join(base_filepath, f"{ensemble_name}_{erlang}.joblib")
    joblib.dump(ensemble_data, ensemble_path)

    logger.info("Saved ensemble of %d models to: %s", len(models), ensemble_path)

    return ensemble_path


def export_model_for_deployment(
    model: Any,
    export_path: str,
    model_format: str = "onnx",
    input_sample: np.ndarray | None = None,
) -> str:
    """
    Export model to deployment-friendly format.

    :param model: Trained model
    :type model: Any
    :param export_path: Path for exported model
    :type export_path: str
    :param model_format: Export format ('onnx', 'pmml', 'pickle')
    :type model_format: str
    :param input_sample: Sample input for ONNX conversion
    :type input_sample: np.ndarray
    :return: Path to exported model
    :rtype: str
    :raises ValueError: If format not supported

    Example:
        >>> sample = np.array([[1, 2, 3, 4]])
        >>> export_path = export_model_for_deployment(
        ...     model, "model.onnx", "onnx", sample
        ... )
    """
    if model_format == "onnx":
        if not HAS_ONNX:
            logger.error("skl2onnx not installed. Install with: pip install skl2onnx")
            raise ImportError("skl2onnx not installed")

        try:
            if input_sample is None:
                raise ValueError("input_sample required for ONNX conversion")

            onnx_model = convert_sklearn(
                model,
                initial_types=[
                    (
                        "input",
                        skl2onnx.common.data_types.FloatTensorType(input_sample.shape),
                    )
                ],
            )

            with open(export_path, "wb") as f:
                f.write(onnx_model.SerializeToString())

        except ImportError:
            logger.error("skl2onnx not installed. Install with: pip install skl2onnx")
            raise

    elif model_format == "pickle":
        with open(export_path, "wb") as f:
            pickle.dump(model, f)

    elif model_format == "pmml":
        if not HAS_PMML:
            logger.error("sklearn2pmml not installed. Install with: pip install sklearn2pmml")
            raise ImportError("sklearn2pmml not installed")

        try:
            sklearn2pmml(model, export_path)
        except (ImportError, AttributeError) as e:
            logger.error("sklearn2pmml not installed or incompatible. Install with: pip install sklearn2pmml")
            raise ImportError("sklearn2pmml not available") from e
    else:
        raise ValueError(f"Export format '{model_format}' not supported")

    logger.info("Exported model to %s format: %s", model_format, export_path)
    return export_path


def check_model_compatibility(model_path: str, expected_features: list[str]) -> dict[str, Any]:
    """
    Check if saved model is compatible with current feature set.

    :param model_path: Path to model file
    :type model_path: str
    :param expected_features: List of expected feature names
    :type expected_features: List[str]
    :return: Compatibility report
    :rtype: Dict[str, Any]

    Example:
        >>> features = ['path_length', 'bandwidth', 'congestion']
        >>> report = check_model_compatibility("model.joblib", features)
        >>> print(f"Compatible: {report['compatible']}")
    """
    try:
        model = joblib.load(model_path)

        # Check if model has feature information
        if hasattr(model, "n_features_in_"):
            n_features = model.n_features_in_
            compatible = n_features == len(expected_features)
        else:
            n_features = None
            compatible = None  # Cannot determine

        # Try to get feature names if available
        if hasattr(model, "feature_names_in_"):
            model_features = list(model.feature_names_in_)
            missing_features = set(expected_features) - set(model_features)
            extra_features = set(model_features) - set(expected_features)
        else:
            model_features = None
            missing_features = None
            extra_features = None

        report = {
            "compatible": compatible,
            "n_features_expected": len(expected_features),
            "n_features_model": n_features,
            "model_features": model_features,
            "missing_features": list(missing_features) if missing_features else None,
            "extra_features": list(extra_features) if extra_features else None,
            "model_type": type(model).__name__,
        }

        return report

    except (FileNotFoundError, EOFError, ValueError, AttributeError) as e:
        logger.error("Error checking model compatibility: %s", e)
        return {"compatible": False, "error": str(e)}
