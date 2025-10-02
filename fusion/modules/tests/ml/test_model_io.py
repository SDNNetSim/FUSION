"""Unit tests for fusion.modules.ml.model_io module."""

from unittest.mock import Mock, mock_open, patch

import numpy as np
import pytest

from fusion.modules.ml.model_io import (
    check_model_compatibility,
    export_model_for_deployment,
    load_model,
    load_model_with_metadata,
    save_model,
    save_model_ensemble,
)


class TestSaveModel:
    """Tests for save_model function."""

    @patch("fusion.modules.ml.model_io.create_directory")
    @patch("fusion.modules.ml.model_io.joblib.dump")
    @patch("builtins.open", new_callable=mock_open)
    def test_save_model_creates_directory(
        self, mock_file: Mock, mock_dump: Mock, mock_create_dir: Mock
    ) -> None:
        """Test that save_model creates necessary directories."""
        # Arrange
        sim_dict = {"train_file_path": "experiment_001"}
        model = Mock()
        algorithm = "random_forest"
        erlang = "1000"

        # Act
        save_model(sim_dict, model, algorithm, erlang)

        # Assert
        mock_create_dir.assert_called_once()

    @patch("fusion.modules.ml.model_io.create_directory")
    @patch("fusion.modules.ml.model_io.joblib.dump")
    @patch("builtins.open", new_callable=mock_open)
    def test_save_model_dumps_model_file(
        self, mock_file: Mock, mock_dump: Mock, mock_create_dir: Mock
    ) -> None:
        """Test that save_model saves the model using joblib."""
        # Arrange
        sim_dict = {"train_file_path": "experiment_001"}
        model = Mock()
        algorithm = "random_forest"
        erlang = "1000"

        # Act
        save_model(sim_dict, model, algorithm, erlang)

        # Assert
        mock_dump.assert_called_once()
        assert mock_dump.call_args[0][0] is model

    @patch("fusion.modules.ml.model_io.create_directory")
    @patch("fusion.modules.ml.model_io.joblib.dump")
    @patch("builtins.open", new_callable=mock_open)
    def test_save_model_with_metadata_saves_metadata_file(
        self, mock_file: Mock, mock_dump: Mock, mock_create_dir: Mock
    ) -> None:
        """Test that metadata is saved when provided."""
        # Arrange
        sim_dict = {"train_file_path": "experiment_001"}
        model = Mock()
        algorithm = "random_forest"
        erlang = "1000"
        metadata = {"accuracy": 0.95, "training_samples": 10000}

        # Act
        save_model(sim_dict, model, algorithm, erlang, metadata)

        # Assert
        mock_file.assert_called_once()

    @patch("fusion.modules.ml.model_io.create_directory")
    @patch("fusion.modules.ml.model_io.joblib.dump")
    def test_save_model_returns_model_path(
        self, mock_dump: Mock, mock_create_dir: Mock
    ) -> None:
        """Test that save_model returns the path where model was saved."""
        # Arrange
        sim_dict = {"train_file_path": "experiment_001"}
        model = Mock()
        algorithm = "random_forest"
        erlang = "1000"

        # Act
        result = save_model(sim_dict, model, algorithm, erlang)

        # Assert
        assert isinstance(result, str)
        assert algorithm in result
        assert erlang in result


class TestLoadModel:
    """Tests for load_model function."""

    @patch("fusion.modules.ml.model_io.os.path.exists")
    @patch("fusion.modules.ml.model_io.joblib.load")
    def test_load_model_loads_from_correct_path(
        self, mock_load: Mock, mock_exists: Mock
    ) -> None:
        """Test that load_model loads from the correct file path."""
        # Arrange
        mock_exists.return_value = True
        mock_model = Mock()
        mock_load.return_value = mock_model
        engine_props = {
            "ml_model": "random_forest",
            "train_file_path": "experiment_001",
            "erlang": 1000.0,
        }

        # Act
        result = load_model(engine_props)

        # Assert
        mock_load.assert_called_once()
        assert result is mock_model

    @patch("fusion.modules.ml.model_io.os.path.exists")
    def test_load_model_raises_error_when_file_not_found(
        self, mock_exists: Mock
    ) -> None:
        """Test that FileNotFoundError is raised when model file missing."""
        # Arrange
        mock_exists.return_value = False
        engine_props = {
            "ml_model": "random_forest",
            "train_file_path": "experiment_001",
            "erlang": 1000.0,
        }

        # Act & Assert
        with pytest.raises(FileNotFoundError) as exc_info:
            load_model(engine_props)

        assert "Model file not found" in str(exc_info.value)


class TestLoadModelWithMetadata:
    """Tests for load_model_with_metadata function."""

    @patch("fusion.modules.ml.model_io.os.path.exists")
    @patch("fusion.modules.ml.model_io.joblib.load")
    @patch("builtins.open", new_callable=mock_open, read_data='{"accuracy": 0.95}')
    def test_load_with_metadata_returns_model_and_metadata(
        self, mock_file: Mock, mock_load: Mock, mock_exists: Mock
    ) -> None:
        """Test that both model and metadata are loaded."""
        # Arrange
        mock_exists.return_value = True
        mock_model = Mock()
        mock_load.return_value = mock_model
        engine_props = {
            "ml_model": "random_forest",
            "train_file_path": "experiment_001",
            "erlang": 1000.0,
        }

        # Act
        model, metadata = load_model_with_metadata(engine_props)

        # Assert
        assert model is mock_model
        assert isinstance(metadata, dict)

    @patch("fusion.modules.ml.model_io.os.path.exists")
    @patch("fusion.modules.ml.model_io.joblib.load")
    def test_load_without_metadata_returns_empty_dict(
        self, mock_load: Mock, mock_exists: Mock
    ) -> None:
        """Test that empty dict is returned when metadata missing."""

        # Arrange
        def exists_side_effect(path: str) -> bool:
            # Model file exists, metadata doesn't
            return "metadata" not in path

        mock_exists.side_effect = exists_side_effect
        mock_model = Mock()
        mock_load.return_value = mock_model
        engine_props = {
            "ml_model": "random_forest",
            "train_file_path": "experiment_001",
            "erlang": 1000.0,
        }

        # Act
        model, metadata = load_model_with_metadata(engine_props)

        # Assert
        assert model is mock_model
        assert metadata == {}


class TestSaveModelEnsemble:
    """Tests for save_model_ensemble function."""

    @patch("fusion.modules.ml.model_io.create_directory")
    @patch("fusion.modules.ml.model_io.joblib.dump")
    def test_save_ensemble_saves_all_models(
        self, mock_dump: Mock, mock_create_dir: Mock
    ) -> None:
        """Test that ensemble saves all models in a list."""
        # Arrange
        models = [Mock(), Mock(), Mock()]
        sim_dict = {"train_file_path": "experiment_001"}
        ensemble_name = "voting_ensemble"
        erlang = "1000"

        # Act
        save_model_ensemble(models, sim_dict, ensemble_name, erlang)

        # Assert
        mock_dump.assert_called_once()
        saved_data = mock_dump.call_args[0][0]
        assert saved_data["n_models"] == 3
        assert len(saved_data["models"]) == 3

    @patch("fusion.modules.ml.model_io.create_directory")
    @patch("fusion.modules.ml.model_io.joblib.dump")
    def test_save_ensemble_returns_path(
        self, mock_dump: Mock, mock_create_dir: Mock
    ) -> None:
        """Test that save_model_ensemble returns save path."""
        # Arrange
        models = [Mock()]
        sim_dict = {"train_file_path": "experiment_001"}
        ensemble_name = "voting_ensemble"
        erlang = "1000"

        # Act
        result = save_model_ensemble(models, sim_dict, ensemble_name, erlang)

        # Assert
        assert isinstance(result, str)
        assert ensemble_name in result


class TestExportModelForDeployment:
    """Tests for export_model_for_deployment function."""

    @patch("fusion.modules.ml.model_io.pickle.dump")
    @patch("builtins.open", new_callable=mock_open)
    def test_export_pickle_format_creates_pickle_file(
        self, mock_file: Mock, mock_pickle_dump: Mock
    ) -> None:
        """Test that pickle format exports correctly."""
        # Arrange
        model = Mock()
        export_path = "model.pkl"

        # Act
        result = export_model_for_deployment(model, export_path, "pickle")

        # Assert
        assert result == export_path
        mock_file.assert_called_once_with(export_path, "wb")
        mock_pickle_dump.assert_called_once()

    @patch("fusion.modules.ml.model_io.HAS_ONNX", True)
    def test_export_onnx_without_input_sample_raises_error(self) -> None:
        """Test that ONNX export requires input sample."""
        # Arrange
        model = Mock()
        export_path = "model.onnx"

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            export_model_for_deployment(model, export_path, "onnx")

        assert "input_sample required" in str(exc_info.value)

    def test_export_invalid_format_raises_error(self) -> None:
        """Test that invalid format raises ValueError."""
        # Arrange
        model = Mock()
        export_path = "model.unknown"

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            export_model_for_deployment(model, export_path, "invalid_format")

        assert "not supported" in str(exc_info.value)


class TestCheckModelCompatibility:
    """Tests for check_model_compatibility function."""

    @patch("fusion.modules.ml.model_io.joblib.load")
    def test_check_compatibility_with_matching_features(self, mock_load: Mock) -> None:
        """Test compatibility check with matching feature count."""
        # Arrange
        mock_model = Mock(spec=["n_features_in_"])
        mock_model.n_features_in_ = 5
        mock_load.return_value = mock_model
        expected_features = ["f1", "f2", "f3", "f4", "f5"]

        # Act
        result = check_model_compatibility("model.joblib", expected_features)

        # Assert
        assert result["compatible"] is True
        assert result["n_features_expected"] == 5
        assert result["n_features_model"] == 5

    @patch("fusion.modules.ml.model_io.joblib.load")
    def test_check_compatibility_with_mismatched_features(
        self, mock_load: Mock
    ) -> None:
        """Test compatibility check with mismatched feature count."""
        # Arrange
        mock_model = Mock(spec=["n_features_in_"])
        mock_model.n_features_in_ = 3
        mock_load.return_value = mock_model
        expected_features = ["f1", "f2", "f3", "f4", "f5"]

        # Act
        result = check_model_compatibility("model.joblib", expected_features)

        # Assert
        assert result["compatible"] is False

    @patch("fusion.modules.ml.model_io.joblib.load")
    def test_check_compatibility_with_feature_names(self, mock_load: Mock) -> None:
        """Test compatibility check includes feature names if available."""
        # Arrange
        mock_model = Mock()
        mock_model.n_features_in_ = 3
        mock_model.feature_names_in_ = np.array(["f1", "f2", "f3"])
        mock_load.return_value = mock_model
        expected_features = ["f1", "f2", "f4"]

        # Act
        result = check_model_compatibility("model.joblib", expected_features)

        # Assert
        assert "model_features" in result
        assert "missing_features" in result
        assert "f4" in result["missing_features"]
        assert "extra_features" in result
        assert "f3" in result["extra_features"]

    @patch("fusion.modules.ml.model_io.joblib.load")
    def test_check_compatibility_handles_load_error(self, mock_load: Mock) -> None:
        """Test that load errors are handled gracefully."""
        # Arrange
        mock_load.side_effect = FileNotFoundError("File not found")
        expected_features = ["f1", "f2"]

        # Act
        result = check_model_compatibility("nonexistent.joblib", expected_features)

        # Assert
        assert result["compatible"] is False
        assert "error" in result
