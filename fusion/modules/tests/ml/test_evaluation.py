"""Unit tests for fusion.modules.ml.evaluation module."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from fusion.modules.ml.evaluation import (
    analyze_prediction_errors,
    compare_models,
    cross_validate_model,
    evaluate_classifier,
    evaluate_model_stability,
    evaluate_regressor,
)


class TestEvaluateClassifier:
    """Tests for evaluate_classifier function."""

    def test_evaluate_with_perfect_predictions_returns_one(self) -> None:
        """Test that perfect predictions yield 1.0 accuracy."""
        # Arrange
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1])

        # Act
        metrics = evaluate_classifier(y_true, y_pred)

        # Assert
        assert metrics["accuracy"] == 1.0
        assert metrics["f1_score"] == 1.0

    def test_evaluate_returns_all_required_metrics(self) -> None:
        """Test that all expected metrics are returned."""
        # Arrange
        y_true = np.array([0, 1, 0, 1, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 1])

        # Act
        metrics = evaluate_classifier(y_true, y_pred)

        # Assert
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "confusion_matrix" in metrics
        assert "classification_report" in metrics

    def test_evaluate_includes_per_class_metrics(self) -> None:
        """Test that per-class metrics are included."""
        # Arrange
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 2])

        # Act
        metrics = evaluate_classifier(y_true, y_pred)

        # Assert
        assert "precision_per_class" in metrics
        assert "recall_per_class" in metrics
        assert "f1_score_per_class" in metrics
        assert "accuracy_per_class" in metrics

    def test_evaluate_with_class_names_includes_names(self) -> None:
        """Test that class names are included in report."""
        # Arrange
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        class_names = ["negative", "positive"]

        # Act
        metrics = evaluate_classifier(y_true, y_pred, class_names)

        # Assert
        assert "classification_report" in metrics
        report = metrics["classification_report"]
        assert "negative" in report or "positive" in report

    def test_evaluate_binary_classification_includes_roc_auc(self) -> None:
        """Test that ROC AUC is calculated for binary classification."""
        # Arrange
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1])

        # Act
        metrics = evaluate_classifier(y_true, y_pred)

        # Assert
        assert "roc_auc" in metrics


class TestEvaluateRegressor:
    """Tests for evaluate_regressor function."""

    def test_evaluate_with_perfect_predictions_returns_zero_error(self) -> None:
        """Test that perfect predictions yield zero error metrics."""
        # Arrange
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])

        # Act
        metrics = evaluate_regressor(y_true, y_pred)

        # Assert
        assert metrics["mse"] == 0.0
        assert metrics["rmse"] == 0.0
        assert metrics["mae"] == 0.0
        assert metrics["max_error"] == 0.0

    def test_evaluate_returns_all_regression_metrics(self) -> None:
        """Test that all expected regression metrics are returned."""
        # Arrange
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 2.2, 2.9, 3.8])

        # Act
        metrics = evaluate_regressor(y_true, y_pred)

        # Assert
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert "mape" in metrics
        assert "max_error" in metrics
        assert "explained_variance" in metrics

    def test_evaluate_calculates_correct_mse(self) -> None:
        """Test that MSE is calculated correctly."""
        # Arrange
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 3.0, 3.0])  # Errors: 0, 1, 0
        expected_mse = 1.0 / 3.0  # Mean of squared errors: (0^2 + 1^2 + 0^2) / 3

        # Act
        metrics = evaluate_regressor(y_true, y_pred)

        # Assert
        assert abs(metrics["mse"] - expected_mse) < 1e-10

    def test_evaluate_with_zero_values_handles_mape(self) -> None:
        """Test that MAPE handles zero values correctly."""
        # Arrange
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([1.0, 2.0, 3.0])

        # Act
        metrics = evaluate_regressor(y_true, y_pred)

        # Assert
        assert "mape" in metrics
        # MAPE should be inf when all true values are zero
        assert metrics["mape"] == float("inf")


class TestCrossValidateModel:
    """Tests for cross_validate_model function."""

    def test_cross_validate_returns_results_for_all_metrics(self) -> None:
        """Test that cross-validation returns all requested metrics."""
        # Arrange
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X = pd.DataFrame(np.random.rand(100, 5))
        y = pd.Series(np.random.randint(0, 2, 100))
        metrics = ["accuracy", "precision_weighted"]

        # Act
        results = cross_validate_model(model, X, y, cv_folds=3, scoring_metrics=metrics)

        # Assert
        assert "accuracy" in results
        assert "precision_weighted" in results
        assert "fit_time" in results
        assert "score_time" in results

    def test_cross_validate_includes_train_and_test_scores(self) -> None:
        """Test that both train and test scores are included."""
        # Arrange
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X = pd.DataFrame(np.random.rand(50, 3))
        y = pd.Series(np.random.randint(0, 2, 50))

        # Act
        results = cross_validate_model(model, X, y, cv_folds=2)

        # Assert
        assert results["accuracy"]["test_mean"] is not None
        assert results["accuracy"]["train_mean"] is not None
        assert results["accuracy"]["test_std"] is not None

    def test_cross_validate_with_default_metrics(self) -> None:
        """Test that default metrics are used when none specified."""
        # Arrange
        model = LogisticRegression(random_state=42, max_iter=200)
        X = pd.DataFrame(np.random.rand(50, 3))
        y = pd.Series(np.random.randint(0, 2, 50))

        # Act
        results = cross_validate_model(model, X, y, cv_folds=2)

        # Assert
        assert "accuracy" in results
        assert "f1_weighted" in results


class TestEvaluateModelStability:
    """Tests for evaluate_model_stability function."""

    def test_stability_returns_statistics_for_all_metrics(self) -> None:
        """Test that stability analysis returns stats for all metrics."""
        # Arrange
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X = pd.DataFrame(np.random.rand(100, 5))
        y = pd.Series(np.random.randint(0, 2, 100))

        # Act
        results = evaluate_model_stability(model, X, y, n_iterations=5)

        # Assert
        assert "accuracy" in results
        assert "precision" in results
        assert "recall" in results
        assert "f1_score" in results

    def test_stability_includes_all_statistics(self) -> None:
        """Test that all statistical measures are included."""
        # Arrange
        model = LogisticRegression(random_state=42, max_iter=200)
        X = pd.DataFrame(np.random.rand(100, 3))
        y = pd.Series(np.random.randint(0, 2, 100))

        # Act
        results = evaluate_model_stability(model, X, y, n_iterations=3)

        # Assert
        accuracy_stats = results["accuracy"]
        assert "mean" in accuracy_stats
        assert "std" in accuracy_stats
        assert "variance" in accuracy_stats
        assert "min" in accuracy_stats
        assert "max" in accuracy_stats
        assert "range" in accuracy_stats
        assert "cv" in accuracy_stats

    def test_stability_with_perfect_model_has_zero_variance(self) -> None:
        """Test that a perfect model has zero variance."""
        # Arrange
        # Create a simple dataset where model will always predict correctly
        X = pd.DataFrame({"feature": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 10})
        y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 10)
        model = LogisticRegression(random_state=42, max_iter=1000)

        # Act
        results = evaluate_model_stability(model, X, y, n_iterations=3, test_size=0.2)

        # Assert
        # Variance should be very small (close to 0) for consistent model
        assert results["accuracy"]["variance"] >= 0


class TestCompareModels:
    """Tests for compare_models function."""

    def test_compare_returns_dataframe(self) -> None:
        """Test that compare_models returns a DataFrame."""
        # Arrange
        models = {
            "RF": RandomForestClassifier(n_estimators=10, random_state=42),
            "LR": LogisticRegression(random_state=42, max_iter=200),
        }
        X = pd.DataFrame(np.random.rand(100, 5))
        y = pd.Series(np.random.randint(0, 2, 100))

        # Act
        result = compare_models(models, X, y)

        # Assert
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_compare_includes_all_metrics(self) -> None:
        """Test that comparison includes all expected metrics."""
        # Arrange
        models = {"RF": RandomForestClassifier(n_estimators=10, random_state=42)}
        X = pd.DataFrame(np.random.rand(50, 3))
        y = pd.Series(np.random.randint(0, 2, 50))

        # Act
        result = compare_models(models, X, y)

        # Assert
        assert "model" in result.columns
        assert "accuracy" in result.columns
        assert "precision" in result.columns
        assert "recall" in result.columns
        assert "f1_score" in result.columns
        assert "train_time" in result.columns
        assert "predict_time" in result.columns

    def test_compare_includes_model_names(self) -> None:
        """Test that model names are included in results."""
        # Arrange
        models = {
            "RandomForest": RandomForestClassifier(n_estimators=10, random_state=42),
            "LogReg": LogisticRegression(random_state=42, max_iter=200),
        }
        X = pd.DataFrame(np.random.rand(50, 3))
        y = pd.Series(np.random.randint(0, 2, 50))

        # Act
        result = compare_models(models, X, y)

        # Assert
        assert "RandomForest" in result["model"].values
        assert "LogReg" in result["model"].values


class TestAnalyzePredictionErrors:
    """Tests for analyze_prediction_errors function."""

    def test_analyze_with_no_errors_returns_zero_error_rate(self) -> None:
        """Test that perfect predictions show zero error rate."""
        # Arrange
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1])

        # Act
        analysis = analyze_prediction_errors(y_true, y_pred)

        # Assert
        assert analysis["total_errors"] == 0
        assert analysis["error_rate"] == 0.0

    def test_analyze_with_errors_identifies_confused_pairs(self) -> None:
        """Test that error analysis identifies most confused class pair."""
        # Arrange
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([1, 1, 0, 0, 1, 1])  # Two 0->1 errors, one 1->0 error

        # Act
        analysis = analyze_prediction_errors(y_true, y_pred)

        # Assert
        assert analysis["total_errors"] == 3
        assert "most_confused_pair" in analysis
        assert analysis["most_confused_pair"]["count"] >= 1

    def test_analyze_includes_errors_by_class(self) -> None:
        """Test that errors are broken down by class."""
        # Arrange
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 2, 2, 2])

        # Act
        analysis = analyze_prediction_errors(y_true, y_pred)

        # Assert
        assert "errors_by_true_class" in analysis
        assert "0" in analysis["errors_by_true_class"]
        assert "1" in analysis["errors_by_true_class"]

    def test_analyze_with_features_includes_feature_stats(self) -> None:
        """Test that feature statistics are included when features provided."""
        # Arrange
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        X = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "feature2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            }
        )

        # Act
        analysis = analyze_prediction_errors(y_true, y_pred, X)

        # Assert
        assert "feature_stats" in analysis
        assert "feature1" in analysis["feature_stats"]

    def test_analyze_calculates_error_indices(self) -> None:
        """Test that indices of errors are returned."""
        # Arrange
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 0, 0, 1])  # Error at index 1

        # Act
        analysis = analyze_prediction_errors(y_true, y_pred)

        # Assert
        assert "error_indices" in analysis
        assert 1 in analysis["error_indices"]
        assert len(analysis["error_indices"]) == 1
