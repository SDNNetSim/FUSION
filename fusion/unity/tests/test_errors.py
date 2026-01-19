"""Unit tests for fusion.unity.errors module."""

import pytest

from fusion.unity.errors import (
    ConfigurationError,
    JobSubmissionError,
    ManifestError,
    ManifestNotFoundError,
    ManifestValidationError,
    RemotePathError,
    SpecificationError,
    SpecNotFoundError,
    SpecValidationError,
    SynchronizationError,
    UnityError,
)


class TestUnityErrorHierarchy:
    """Tests for Unity exception class hierarchy."""

    def test_unity_error_is_exception(self) -> None:
        """Test that UnityError inherits from Exception."""
        # Arrange & Act
        error = UnityError("Test error")

        # Assert
        assert isinstance(error, Exception)

    def test_manifest_error_inherits_from_unity_error(self) -> None:
        """Test that ManifestError inherits from UnityError."""
        # Arrange & Act
        error = ManifestError("Test manifest error")

        # Assert
        assert isinstance(error, UnityError)
        assert isinstance(error, Exception)

    def test_manifest_not_found_error_inherits_from_manifest_error(self) -> None:
        """Test that ManifestNotFoundError inherits from ManifestError."""
        # Arrange & Act
        error = ManifestNotFoundError("Manifest not found")

        # Assert
        assert isinstance(error, ManifestError)
        assert isinstance(error, UnityError)

    def test_manifest_validation_error_inherits_from_manifest_error(self) -> None:
        """Test that ManifestValidationError inherits from ManifestError."""
        # Arrange & Act
        error = ManifestValidationError("Invalid manifest")

        # Assert
        assert isinstance(error, ManifestError)
        assert isinstance(error, UnityError)

    def test_specification_error_inherits_from_unity_error(self) -> None:
        """Test that SpecificationError inherits from UnityError."""
        # Arrange & Act
        error = SpecificationError("Test specification error")

        # Assert
        assert isinstance(error, UnityError)
        assert isinstance(error, Exception)

    def test_spec_not_found_error_inherits_from_specification_error(self) -> None:
        """Test that SpecNotFoundError inherits from SpecificationError."""
        # Arrange & Act
        error = SpecNotFoundError("Spec not found")

        # Assert
        assert isinstance(error, SpecificationError)
        assert isinstance(error, UnityError)

    def test_spec_validation_error_inherits_from_specification_error(self) -> None:
        """Test that SpecValidationError inherits from SpecificationError."""
        # Arrange & Act
        error = SpecValidationError("Invalid spec")

        # Assert
        assert isinstance(error, SpecificationError)
        assert isinstance(error, UnityError)

    def test_job_submission_error_inherits_from_unity_error(self) -> None:
        """Test that JobSubmissionError inherits from UnityError."""
        # Arrange & Act
        error = JobSubmissionError("Job submission failed")

        # Assert
        assert isinstance(error, UnityError)
        assert isinstance(error, Exception)

    def test_synchronization_error_inherits_from_unity_error(self) -> None:
        """Test that SynchronizationError inherits from UnityError."""
        # Arrange & Act
        error = SynchronizationError("Sync failed")

        # Assert
        assert isinstance(error, UnityError)
        assert isinstance(error, Exception)

    def test_remote_path_error_inherits_from_synchronization_error(self) -> None:
        """Test that RemotePathError inherits from SynchronizationError."""
        # Arrange & Act
        error = RemotePathError("Remote path error")

        # Assert
        assert isinstance(error, SynchronizationError)
        assert isinstance(error, UnityError)

    def test_configuration_error_inherits_from_unity_error(self) -> None:
        """Test that ConfigurationError inherits from UnityError."""
        # Arrange & Act
        error = ConfigurationError("Configuration error")

        # Assert
        assert isinstance(error, UnityError)
        assert isinstance(error, Exception)


class TestUnityErrorMessages:
    """Tests for Unity exception error messages."""

    def test_unity_error_with_message_stores_message(self) -> None:
        """Test that UnityError stores the error message."""
        # Arrange
        error_message = "Test error message"

        # Act
        error = UnityError(error_message)

        # Assert
        assert str(error) == error_message

    def test_manifest_not_found_error_with_message_stores_message(self) -> None:
        """Test that ManifestNotFoundError stores the error message."""
        # Arrange
        error_message = "Manifest file not found at /path/to/manifest.csv"

        # Act
        error = ManifestNotFoundError(error_message)

        # Assert
        assert str(error) == error_message

    def test_job_submission_error_with_message_stores_message(self) -> None:
        """Test that JobSubmissionError stores the error message."""
        # Arrange
        error_message = "SLURM submission failed with code 1"

        # Act
        error = JobSubmissionError(error_message)

        # Assert
        assert str(error) == error_message


class TestUnityErrorRaising:
    """Tests for raising Unity exceptions."""

    def test_unity_error_can_be_raised_and_caught(self) -> None:
        """Test that UnityError can be raised and caught."""
        # Arrange & Act & Assert
        with pytest.raises(UnityError) as exc_info:
            raise UnityError("Test error")

        assert "Test error" in str(exc_info.value)

    def test_manifest_not_found_error_can_be_raised_and_caught(self) -> None:
        """Test that ManifestNotFoundError can be raised and caught."""
        # Arrange & Act & Assert
        with pytest.raises(ManifestNotFoundError) as exc_info:
            raise ManifestNotFoundError("Manifest not found")

        assert "Manifest not found" in str(exc_info.value)

    def test_catching_base_error_catches_derived_errors(self) -> None:
        """Test that catching UnityError catches all derived exceptions."""
        # Arrange & Act & Assert
        with pytest.raises(UnityError):
            raise ManifestNotFoundError("Derived error")

    def test_remote_path_error_can_be_raised_and_caught(self) -> None:
        """Test that RemotePathError can be raised and caught."""
        # Arrange & Act & Assert
        with pytest.raises(RemotePathError) as exc_info:
            raise RemotePathError("Invalid remote path")

        assert "Invalid remote path" in str(exc_info.value)

    def test_configuration_error_can_be_raised_and_caught(self) -> None:
        """Test that ConfigurationError can be raised and caught."""
        # Arrange & Act & Assert
        with pytest.raises(ConfigurationError) as exc_info:
            raise ConfigurationError("Configuration missing")

        assert "Configuration missing" in str(exc_info.value)
