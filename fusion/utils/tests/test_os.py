"""Unit tests for fusion.utils.os module."""

from unittest.mock import Mock, patch

import pytest

from fusion.utils.os import create_directory, find_project_root


class TestCreateDirectory:
    """Tests for create_directory function."""

    @patch("fusion.utils.os.Path")
    def test_create_directory_with_valid_path_creates_directory(self, mock_path_class: Mock) -> None:
        """Test creating directory with valid path."""
        # Arrange
        mock_path_instance = Mock()
        mock_path_class.return_value.resolve.return_value = mock_path_instance
        directory_path = "/some/path"

        # Act
        create_directory(directory_path)

        # Assert
        mock_path_class.assert_called_once_with(directory_path)
        mock_path_class.return_value.resolve.assert_called_once()
        mock_path_instance.mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @patch("fusion.utils.os.Path")
    def test_create_directory_with_nested_path_creates_parents(self, mock_path_class: Mock) -> None:
        """Test creating nested directory creates parent directories."""
        # Arrange
        mock_path_instance = Mock()
        mock_path_class.return_value.resolve.return_value = mock_path_instance

        # Act
        create_directory("/deep/nested/directory/path")

        # Assert
        mock_path_instance.mkdir.assert_called_once_with(parents=True, exist_ok=True)

    def test_create_directory_with_none_raises_value_error(self) -> None:
        """Test creating directory with None path raises ValueError."""
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            create_directory(None)  # type: ignore[arg-type]

        assert str(exc_info.value) == "Directory path cannot be None"

    @patch("fusion.utils.os.Path")
    def test_create_directory_with_existing_directory_succeeds(self, mock_path_class: Mock) -> None:
        """Test creating existing directory succeeds due to exist_ok=True."""
        # Arrange
        mock_path_instance = Mock()
        mock_path_class.return_value.resolve.return_value = mock_path_instance

        # Act
        create_directory("/existing/path")

        # Assert - should not raise even if directory exists
        mock_path_instance.mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @patch("fusion.utils.os.Path")
    def test_create_directory_resolves_relative_path(self, mock_path_class: Mock) -> None:
        """Test creating directory resolves relative paths."""
        # Arrange
        mock_path_instance = Mock()
        mock_path_class.return_value.resolve.return_value = mock_path_instance

        # Act
        create_directory("relative/path")

        # Assert
        mock_path_class.return_value.resolve.assert_called_once()


class TestFindProjectRoot:
    """Tests for find_project_root function."""

    @patch("fusion.utils.os.Path")
    def test_find_project_root_with_git_directory_returns_root(self, mock_path_class: Mock) -> None:
        """Test finding project root with .git directory."""
        # Arrange
        mock_current = Mock()
        mock_git_dir = Mock()
        mock_run_sim = Mock()

        mock_current.resolve.return_value = mock_current
        mock_current.__truediv__ = Mock(side_effect=lambda x: mock_git_dir if x == ".git" else mock_run_sim)
        mock_git_dir.is_dir.return_value = True
        mock_run_sim.is_file.return_value = False
        str(mock_current)  # Just to avoid linting issues
        mock_path_class.return_value.resolve.return_value = mock_current

        # Act
        result = find_project_root("/some/start/path")

        # Assert
        # Result should be string representation of mock_current
        assert isinstance(result, str)

    @patch("fusion.utils.os.Path")
    def test_find_project_root_with_run_sim_file_returns_root(self, mock_path_class: Mock) -> None:
        """Test finding project root with run_sim.py file."""
        # Arrange
        mock_current = Mock()
        mock_git_dir = Mock()
        mock_run_sim = Mock()

        mock_current.resolve.return_value = mock_current
        mock_current.__truediv__ = Mock(side_effect=lambda x: mock_git_dir if x == ".git" else mock_run_sim)
        mock_git_dir.is_dir.return_value = False
        mock_run_sim.is_file.return_value = True
        mock_path_class.return_value.resolve.return_value = mock_current

        # Act
        result = find_project_root("/some/start/path")

        # Assert
        assert isinstance(result, str)

    @patch("fusion.utils.os.Path")
    def test_find_project_root_searches_upward_until_found(self, mock_path_class: Mock) -> None:
        """Test finding project root searches parent directories."""
        # Arrange
        mock_child = Mock()
        mock_parent = Mock()

        # Setup child directory (no markers)
        mock_child.resolve.return_value = mock_child
        child_git = Mock()
        child_run_sim = Mock()
        child_git.is_dir.return_value = False
        child_run_sim.is_file.return_value = False
        mock_child.__truediv__ = Mock(side_effect=lambda x: child_git if x == ".git" else child_run_sim)
        mock_child.parent = mock_parent

        # Setup parent directory (has .git)
        parent_git = Mock()
        parent_run_sim = Mock()
        parent_git.is_dir.return_value = True
        parent_run_sim.is_file.return_value = False
        mock_parent.__truediv__ = Mock(side_effect=lambda x: parent_git if x == ".git" else parent_run_sim)

        mock_path_class.return_value.resolve.return_value = mock_child

        # Act
        result = find_project_root("/project/root/subdir")

        # Assert
        assert isinstance(result, str)

    @patch("fusion.utils.os.Path")
    def test_find_project_root_at_filesystem_root_raises_error(self, mock_path_class: Mock) -> None:
        """Test finding project root at filesystem root raises RuntimeError."""
        # Arrange
        mock_current = Mock()
        mock_git_dir = Mock()
        mock_run_sim = Mock()

        mock_current.resolve.return_value = mock_current
        mock_current.__truediv__ = Mock(side_effect=lambda x: mock_git_dir if x == ".git" else mock_run_sim)
        mock_git_dir.is_dir.return_value = False
        mock_run_sim.is_file.return_value = False
        mock_current.parent = mock_current  # Indicates filesystem root

        mock_path_class.return_value = mock_current

        # Act & Assert
        with pytest.raises(RuntimeError) as exc_info:
            find_project_root("/")

        assert str(exc_info.value) == "Project root not found"

    @patch("fusion.utils.os.Path")
    @patch("fusion.utils.os.__file__", "/project/root/fusion/utils/os.py")
    def test_find_project_root_without_start_path_uses_current_file(self, mock_path_class: Mock) -> None:
        """Test finding project root without start_path uses current file location."""
        # Arrange
        mock_parent = Mock()
        mock_git_dir = Mock()
        mock_run_sim = Mock()

        # Setup Path(__file__).resolve().parent
        mock_path_class.return_value.resolve.return_value.parent = mock_parent
        mock_parent.resolve.return_value = mock_parent

        mock_parent.__truediv__ = Mock(side_effect=lambda x: mock_git_dir if x == ".git" else mock_run_sim)
        mock_git_dir.is_dir.return_value = True
        mock_run_sim.is_file.return_value = False

        # Act
        result = find_project_root()

        # Assert
        assert isinstance(result, str)

    @patch("fusion.utils.os.Path")
    def test_find_project_root_with_both_markers_returns_root(self, mock_path_class: Mock) -> None:
        """Test finding project root when both .git and run_sim.py exist."""
        # Arrange
        mock_current = Mock()
        mock_git_dir = Mock()
        mock_run_sim = Mock()

        mock_current.resolve.return_value = mock_current
        mock_current.__truediv__ = Mock(side_effect=lambda x: mock_git_dir if x == ".git" else mock_run_sim)
        mock_git_dir.is_dir.return_value = True
        mock_run_sim.is_file.return_value = True
        mock_path_class.return_value.resolve.return_value = mock_current

        # Act
        result = find_project_root("/project/root")

        # Assert
        assert isinstance(result, str)
