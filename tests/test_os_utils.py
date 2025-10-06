import unittest
from unittest.mock import patch

from fusion.utils.os import create_directory


class TestOSHelpers(unittest.TestCase):
    """
    Unit tests for the os_helpers module.
    """

    @patch('fusion.utils.os.Path')
    def test_create_directory(self, mock_path_class):
        """
        Test the create_directory function to ensure it creates the directory if it doesn't exist.
        """
        # Mock the Path class and its methods
        mock_path_instance = mock_path_class.return_value.resolve.return_value
        
        create_directory('/some/path')

        # Verify Path was called with the correct argument
        mock_path_class.assert_called_once_with('/some/path')
        # Verify resolve was called
        mock_path_class.return_value.resolve.assert_called_once()
        # Verify mkdir was called on the resolved path
        mock_path_instance.mkdir.assert_called_once_with(parents=True, exist_ok=True)

    def test_create_directory_none_path(self):
        """
        Test the create_directory function to ensure it raises a ValueError if the directory_path is None.
        """
        with self.assertRaises(ValueError) as context:
            create_directory(None)

        self.assertEqual(str(context.exception), "Directory path cannot be None")


if __name__ == '__main__':
    unittest.main()
