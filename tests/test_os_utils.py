import unittest
from unittest.mock import patch

from fusion.utils.os import create_directory


class TestOSHelpers(unittest.TestCase):
    """
    Unit tests for the os_helpers module.
    """

    @patch('os.makedirs')
    @patch('os.path.abspath')
    def test_create_directory(self, mock_abspath, mock_makedirs):
        """
        Test the create_directory function to ensure it creates the directory if it doesn't exist.
        """
        # Mock the absolute path to avoid creating directories during the test
        mock_abspath.return_value = '/mocked/abs/path'

        create_directory('/some/path')

        mock_abspath.assert_called_once_with('/some/path')
        mock_makedirs.assert_called_once_with('/mocked/abs/path', exist_ok=True)

    def test_create_directory_none_path(self):
        """
        Test the create_directory function to ensure it raises a ValueError if the directory_path is None.
        """
        with self.assertRaises(ValueError) as context:
            create_directory(None)

        self.assertEqual(str(context.exception), "Directory path cannot be None")


if __name__ == '__main__':
    unittest.main()
