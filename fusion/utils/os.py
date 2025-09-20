"""Operating system utility functions for FUSION."""

import os


def create_directory(directory_path: str) -> None:
    """Create a directory at the specified path if it doesn't already exist.

    Creates all intermediate directories as needed. If the directory already
    exists, this function does nothing.

    :param directory_path: The path to the directory that should be created
    :type directory_path: str
    :raises ValueError: If directory_path is None

    Example:
        >>> create_directory("/path/to/new/directory")
        # Creates the directory and any missing parent directories
    """
    if directory_path is None:
        raise ValueError("Directory path cannot be None")

    absolute_path = os.path.abspath(directory_path)
    os.makedirs(absolute_path, exist_ok=True)


def find_project_root(start_path: str | None = None) -> str:
    """Find the project root directory by looking for git or project markers.

    Searches upward from the given start path (or current file location)
    until it finds a directory containing either a .git directory or
    run_sim.py file, which indicates the project root.

    :param start_path: Directory to start searching from, defaults to current file location
    :type start_path: Optional[str]
    :return: Absolute path to the project root directory
    :rtype: str
    :raises RuntimeError: If project root cannot be found

    Example:
        >>> root = find_project_root()
        >>> print(root)  # /path/to/FUSION
    """
    if start_path is None:
        current_directory = os.path.abspath(os.path.dirname(__file__))
    else:
        current_directory = os.path.abspath(start_path)

    while True:
        git_directory = os.path.join(current_directory, ".git")
        run_sim_file = os.path.join(current_directory, "run_sim.py")

        if os.path.isdir(git_directory) or os.path.isfile(run_sim_file):
            return current_directory

        parent_directory = os.path.dirname(current_directory)
        if parent_directory == current_directory:
            raise RuntimeError("Project root not found")
        current_directory = parent_directory
