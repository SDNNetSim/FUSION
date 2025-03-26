import os


def create_dir(file_path: str):
    """
    Create a directory at the specified file path if it doesn't already exist.

    :param file_path: The path to the directory that should be created.
    """
    if file_path is None:
        raise ValueError("File path cannot be None.")
    if file_path == "":
        raise ValueError("File path cannot be empty.")
    abs_path = os.path.abspath(file_path)
    os.makedirs(abs_path, exist_ok=True)
