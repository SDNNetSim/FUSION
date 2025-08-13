import os


def create_dir(file_path: str):
    """
    Create a directory at the specified file path if it doesn't already exist.

    :param file_path: The path to the directory that should be created.
    """
    if file_path is None:
        raise ValueError("File path cannot be None.")

    abs_path = os.path.abspath(file_path)
    os.makedirs(abs_path, exist_ok=True)


def find_project_root():
    """
    Find the project root.
    """
    curr_dir = os.path.abspath(os.path.dirname(__file__))
    while True:
        if os.path.isdir(os.path.join(curr_dir, ".git")) or \
                os.path.isfile(os.path.join(curr_dir, "run_sim.py")):
            return curr_dir
        parent = os.path.dirname(curr_dir)
        if parent == curr_dir:
            raise RuntimeError("Project root not found.")
        curr_dir = parent
