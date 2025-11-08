import os


def set_cwd_project_root():
    """Set the working directory to the project root."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    os.chdir(project_root)
