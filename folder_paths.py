"""
Root-level shim so `import folder_paths` resolves when running from the repo root.
Re-exports the standalone mock that sets up a local models/ directory.
"""
from src.utils.folder_paths_mock import *  # noqa: F401, F403
from src.utils.folder_paths_mock import models_dir  # noqa: F401
