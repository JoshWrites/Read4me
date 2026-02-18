"""
Mock folder_paths for ChatterBox compatibility
Provides ComfyUI folder_paths interface when running standalone
"""

import os
from pathlib import Path


class MockFolderPaths:
    """Mock folder_paths for standalone use"""
    
    def __init__(self):
        # Use a local models directory
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.models_dir = os.path.join(base_dir, "models")
        
        # Create directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
    
    @property
    def models_dir(self):
        return self._models_dir
    
    @models_dir.setter
    def models_dir(self, value):
        self._models_dir = value


# Create singleton instance
_folder_paths = MockFolderPaths()

# Provide same interface as ComfyUI's folder_paths
models_dir = _folder_paths.models_dir

