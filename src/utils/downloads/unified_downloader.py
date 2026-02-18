"""
Thin HuggingFace downloader that satisfies the interface expected by ChatterboxTTS.from_pretrained().
"""
import os
from pathlib import Path

try:
    import folder_paths as _fp
    _models_dir = _fp.models_dir
except Exception:
    _models_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "models")


class UnifiedDownloader:
    """Downloads Chatterbox model files from HuggingFace Hub."""

    def download_chatterbox_model(
        self,
        repo_id: str,
        model_name: str,
        subdirectory: str | None = None,
        files: list[str] | None = None,
    ) -> str | None:
        """
        Download model files for a named language variant.

        Args:
            repo_id:      HuggingFace repo (e.g. 'ResembleAI/chatterbox')
            model_name:   Language name used as the local directory name
            subdirectory: Optional subdirectory within the repo to fetch from
            files:        List of filenames to download (downloads all if None)

        Returns:
            Local directory path where files were saved, or None on failure.
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            print("Error: huggingface_hub is not installed. Run: pip install huggingface_hub")
            return None

        local_dir = os.path.join(_models_dir, "TTS", "chatterbox", model_name)
        os.makedirs(local_dir, exist_ok=True)

        if not files:
            # Fall back to downloading the whole repo snapshot
            try:
                from huggingface_hub import snapshot_download
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=local_dir,
                    ignore_patterns=["*.md", "*.txt"],
                )
                return local_dir
            except Exception as e:
                print(f"Error downloading snapshot for {repo_id}: {e}")
                return None

        failed = []
        for fname in files:
            repo_filename = f"{subdirectory}/{fname}" if subdirectory else fname
            dest = os.path.join(local_dir, fname)
            if os.path.exists(dest):
                continue
            try:
                hf_hub_download(
                    repo_id=repo_id,
                    filename=repo_filename,
                    local_dir=local_dir,
                )
            except Exception as e:
                print(f"Warning: could not download {repo_filename} from {repo_id}: {e}")
                failed.append(fname)

        if failed:
            print(f"Warning: {len(failed)} file(s) failed to download: {failed}")

        return local_dir if os.path.isdir(local_dir) else None


# Singleton used by tts.py
unified_downloader = UnifiedDownloader()
