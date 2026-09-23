"""Locate the TensorBoard command installed in the active Python environment."""

import sys
import sysconfig
from pathlib import Path


def tensorboard_executable() -> str:
    scripts_dir = Path(sysconfig.get_path("scripts"))
    name = "tensorboard.exe" if sys.platform == "win32" else "tensorboard"
    executable = scripts_dir / name
    if not executable.is_file():
        raise FileNotFoundError(
            f"TensorBoard executable not found at {executable}. "
            "Install the dependencies from requirements-global.txt."
        )
    return str(executable)
