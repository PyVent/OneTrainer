"""Publish complete training backups and find backups safe to resume."""

import json
import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path

COMPLETE_MARKER = ".backup-complete.json"
INCOMPLETE_PREFIX = ".incomplete-"
MARKED_BACKUP_SUFFIX = "-v2"
_WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".ckpt"}


def _has_resume_artifacts(backup_path: Path) -> bool:
    required = (
        backup_path / "meta.json",
        backup_path / "optimizer" / "optimizer.pt",
        backup_path / "onetrainer_config" / "args.json",
    )
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        return False

    try:
        with required[0].open(encoding="utf-8") as file:
            progress = json.load(file)["train_progress"]
        if not all(key in progress for key in ("epoch", "epoch_step", "epoch_sample", "global_step")):
            return False
        with required[2].open(encoding="utf-8") as file:
            if not isinstance(json.load(file), dict):
                return False
    except (OSError, ValueError, KeyError, TypeError):
        return False

    return any(
        path.is_file() and path.stat().st_size > 0 and path.suffix.lower() in _WEIGHT_SUFFIXES
        for path in backup_path.rglob("*")
        if "optimizer" not in path.relative_to(backup_path).parts
        and "ema" not in path.relative_to(backup_path).parts
    )


def is_complete_backup(backup_path: str | Path) -> bool:
    """Accept marked backups and structurally complete backups from older releases."""
    backup_path = Path(backup_path)
    if backup_path.name.startswith(INCOMPLETE_PREFIX) or not backup_path.is_dir():
        return False
    try:
        if not _has_resume_artifacts(backup_path):
            return False
    except OSError:
        return False

    marker_path = backup_path / COMPLETE_MARKER
    if not marker_path.exists():
        return not backup_path.name.endswith(MARKED_BACKUP_SUFFIX)

    try:
        with marker_path.open(encoding="utf-8") as file:
            marker = json.load(file)
        if marker.get("version") != 1 or not isinstance(marker.get("files"), dict) or not marker["files"]:
            return False
        for relative, expected_size in marker["files"].items():
            parts = Path(relative).parts
            if not parts or Path(relative).is_absolute() or ".." in parts or relative == COMPLETE_MARKER:
                return False
            path = backup_path / relative
            if not path.is_file() or path.stat().st_size != expected_size:
                return False
    except (OSError, ValueError, TypeError, AttributeError):
        return False
    return True


def complete_backup_paths(backups_path: str | Path) -> list[str]:
    backups_path = Path(backups_path)
    if not backups_path.is_dir():
        return []
    return [str(path) for path in sorted(backups_path.iterdir(), key=lambda path: path.name, reverse=True)
            if is_complete_backup(path)]


def save_backup(backups_path: str | Path, backup_name: str, write: Callable[[str], None]) -> str:
    """Write in a sibling staging directory, then publish it with one rename."""
    if not backup_name.endswith(MARKED_BACKUP_SUFFIX):
        raise ValueError(f"Backup name must end in {MARKED_BACKUP_SUFFIX}")
    backups_path = Path(backups_path)
    backups_path.mkdir(parents=True, exist_ok=True)
    destination = backups_path / backup_name
    if destination.exists():
        raise FileExistsError(destination)

    staging = Path(tempfile.mkdtemp(prefix=f"{INCOMPLETE_PREFIX}{backup_name}-", dir=backups_path))
    try:
        write(str(staging))
        if not _has_resume_artifacts(staging):
            raise ValueError(f"Backup is missing files required for resume: {staging}")

        files = {path.relative_to(staging).as_posix(): path.stat().st_size
                 for path in staging.rglob("*") if path.is_file()}
        with (staging / COMPLETE_MARKER).open("w", encoding="utf-8") as file:
            json.dump({"version": 1, "files": files}, file)
            file.flush()
            os.fsync(file.fileno())

        if destination.exists():
            raise FileExistsError(destination)
        staging.rename(destination)
        return str(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
