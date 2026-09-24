import json
import os.path
import tempfile
import threading
from contextlib import suppress
from pathlib import Path
from typing import Any

_json_replace_lock = threading.Lock()


def safe_filename(
        text: str,
        allow_spaces: bool = True,
        max_length: int | None = 32,
):
    legal_chars = [' ', '.', '_', '-', '#']
    if not allow_spaces:
        text = text.replace(' ', '_')

    text = ''.join(filter(lambda x: str.isalnum(x) or x in legal_chars, text)).strip()

    if max_length is not None:
        text = text[0: max_length]

    return text.strip()


def canonical_join(base_path: str, *paths: str):
    # Creates a canonical path name that can be used for comparisons.
    # Also, Windows does understand / instead of \, so these paths can be used as usual.

    joined = os.path.join(base_path, *paths)
    return joined.replace('\\', '/')


def write_json_atomic(path: str, obj: Any):
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=os.path.dirname(os.path.abspath(path)),
            prefix=f".{os.path.basename(path)}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = temporary_file.name
            json.dump(obj, temporary_file, indent=4)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        # Windows can reject simultaneous replacements of the same destination.
        with _json_replace_lock:
            os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            with suppress(FileNotFoundError):
                os.unlink(temporary_path)


SUPPORTED_IMAGE_EXTENSIONS = {'.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp', '.avif'}
SUPPORTED_VIDEO_EXTENSIONS = {'.webm', '.mkv', '.flv', '.avi', '.mov', '.wmv', '.mp4', '.mpeg', '.m4v'}
SUPPORTED_CAPTION_EXTENSIONS = {'.txt'}


def supported_image_extensions() -> set[str]:
    return SUPPORTED_IMAGE_EXTENSIONS


def is_supported_image_extension(extension: str) -> bool:
    return extension.lower() in SUPPORTED_IMAGE_EXTENSIONS


def supported_video_extensions() -> set[str]:
    return SUPPORTED_VIDEO_EXTENSIONS


def is_supported_video_extension(extension: str) -> bool:
    return extension.lower() in SUPPORTED_VIDEO_EXTENSIONS


def supported_caption_extensions() -> set[str]:
    return SUPPORTED_CAPTION_EXTENSIONS


def json_path_modifier(x: str | Path) -> Path:
    x = Path(x).absolute()
    return x.parent if x.suffix == ".json" else x
